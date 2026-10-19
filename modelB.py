"""
Model B - Lightweight CNN (Depthwise Separable Convolutions)

Assignment items covered by this script
  1. Data preparation : 64x64 images, stratified 70/15/15 split (identical to
                        Model A: same seed -> same split)
  2. Architecture     : per-layer parameter table, trainable-parameter count
                        (hard limit: <= 100,000) and MACs
  3. Optimizers       : Adam vs SGD vs SGD+Momentum (plus a momentum sweep and a
                        convergence/momentum analysis)
  4. Training/eval    : >= 20 epochs, loss curves, accuracy, confusion matrix,
                        precision, recall, model size, time per epoch

Usage examples
  # just run it: 50 epochs, no augmentation, Adam vs SGD vs SGD+Momentum(0.9)
  python modelB.py

  # full momentum sweep (0, 0.5, 0.9) plus both Adam learning rates
  python modelB.py --optimizers all

  # only Adam (single run, no comparison)
  python modelB.py --optimizers adam

Every run records the exact command line and all arguments in
run_metadata.json and command.txt inside the results folder.

Architecture idea
  * Stem: one standard 3x3 conv (only 3 input channels, so a depthwise
    separable conv would save almost nothing and hurt accuracy; this is the
    same convention as MobileNet). Every other conv is a SeparableConv2D
    (3x3 depthwise + 1x1 pointwise).
  * Parameters of a conv layer, k x k kernel, Cin -> Cout:
        standard  : k*k*Cin*Cout
        separable : k*k*Cin + Cin*Cout     (ratio ~ 1/Cout + 1/(k*k))
    For 3x3 kernels and Cout >= 64 this is roughly an 8-9x reduction in both
    parameters and MACs.
  * Classifier head: GlobalAveragePooling2D instead of Flatten. Model A's
    Flatten->Dense(128) layer alone holds ~1M parameters; global average
    pooling removes it entirely and is what makes the 100k budget possible.

Hardware-aware activation choice
  * ReLU6 = min(max(0, x), 6) in the conv blocks: two comparisons, no
    exponential, no division. The bounded output range (0..6) is what makes
    int8 quantisation robust (MobileNet convention). Plain ReLU in the dense
    layer.
  * Sigmoid / tanh / swish / GELU / ELU all need exp(); softmax too.
  * The output layer emits raw logits (no softmax). The softmax exponential is
    only needed for training; at inference argmax(logits) == argmax(softmax),
    so the exp() is never computed on the device.
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)

# ----------------------------------------------------------------------------
# Optimizer registry (identical to Model A so the comparison is like-for-like)
# ----------------------------------------------------------------------------
OPTIMIZER_REGISTRY = {
    "adam": {"label": "Adam (lr=1e-4)",
             "make": lambda: keras.optimizers.Adam(learning_rate=1e-4)},
    "adam_1e-3": {"label": "Adam (lr=1e-3)",
                  "make": lambda: keras.optimizers.Adam(learning_rate=1e-3)},
    "sgd": {"label": "SGD (lr=0.01, momentum=0)",
            "make": lambda: keras.optimizers.SGD(learning_rate=1e-2, momentum=0.0)},
    "sgd_m0.5": {"label": "SGD + Momentum 0.5 (lr=0.01)",
                 "make": lambda: keras.optimizers.SGD(learning_rate=1e-2, momentum=0.5)},
    "sgd_m0.9": {"label": "SGD + Momentum 0.9 (lr=0.01)",
                 "make": lambda: keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9)},
}
DEFAULT_OPTIMIZERS = "adam,sgd,sgd_m0.9"  # the three the assignment asks for
SGD_FAMILY = ["sgd", "sgd_m0.5", "sgd_m0.9"]

# ----------------------------------------------------------------------------
# Command-line configuration (env vars still work as defaults)
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Model B (depthwise separable CNN) training with optimizer comparison.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--aug", choices=["none", "geometric", "full"],
                    default=os.environ.get("AUG_MODE", "none"),
                    help="training-only augmentation (keep equal to Model A "
                         "for a fair comparison)")
parser.add_argument("--epochs", type=int,
                    default=int(os.environ.get("EPOCHS", 50)),
                    help="epochs per optimizer (assignment requires >= 20)")
parser.add_argument("--optimizers",
                    default=os.environ.get("OPT_KEYS") or DEFAULT_OPTIMIZERS,
                    help="comma-separated keys from "
                         f"{list(OPTIMIZER_REGISTRY)} or 'all'")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", default=os.environ.get("DATASET_DIR"),
                    help="dataset root (one sub-folder per class); "
                         "default: <script dir>/dataset6")
parser.add_argument("--tag", default="modelB", help="results folder name")
parser.add_argument("--no-cache", action="store_true",
                    help="do not cache decoded images in RAM")
args = parser.parse_args()

if args.epochs < 20:
    print(f"WARNING: --epochs={args.epochs} is below the assignment minimum of 20.")

if args.optimizers.strip().lower() == "all":
    opt_keys = list(OPTIMIZER_REGISTRY)
else:
    opt_keys = [k.strip() for k in args.optimizers.split(",") if k.strip()]
unknown = [k for k in opt_keys if k not in OPTIMIZER_REGISTRY]
if unknown:
    parser.error(f"unknown optimizer key(s) {unknown}; "
                 f"choose from {list(OPTIMIZER_REGISTRY)} or 'all'")
opt_keys = list(dict.fromkeys(opt_keys))  # de-duplicate, keep order
OPTIMIZERS = [{"key": k, **OPTIMIZER_REGISTRY[k]} for k in opt_keys]

MODEL_TAG = args.tag
MODEL_LABEL = "Model B (Lightweight CNN, depthwise separable)"
PARAM_LIMIT = 100_000  # assignment: trainable parameters of Model B <= 100,000

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(args.dataset) if args.dataset else PROJECT_DIR / "dataset6"
IMG_SIZE = (64, 64)
BATCH_SIZE = args.batch_size
SEED = args.seed
EPOCHS = args.epochs
AUG_MODE = args.aug
USE_CACHE = not args.no_cache
RESULTS_DIR = PROJECT_DIR / "results" / f"{MODEL_TAG}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COMMAND_LINE = "python " + " ".join(
    [Path(sys.argv[0]).name] + sys.argv[1:])
print(f"\nCommand : {COMMAND_LINE}")
print(f"Arguments: aug={AUG_MODE} epochs={EPOCHS} optimizers={opt_keys} "
      f"batch={BATCH_SIZE} seed={SEED} cache={USE_CACHE}")
(RESULTS_DIR / "command.txt").write_text(
    COMMAND_LINE + "\n" + json.dumps(vars(args), indent=2) + "\n",
    encoding="utf-8")

# ----------------------------------------------------------------------------
# Data preparation (same code and seed as Model A -> identical splits)
# ----------------------------------------------------------------------------
class_names = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())
class_to_idx = {name: i for i, name in enumerate(class_names)}
num_classes = len(class_names)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
paths, labels = [], []
for class_name in class_names:
    for path in sorted((DATASET_DIR / class_name).rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            paths.append(str(path))
            labels.append(class_to_idx[class_name])

if not paths:
    raise ValueError(f"No images found in {DATASET_DIR}")

class_counts = Counter(labels)
print("\nImages per class:")
for name, idx in class_to_idx.items():
    print(f"{name:30s}: {class_counts[idx]}")
print(f"\nTotal images: {len(paths)}")
print(f"Number of classes: {num_classes}")

# Stratified 70% / 15% / 15% split
X_train, X_temp, y_train, y_temp = train_test_split(
    paths, labels, test_size=0.30, stratify=labels, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)

print(f"\nTraining:   {len(X_train)} ({len(X_train)/len(paths):.1%})")
print(f"Validation: {len(X_val)} ({len(X_val)/len(paths):.1%})")
print(f"Testing:    {len(X_test)} ({len(X_test)/len(paths):.1%})")


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)  # downscale to 64x64
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def build_augmentation():
    """Train-only augmentation (no trainable parameters, not part of the model).
    geometric: flips + mild rotation/zoom (colour is left untouched, because
    colour is a key cue for glass / vegetation / cardboard).
    full: adds translation, contrast and brightness jitter."""
    ops = [layers.RandomFlip("horizontal_and_vertical"),
           layers.RandomRotation(0.10, fill_mode="reflect"),
           layers.RandomZoom(0.10, fill_mode="reflect")]
    if AUG_MODE == "full":
        ops += [layers.RandomTranslation(0.10, 0.10, fill_mode="reflect"),
                layers.RandomContrast(0.20),
                layers.RandomBrightness(0.20, value_range=(0.0, 1.0))]
    return keras.Sequential(ops, name="train_augmentation")


augmentation = build_augmentation()


def make_dataset(paths_, labels_, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths_, labels_))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if USE_CACHE:
        ds = ds.cache()  # cache the clean decoded images; augment afterwards
    if training:
        ds = ds.shuffle(len(paths_), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE)
    if training and AUG_MODE != "none":
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(X_train, y_train, training=True)
val_ds = make_dataset(X_val, y_val)
test_ds = make_dataset(X_test, y_test)


# ----------------------------------------------------------------------------
# ===== MODEL DEFINITION (START) =====
# ----------------------------------------------------------------------------
def sep_block(filters):
    """Depthwise 3x3 + pointwise 1x1 (SeparableConv2D) -> BN -> ReLU6."""
    return [layers.SeparableConv2D(filters, kernel_size=(3, 3), padding="same",
                                   use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(max_value=6.0)]


def build_model():
    """Model B: depthwise-separable CNN, logits output.

    Hand-computed trainable parameters (BN counts gamma+beta = 2*C):
      stem   Conv2D 3x3 3->16            432  + BN 32           =    464
      S1     Sep 16->32     144+512=656       + BN 64           =    720
      S2     Sep 32->64    288+2048=2,336     + BN 128
             Sep 64->64    576+4096=4,672     + BN 128          =  7,264
      S3     Sep 64->128  576+8192=8,768      + BN 256
             Sep 128->128 1152+16384=17,536   + BN 256          = 26,816
      S4     Sep 128->256 1152+32768=33,920   + BN 512          = 34,432
      head   Dense 256->64 (16,448) + Dense 64->9 (585)         = 17,033
      TOTAL trainable                                           = 86,729
    (the script recomputes this from the real model and asserts <= 100,000)
    """
    return keras.Sequential([
        layers.Input(shape=(64, 64, 3)),

        # Stem: standard 3x3 conv, 16 filters (3 input channels only)
        layers.Conv2D(16, kernel_size=(3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(max_value=6.0),
        layers.MaxPooling2D(pool_size=(2, 2)),          # 32x32x16

        # Stage 1: 1 separable conv, 32 filters
        *sep_block(32),
        layers.MaxPooling2D(pool_size=(2, 2)),          # 16x16x32

        # Stage 2: 2 separable convs, 64 filters
        *sep_block(64),
        *sep_block(64),
        layers.MaxPooling2D(pool_size=(2, 2)),          # 8x8x64
        layers.Dropout(0.10),

        # Stage 3: 2 separable convs, 128 filters
        *sep_block(128),
        *sep_block(128),
        layers.MaxPooling2D(pool_size=(2, 2)),          # 4x4x128
        layers.Dropout(0.10),

        # Stage 4: 1 separable conv, 256 filters
        *sep_block(256),                                # 4x4x256

        # Head: GAP -> FC(64) -> FC(num_classes)
        layers.GlobalAveragePooling2D(),                # 256
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(num_classes),  # logits; softmax folded into the loss
    ], name="ModelB_Lightweight_SeparableCNN")
# ----------------------------------------------------------------------------
# ===== MODEL DEFINITION (END) =====
# ----------------------------------------------------------------------------


def layer_breakdown(model):
    """Per-layer output shape, parameters (total / trainable) and MACs."""
    rows = []
    for layer in model.layers:
        out_shape = tuple(layer.output.shape[1:])
        in_shape = tuple(layer.input.shape[1:])
        trainable = sum(int(np.prod(w.shape)) for w in layer.trainable_weights)
        total = layer.count_params()
        macs = 0
        if isinstance(layer, layers.SeparableConv2D):
            k = layer.kernel_size[0] * layer.kernel_size[1]
            cin, cout = in_shape[-1], layer.filters
            macs = out_shape[0] * out_shape[1] * (k * cin + cin * cout)
        elif isinstance(layer, layers.Conv2D):
            k = layer.kernel_size[0] * layer.kernel_size[1]
            cin, cout = in_shape[-1], layer.filters
            macs = out_shape[0] * out_shape[1] * k * cin * cout
        elif isinstance(layer, layers.Dense):
            macs = in_shape[-1] * out_shape[-1]
        rows.append({
            "layer": layer.name,
            "type": layer.__class__.__name__,
            "output_shape": list(out_shape),
            "total_params": int(total),
            "trainable_params": int(trainable),
            "macs": int(macs),
        })
    return rows


def format_breakdown(rows):
    header = f"{'Layer':28s}{'Type':22s}{'Output shape':18s}{'Params':>10s}{'Trainable':>12s}{'MACs':>14s}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['layer']:28s}{r['type']:22s}{str(tuple(r['output_shape'])):18s}"
            f"{r['total_params']:>10,}{r['trainable_params']:>12,}{r['macs']:>14,}")
    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL':28s}{'':22s}{'':18s}"
        f"{sum(r['total_params'] for r in rows):>10,}"
        f"{sum(r['trainable_params'] for r in rows):>12,}"
        f"{sum(r['macs'] for r in rows):>14,}")
    return "\n".join(lines)


class EpochTimer(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(float(time.perf_counter() - self.t0))


def first_epoch_reaching(values, threshold, mode="ge"):
    """1-based epoch at which `values` first reaches the threshold, else None."""
    for i, v in enumerate(values):
        if (mode == "ge" and v >= threshold) or (mode == "le" and v <= threshold):
            return i + 1
    return None


def run_experiment(opt_cfg):
    """Train a fresh Model B with one optimizer for EPOCHS epochs (no early
    stopping, so every run completes >= 20 epochs), restore the best
    validation-loss weights, then evaluate on the test set."""
    print(f"\n{'=' * 70}\nTraining {MODEL_LABEL} with {opt_cfg['label']}\n{'=' * 70}")
    keras.utils.set_random_seed(SEED)  # identical initial weights per optimizer
    model = build_model()
    model.compile(
        optimizer=opt_cfg["make"](),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    weights_path = RESULTS_DIR / f"best_{opt_cfg['key']}.weights.h5"
    timer = EpochTimer()
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(weights_path), monitor="val_loss", mode="min",
            save_best_only=True, save_weights_only=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6,
            mode="min", verbose=1),
        timer,
    ]

    t0 = time.perf_counter()
    history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds,
                        callbacks=callbacks, verbose=2)
    total_time = time.perf_counter() - t0

    if weights_path.exists():
        model.load_weights(weights_path)  # best validation-loss weights

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    logits = model.predict(test_ds, verbose=0)
    preds = logits.argmax(axis=1)

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    val_acc = history.history["val_accuracy"]
    best_vl = float(min(val_loss))
    last10 = val_loss[-10:]
    # The first epoch includes XLA/graph compilation, so the median (and the
    # mean excluding epoch 1) are the fair "time per epoch" figures.
    et = timer.epoch_times
    result = {
        "key": opt_cfg["key"],
        "label": opt_cfg["label"],
        "epochs_completed": len(val_loss),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "test_f1_weighted": float(f1_score(y_test, preds, average="weighted", zero_division=0)),
        "test_precision_macro": float(precision_score(y_test, preds, average="macro", zero_division=0)),
        "test_recall_macro": float(recall_score(y_test, preds, average="macro", zero_division=0)),
        "test_confusion_matrix": confusion_matrix(
            y_test, preds, labels=list(range(num_classes))).tolist(),
        "best_validation_loss": best_vl,
        "best_validation_loss_epoch": int(np.argmin(val_loss)) + 1,
        "best_validation_accuracy": float(max(val_acc)),
        "best_validation_accuracy_epoch": int(np.argmax(val_acc)) + 1,
        "final_training_loss": float(train_loss[-1]),
        "final_training_accuracy": float(history.history["accuracy"][-1]),
        "final_validation_loss": float(val_loss[-1]),
        "final_validation_accuracy": float(val_acc[-1]),
        # convergence-speed indicators used in the momentum analysis
        "epochs_to_val_acc_60": first_epoch_reaching(val_acc, 0.60),
        "epochs_to_val_acc_70": first_epoch_reaching(val_acc, 0.70),
        "epochs_to_val_loss_1.0": first_epoch_reaching(val_loss, 1.0, "le"),
        "epochs_to_within_5pct_best_val_loss": first_epoch_reaching(
            val_loss, best_vl * 1.05, "le"),
        "train_loss_epoch_5": float(train_loss[4]) if len(train_loss) >= 5 else None,
        "train_loss_epoch_10": float(train_loss[9]) if len(train_loss) >= 10 else None,
        "val_loss_std_last10": float(np.std(last10)),
        "generalisation_gap_acc": float(
            history.history["accuracy"][-1] - val_acc[-1]),
        "total_training_time_seconds": float(total_time),
        "average_epoch_time_seconds": float(np.mean(et)),
        "average_epoch_time_excl_first_seconds": float(np.mean(et[1:])) if len(et) > 1 else float(np.mean(et)),
        "median_epoch_time_seconds": float(np.median(et)),
        "epoch_time_seconds": et,
        "history": hist,
    }
    print(f"{opt_cfg['label']}: test acc {test_acc:.4f} | macro F1 "
          f"{result['test_f1_macro']:.4f} | best val loss "
          f"{best_vl:.4f} @ epoch {result['best_validation_loss_epoch']} | "
          f"{result['median_epoch_time_seconds']:.2f}s/epoch (median)")
    return model, preds, result, weights_path


# ----------------------------------------------------------------------------
# Architecture report (parameters, MACs, size)
# ----------------------------------------------------------------------------
keras.utils.set_random_seed(SEED)
arch_model = build_model()
arch_model.summary()
rows = layer_breakdown(arch_model)
breakdown_text = format_breakdown(rows)
print("\n" + breakdown_text)

trainable_params = sum(r["trainable_params"] for r in rows)
total_params = sum(r["total_params"] for r in rows)
non_trainable_params = total_params - trainable_params
total_macs = sum(r["macs"] for r in rows)
float32_size_kb = total_params * 4 / 1024

print(f"\nTrainable parameters:     {trainable_params:,}")
print(f"Non-trainable parameters: {non_trainable_params:,} (BatchNorm moving statistics)")
print(f"Total parameters:         {total_params:,}")
print(f"MACs per inference:       {total_macs:,}")
print(f"Float32 weight size:      {float32_size_kb:.1f} KB")

assert trainable_params <= PARAM_LIMIT, (
    f"{MODEL_LABEL} exceeds the {PARAM_LIMIT:,} trainable-parameter "
    f"limit: {trainable_params:,}")
print(f"Parameter constraint satisfied: {trainable_params:,} <= {PARAM_LIMIT:,} "
      f"({trainable_params / PARAM_LIMIT:.1%} of budget).")

(RESULTS_DIR / "layer_breakdown.txt").write_text(breakdown_text, encoding="utf-8")
(RESULTS_DIR / "layer_breakdown.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
del arch_model

# ----------------------------------------------------------------------------
# Run all optimizer experiments
# ----------------------------------------------------------------------------
results, runs = {}, {}
for cfg in OPTIMIZERS:
    model, preds, res, weights_path = run_experiment(cfg)
    results[cfg["key"]] = res
    runs[cfg["key"]] = (model, preds)
    weights_path.unlink(missing_ok=True)

# Select the optimizer by VALIDATION loss (never by test accuracy)
MAIN_OPTIMIZER_KEY = min(results, key=lambda k: results[k]["best_validation_loss"])
main_model, main_preds = runs[MAIN_OPTIMIZER_KEY]
main_res = results[MAIN_OPTIMIZER_KEY]
print(f"\nSelected optimizer (lowest validation loss): {main_res['label']}")

# ----------------------------------------------------------------------------
# Save artifacts for the chosen (main) optimizer
# ----------------------------------------------------------------------------
model_path = RESULTS_DIR / "model.keras"
main_model.save(model_path)
disk_size_kb = model_path.stat().st_size / 1024
print(f"\nSaved model file size: {disk_size_kb:.1f} KB")

report_dict = classification_report(
    y_test, main_preds, labels=list(range(num_classes)),
    target_names=class_names, output_dict=True, zero_division=0)
report_text = classification_report(
    y_test, main_preds, labels=list(range(num_classes)),
    target_names=class_names, zero_division=0)
print("\n" + report_text)

cm = confusion_matrix(y_test, main_preds, labels=list(range(num_classes)))

final_metrics = {
    "command": COMMAND_LINE,
    "model": MODEL_LABEL,
    "augmentation": AUG_MODE,
    "optimizer": main_res["label"],
    "test_loss": main_res["test_loss"],
    "test_accuracy": main_res["test_accuracy"],
    "test_precision_macro": main_res["test_precision_macro"],
    "test_recall_macro": main_res["test_recall_macro"],
    "test_f1_macro": main_res["test_f1_macro"],
    "test_f1_weighted": main_res["test_f1_weighted"],
    "test_classification_report": report_dict,
    "confusion_matrix": cm.tolist(),
    "trainable_parameters": int(trainable_params),
    "non_trainable_parameters": int(non_trainable_params),
    "total_parameters": int(total_params),
    "parameter_limit": PARAM_LIMIT,
    "macs_per_inference": int(total_macs),
    "float32_weight_size_kb": float(float32_size_kb),
    "model_file_size_kb": float(disk_size_kb),
    "epochs_completed": main_res["epochs_completed"],
    "best_validation_loss": main_res["best_validation_loss"],
    "best_validation_loss_epoch": main_res["best_validation_loss_epoch"],
    "best_validation_accuracy": main_res["best_validation_accuracy"],
    "best_validation_accuracy_epoch": main_res["best_validation_accuracy_epoch"],
    "final_training_loss": main_res["final_training_loss"],
    "final_training_accuracy": main_res["final_training_accuracy"],
    "final_validation_loss": main_res["final_validation_loss"],
    "final_validation_accuracy": main_res["final_validation_accuracy"],
    "total_training_time_seconds": main_res["total_training_time_seconds"],
    "average_epoch_time_seconds": main_res["average_epoch_time_seconds"],
    "average_epoch_time_excl_first_seconds": main_res["average_epoch_time_excl_first_seconds"],
    "median_epoch_time_seconds": main_res["median_epoch_time_seconds"],
}
(RESULTS_DIR / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
(RESULTS_DIR / "training_history.json").write_text(
    json.dumps({**main_res["history"], "epoch_time_seconds": main_res["epoch_time_seconds"]}, indent=2),
    encoding="utf-8")
(RESULTS_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")

(RESULTS_DIR / "run_metadata.json").write_text(json.dumps({
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "command": COMMAND_LINE,
    "arguments": vars(args),
    "model": MODEL_LABEL,
    "dataset": str(DATASET_DIR),
    "image_size": list(IMG_SIZE),
    "batch_size": BATCH_SIZE,
    "epochs_requested": EPOCHS,
    "seed": SEED,
    "optimizers_compared": [r["label"] for r in results.values()],
    "main_optimizer": main_res["label"],
    "loss": "SparseCategoricalCrossentropy(from_logits=True)",
    "augmentation": AUG_MODE,
    "lr_schedule": "ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6)",
    "class_names": class_names,
    "class_to_idx": class_to_idx,
    "num_images": len(paths),
    "train_images": len(X_train),
    "validation_images": len(X_val),
    "test_images": len(X_test),
}, indent=2), encoding="utf-8")

# ----------------------------------------------------------------------------
# Optimizer comparison table (Adam vs SGD vs SGD+Momentum)
# ----------------------------------------------------------------------------
def fmt_ep(v):
    return "-" if v is None else str(v)


lines = [
    f"{MODEL_LABEL} [aug={AUG_MODE}]: optimizer comparison "
    f"({EPOCHS} epochs each, same seed/init)",
    f"Command: {COMMAND_LINE}",
    "",
    f"{'Optimizer':34s}{'BestValLoss':>12s}{'@Ep':>5s}{'ValAcc':>8s}{'TestAcc':>9s}"
    f"{'MacroF1':>9s}{'Prec':>7s}{'Rec':>7s}{'s/ep(med)':>10s}",
    "-" * 101,
]
for r in results.values():
    lines.append(
        f"{r['label']:34s}{r['best_validation_loss']:>12.4f}"
        f"{r['best_validation_loss_epoch']:>5d}{r['best_validation_accuracy']:>8.4f}"
        f"{r['test_accuracy']:>9.4f}{r['test_f1_macro']:>9.4f}"
        f"{r['test_precision_macro']:>7.3f}{r['test_recall_macro']:>7.3f}"
        f"{r['median_epoch_time_seconds']:>10.2f}")
comparison_text = "\n".join(lines)
print("\n" + comparison_text)
(RESULTS_DIR / "optimizer_comparison.txt").write_text(comparison_text, encoding="utf-8")
(RESULTS_DIR / "optimizer_comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

# ----------------------------------------------------------------------------
# Convergence / momentum analysis
# ----------------------------------------------------------------------------
conv_lines = [
    f"Convergence analysis [{MODEL_LABEL}, aug={AUG_MODE}] (lower epoch numbers = faster)",
    "",
    f"{'Optimizer':34s}{'Ep->70%acc':>11s}{'Ep->60%acc':>11s}{'Ep->vl<1.0':>11s}"
    f"{'Ep->5%best':>11s}{'TrL@5':>8s}{'TrL@10':>8s}{'ValLossStd10':>14s}{'TrainVal gap':>14s}",
    "-" * 122,
]
for r in results.values():
    t5 = "-" if r["train_loss_epoch_5"] is None else f"{r['train_loss_epoch_5']:.3f}"
    t10 = "-" if r["train_loss_epoch_10"] is None else f"{r['train_loss_epoch_10']:.3f}"
    conv_lines.append(
        f"{r['label']:34s}{fmt_ep(r['epochs_to_val_acc_70']):>11s}"
        f"{fmt_ep(r['epochs_to_val_acc_60']):>11s}{fmt_ep(r['epochs_to_val_loss_1.0']):>11s}"
        f"{fmt_ep(r['epochs_to_within_5pct_best_val_loss']):>11s}{t5:>8s}{t10:>8s}"
        f"{r['val_loss_std_last10']:>14.4f}{r['generalisation_gap_acc']:>14.3f}")

present_sgd = [k for k in SGD_FAMILY if k in results]
if len(present_sgd) >= 2:
    conv_lines += ["", "Momentum effect (SGD family, lr=0.01):"]
    base = results[present_sgd[0]]
    for k in present_sgd[1:]:
        r = results[k]
        d_loss = base["best_validation_loss"] - r["best_validation_loss"]
        d_acc = r["test_accuracy"] - base["test_accuracy"]
        conv_lines.append(
            f"  {r['label']} vs {base['label']}: best val loss "
            f"{'lower' if d_loss > 0 else 'higher'} by {abs(d_loss):.4f}, "
            f"test accuracy {'+' if d_acc >= 0 else '-'}{abs(d_acc) * 100:.2f} pts, "
            f"epochs to 70% val acc: {fmt_ep(r['epochs_to_val_acc_70'])} vs "
            f"{fmt_ep(base['epochs_to_val_acc_70'])}")
    conv_lines += [
        "",
        "Interpretation guide: momentum m accumulates a velocity v <- m*v - lr*grad,",
        "so the effective step in a consistent gradient direction is ~lr/(1-m)",
        "(x2 for m=0.5, x10 for m=0.9). This speeds up progress along shallow,",
        "consistent directions and damps oscillation across steep ones, but a high",
        "m can overshoot and make validation loss noisier (see ValLossStd10).",
    ]
if any(k.startswith("adam") for k in results):
    conv_lines += [
        "",
        "Adam keeps per-parameter running estimates of the gradient mean and",
        "squared gradient, giving each weight its own adaptive step size; this is",
        "why it usually needs far less learning-rate tuning than plain SGD.",
    ]
convergence_text = "\n".join(conv_lines)
print("\n" + convergence_text)
(RESULTS_DIR / "convergence_analysis.txt").write_text(convergence_text, encoding="utf-8")

# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
# Class distribution
plt.figure(figsize=(12, 6))
plt.bar(class_names, [class_counts[class_to_idx[n]] for n in class_names])
plt.title("Images per Class")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=150)
plt.close()

# Training / validation loss and accuracy (chosen optimizer)
epochs_axis = range(1, main_res["epochs_completed"] + 1)
h = main_res["history"]

plt.figure(figsize=(10, 6))
plt.plot(epochs_axis, h["loss"], label="Training Loss")
plt.plot(epochs_axis, h["val_loss"], label="Validation Loss")
plt.title(f"{MODEL_LABEL}: Training and Validation Loss ({main_res['label']})")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "loss_curve.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(epochs_axis, h["accuracy"], label="Training Accuracy")
plt.plot(epochs_axis, h["val_accuracy"], label="Validation Accuracy")
plt.title(f"{MODEL_LABEL}: Training and Validation Accuracy ({main_res['label']})")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "accuracy_curve.png", dpi=150)
plt.close()

# Optimizer comparison curves (train loss, val loss, val acc) + test accuracy bars
if len(results) > 1:
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    for r in results.values():
        ep = range(1, r["epochs_completed"] + 1)
        axes[0].plot(ep, r["history"]["loss"], label=r["label"])
        axes[1].plot(ep, r["history"]["val_loss"], label=r["label"])
        axes[2].plot(ep, r["history"]["val_accuracy"], label=r["label"])
    for ax, title, ylab in zip(
            axes[:3],
            ["Training loss", "Validation loss", "Validation accuracy"],
            ["Loss", "Loss", "Accuracy"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    short = [r["label"].replace(" (", "\n(") for r in results.values()]
    accs = [r["test_accuracy"] for r in results.values()]
    bars = axes[3].bar(range(len(accs)), accs)
    axes[3].set_xticks(range(len(accs)))
    axes[3].set_xticklabels(short, fontsize=7)
    axes[3].set_ylim(max(0.0, min(accs) - 0.1), min(1.0, max(accs) + 0.05))
    axes[3].set_title("Test accuracy")
    axes[3].grid(axis="y", alpha=0.3)
    for b_, a_ in zip(bars, accs):
        axes[3].text(b_.get_x() + b_.get_width() / 2, a_, f"{a_:.3f}",
                     ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"{MODEL_LABEL} [aug={AUG_MODE}]: optimizer comparison")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "optimizer_comparison.png", dpi=150)
    plt.close()

# Momentum effect plot (SGD family only)
if len(present_sgd) >= 2:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for k in present_sgd:
        r = results[k]
        ep = range(1, r["epochs_completed"] + 1)
        axes[0].plot(ep, r["history"]["loss"], label=r["label"])
        axes[1].plot(ep, r["history"]["val_loss"], label=r["label"])
        axes[2].plot(ep, r["history"]["val_accuracy"], label=r["label"])
    for ax, title in zip(axes, ["Training loss", "Validation loss", "Validation accuracy"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Effect of momentum on SGD convergence ({MODEL_LABEL}, lr=0.01)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "momentum_effect.png", dpi=150)
    plt.close()

# Confusion matrix (chosen optimizer)
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
    ax=ax, cmap="Blues", xticks_rotation=45, values_format="d", colorbar=False)
ax.set_title(f"{MODEL_LABEL}: Test Confusion Matrix ({main_res['label']})")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close()

print(f"\nResults saved to: {RESULTS_DIR}")