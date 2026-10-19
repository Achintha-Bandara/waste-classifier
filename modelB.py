"""
Model B - Lightweight CNN (Depthwise Separable Convolutions)

Assignment items covered by this script
  1. Data preparation : 64x64 images, stratified 70/15/15 split
  2. Architecture     : per-layer parameter table + trainable-parameter count + MACs
  3. Optimizers       : Adam vs SGD vs SGD+Momentum (plus a momentum sweep)
  4. Training/eval    : >= 20 epochs, loss curves, accuracy, confusion matrix,
                        precision, recall, model size, time per epoch

Hardware-aware activation choice
  * ReLU = max(0, x): one comparison, no exponential, no division, trivially
    quantisable to int8. Sigmoid / tanh / swish / GELU / ELU all need exp().
  * The output layer emits raw logits (no softmax). The softmax exponential is
    only needed for training; at inference argmax(logits) == argmax(softmax),
    so the exp() is never computed on the device.
"""
import os
import json
import time
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
# Configuration
# ----------------------------------------------------------------------------
MODEL_TAG = "modelB_v2"
MODEL_LABEL = "Model B (Lightweight CNN, improved)"
PARAM_LIMIT = 100_000  # assignment: <= 100,000 trainable parameters

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(os.environ.get("DATASET_DIR", PROJECT_DIR / "dataset6"))
RESULTS_DIR = PROJECT_DIR / "results" / MODEL_TAG

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
SEED = 42
EPOCHS = int(os.environ.get("EPOCHS", 50))  # assignment requires >= 20
AUGMENT = True  # train-only data augmentation
USE_CACHE = True  # cache decoded 64x64 images in RAM (set False if RAM is tight)

# Optimizers to compare. The first entry is the "chosen" optimizer; the full
# model artifacts (model file, confusion matrix, ...) are saved for it.
OPTIMIZERS = [
    {"key": "adam", "label": "Adam (lr=1e-4)",
     "make": lambda: keras.optimizers.Adam(learning_rate=1e-4)},
    {"key": "adam_1e-3", "label": "Adam (lr=1e-3)",
     "make": lambda: keras.optimizers.Adam(learning_rate=1e-3)},
    {"key": "sgd", "label": "SGD (lr=0.01, momentum=0)",
     "make": lambda: keras.optimizers.SGD(learning_rate=1e-2, momentum=0.0)},
    {"key": "sgd_m0.5", "label": "SGD + Momentum 0.5 (lr=0.01)",
     "make": lambda: keras.optimizers.SGD(learning_rate=1e-2, momentum=0.5)},
    {"key": "sgd_m0.9", "label": "SGD + Momentum 0.9 (lr=0.01)",
     "make": lambda: keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9)},
    {"key": "sgd_m0.99", "label": "SGD + Momentum 0.99 (lr=0.01)",
     "make": lambda: keras.optimizers.SGD(learning_rate=1e-2, momentum=0.99)},
]
# MAIN_OPTIMIZER_KEY is chosen after training: lowest validation loss.

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Data preparation
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
    Waste items have no canonical orientation, so vertical flips and rotations
    are label-preserving."""
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15, fill_mode="reflect"),
        layers.RandomZoom(0.15, fill_mode="reflect"),
        layers.RandomTranslation(0.10, 0.10, fill_mode="reflect"),
        layers.RandomContrast(0.20),
        layers.RandomBrightness(0.20, value_range=(0.0, 1.0)),
    ], name="train_augmentation")


augmentation = build_augmentation()


def make_dataset(paths_, labels_, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths_, labels_))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if USE_CACHE:
        ds = ds.cache()  # cache the clean decoded images; augment afterwards
    if training:
        ds = ds.shuffle(len(paths_), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE)
    if training and AUGMENT:
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(X_train, y_train, training=True)
val_ds = make_dataset(X_val, y_val)
test_ds = make_dataset(X_test, y_test)


# ----------------------------------------------------------------------------
# ===== MODEL DEFINITION (START) =====
# ----------------------------------------------------------------------------
def build_model():
    """Model B v2: same topology as Model A v2 but every Conv2D is a
    depthwise-separable conv. Each stage = 2 x (SeparableConv2D 3x3 -> BN -> ReLU) then
    MaxPooling2D. Output = logits."""
    def stage(filters, dropout=0.0):
        s = []
        for _ in range(2):
            s += [layers.SeparableConv2D(filters, kernel_size=(3, 3),
                                       padding="same", use_bias=False),
                  layers.BatchNormalization(),
                  layers.Activation("relu")]
        s.append(layers.MaxPooling2D(pool_size=(2, 2)))
        if dropout:
            s.append(layers.Dropout(dropout))
        return s

    return keras.Sequential(
        [layers.Input(shape=(64, 64, 3))]
        + stage(32) + stage(64, 0.10) + stage(128, 0.10)
        + [layers.GlobalAveragePooling2D(),
           layers.Dense(128, activation="relu"),
           layers.Dropout(0.25),
           layers.Dense(num_classes)],  # logits; softmax folded into the loss
        name="ModelB_Lightweight_CNN_v2")
# ----------------------------------------------------------------------------
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
    val_loss = history.history["val_loss"]
    val_acc = history.history["val_accuracy"]
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
        "best_validation_loss": float(min(val_loss)),
        "best_validation_loss_epoch": int(np.argmin(val_loss)) + 1,
        "best_validation_accuracy": float(max(val_acc)),
        "best_validation_accuracy_epoch": int(np.argmax(val_acc)) + 1,
        "final_training_loss": float(history.history["loss"][-1]),
        "final_training_accuracy": float(history.history["accuracy"][-1]),
        "final_validation_loss": float(val_loss[-1]),
        "final_validation_accuracy": float(val_acc[-1]),
        "total_training_time_seconds": float(total_time),
        "average_epoch_time_seconds": float(np.mean(timer.epoch_times)),
        "median_epoch_time_seconds": float(np.median(timer.epoch_times)),
        "epoch_time_seconds": timer.epoch_times,
        "history": hist,
    }
    print(f"{opt_cfg['label']}: test acc {test_acc:.4f} | macro F1 "
          f"{result['test_f1_macro']:.4f} | best val loss "
          f"{result['best_validation_loss']:.4f} @ epoch "
          f"{result['best_validation_loss_epoch']} | "
          f"{result['average_epoch_time_seconds']:.2f}s/epoch")
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
print(f"Non-trainable parameters: {non_trainable_params:,}")
print(f"Total parameters:         {total_params:,}")
print(f"MACs per inference:       {total_macs:,}")
print(f"Float32 weight size:      {float32_size_kb:.1f} KB")

if PARAM_LIMIT is not None:
    assert trainable_params <= PARAM_LIMIT, (
        f"{MODEL_LABEL} exceeds the {PARAM_LIMIT:,} trainable-parameter "
        f"limit: {trainable_params:,}")
    print(f"Parameter constraint satisfied (<= {PARAM_LIMIT:,}).")

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
    "model": MODEL_LABEL,
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
    "median_epoch_time_seconds": main_res["median_epoch_time_seconds"],
}
(RESULTS_DIR / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
(RESULTS_DIR / "training_history.json").write_text(
    json.dumps({**main_res["history"], "epoch_time_seconds": main_res["epoch_time_seconds"]}, indent=2),
    encoding="utf-8")
(RESULTS_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")

(RESULTS_DIR / "run_metadata.json").write_text(json.dumps({
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "model": MODEL_LABEL,
    "dataset": str(DATASET_DIR),
    "image_size": list(IMG_SIZE),
    "batch_size": BATCH_SIZE,
    "epochs_requested": EPOCHS,
    "seed": SEED,
    "main_optimizer": main_res["label"],
    "loss": "SparseCategoricalCrossentropy(from_logits=True)",
    "augmentation": "flip(h+v), rotation 0.15, zoom 0.15, translation 0.10, contrast 0.2, brightness 0.2 (train only)",
    "lr_schedule": "ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6)",
    "class_names": class_names,
    "class_to_idx": class_to_idx,
    "num_images": len(paths),
    "train_images": len(X_train),
    "validation_images": len(X_val),
    "test_images": len(X_test),
}, indent=2), encoding="utf-8")

# Optimizer comparison table
lines = [
    f"{MODEL_LABEL}: optimizer comparison ({EPOCHS} epochs each, same seed/init)",
    "",
    f"{'Optimizer':34s}{'BestValLoss':>12s}{'@Ep':>5s}{'ValAcc':>8s}{'TestAcc':>9s}{'MacroF1':>9s}{'s/epoch':>9s}",
    "-" * 86,
]
for r in results.values():
    lines.append(
        f"{r['label']:34s}{r['best_validation_loss']:>12.4f}"
        f"{r['best_validation_loss_epoch']:>5d}{r['best_validation_accuracy']:>8.4f}"
        f"{r['test_accuracy']:>9.4f}{r['test_f1_macro']:>9.4f}"
        f"{r['average_epoch_time_seconds']:>9.2f}")
comparison_text = "\n".join(lines)
print("\n" + comparison_text)
(RESULTS_DIR / "optimizer_comparison.txt").write_text(comparison_text, encoding="utf-8")
(RESULTS_DIR / "optimizer_comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

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

# Optimizer comparison curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for r in results.values():
    ep = range(1, r["epochs_completed"] + 1)
    axes[0].plot(ep, r["history"]["loss"], label=r["label"])
    axes[1].plot(ep, r["history"]["val_loss"], label=r["label"])
    axes[2].plot(ep, r["history"]["val_accuracy"], label=r["label"])
for ax, title, ylab in zip(
        axes,
        ["Training loss", "Validation loss", "Validation accuracy"],
        ["Loss", "Loss", "Accuracy"]):
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3)
axes[0].legend(fontsize=8)
fig.suptitle(f"{MODEL_LABEL}: optimizer comparison")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "optimizer_comparison.png", dpi=150)
plt.close()

# Confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
    ax=ax, cmap="Blues", xticks_rotation=45, values_format="d", colorbar=False)
ax.set_title(f"{MODEL_LABEL}: Test Confusion Matrix")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close()

print(f"\nResults saved to: {RESULTS_DIR}")