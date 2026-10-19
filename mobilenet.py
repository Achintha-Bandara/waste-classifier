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
MODEL_KEY = "mobilenetv2"
MODEL_LABEL = "MobileNetV2"
ALLOWED_SIZES = {96, 128, 160, 192, 224}

parser = argparse.ArgumentParser(
    description=f"Fine-tune ImageNet-pretrained {MODEL_LABEL}.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--img-size", type=int, default=128,
                    help="resolution the 64x64 images are upsampled to inside the model")
parser.add_argument("--head-epochs", type=int, default=8,
                    help="phase 1: epochs with the backbone frozen")
parser.add_argument("--fine-epochs", type=int, default=30,
                    help="phase 2: epochs with the top backbone layers unfrozen")
parser.add_argument("--fine-tune-layers", type=int, default=60,
                    help="number of backbone layers (from the top) to unfreeze in phase 2")
parser.add_argument("--head-lr", type=float, default=1e-3, help="Adam lr, phase 1")
parser.add_argument("--fine-lr", type=float, default=1e-4, help="Adam lr, phase 2")
parser.add_argument("--dropout", type=float, default=0.2, help="dropout before the classifier")
parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet",
                    help="'none' = random initialisation (no pretraining)")
parser.add_argument("--aug", choices=["none", "geometric", "full"],
                    default=os.environ.get("AUG_MODE", "none"),
                    help="training-only augmentation (keep equal to modelA/modelB)")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", default=os.environ.get("DATASET_DIR"),
                    help="dataset root (one sub-folder per class); "
                         "default: <script dir>/dataset6")
parser.add_argument("--no-cache", action="store_true",
                    help="do not cache decoded images in RAM")
args = parser.parse_args()

if args.head_epochs < 1 or args.fine_epochs < 1:
    parser.error("--head-epochs and --fine-epochs must both be >= 1")
if args.fine_tune_layers < 1:
    parser.error("--fine-tune-layers must be at least 1")
if ALLOWED_SIZES and args.weights == "imagenet" and args.img_size not in ALLOWED_SIZES:
    parser.error(f"{MODEL_LABEL} ImageNet weights support --img-size in "
                 f"{sorted(ALLOWED_SIZES)} (got {args.img_size})")
TOTAL_EPOCHS = args.head_epochs + args.fine_epochs
if TOTAL_EPOCHS < 20:
    print(f"WARNING: {TOTAL_EPOCHS} total epochs is below the assignment minimum of 20.")

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(args.dataset) if args.dataset else PROJECT_DIR / "dataset6"
IMG_SIZE = (64, 64)
MODEL_RES = args.img_size
BATCH_SIZE = args.batch_size
SEED = args.seed
AUG_MODE = args.aug
USE_CACHE = not args.no_cache
WEIGHTS = None if args.weights == "none" else "imagenet"
RESULTS_DIR = PROJECT_DIR / "results" / MODEL_KEY
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COMMAND_LINE = "python " + " ".join([Path(sys.argv[0]).name] + sys.argv[1:])
print(f"\nCommand : {COMMAND_LINE}")
print(f"Arguments: model={MODEL_KEY} img_size={MODEL_RES} head_epochs={args.head_epochs} "
      f"fine_epochs={args.fine_epochs} fine_tune_layers={args.fine_tune_layers} "
      f"weights={args.weights} aug={AUG_MODE} batch={BATCH_SIZE} seed={SEED}")
(RESULTS_DIR / "command.txt").write_text(
    COMMAND_LINE + "\n" + json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
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
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def build_augmentation():
    """Train-only augmentation (no trainable parameters, not part of the model)."""
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
        ds = ds.cache()
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
def build_model():
    """64x64 [0,1] image -> upsample -> backbone preprocessing -> pretrained
    backbone -> GAP -> Dropout -> Dense(num_classes) logits."""
    backbone = keras.applications.MobileNetV2(
        input_shape=(MODEL_RES, MODEL_RES, 3), include_top=False,
        weights=WEIGHTS, alpha=1.0)
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="image_64x64")
    x = layers.Resizing(MODEL_RES, MODEL_RES, interpolation="bilinear",
                        name="upsample")(inputs)
    x = layers.Rescaling(2.0, offset=-1.0, name="mobilenetv2_preprocess")(x)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(args.dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, name="classifier")(x)
    return keras.Model(inputs, outputs, name=f"{MODEL_KEY}_finetuned"), backbone


def set_fine_tune(backbone, n_layers):
    """Unfreeze the top n_layers of the backbone; BatchNorm layers stay frozen."""
    backbone.trainable = True
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False



def count_trainable(model):
    return int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))


def count_macs(model):
    """Approximate MACs per inference: Conv2D, DepthwiseConv2D, SeparableConv2D
    and Dense layers (same convention as modelA/modelB). Walks into the backbone."""
    total = 0
    skipped = 0

    def walk(m):
        nonlocal total, skipped
        for layer in m.layers:
            if isinstance(layer, keras.Model):
                walk(layer)
                continue
            try:
                out_shape = tuple(layer.output.shape[1:])
                in_shape = tuple(layer.input.shape[1:])
            except Exception:
                skipped += 1
                continue
            if isinstance(layer, layers.DepthwiseConv2D):
                k = layer.kernel_size[0] * layer.kernel_size[1]
                total += out_shape[0] * out_shape[1] * k * out_shape[-1]
            elif isinstance(layer, layers.SeparableConv2D):
                k = layer.kernel_size[0] * layer.kernel_size[1]
                cin, cout = in_shape[-1], layer.filters
                total += out_shape[0] * out_shape[1] * (k * cin + cin * cout)
            elif isinstance(layer, layers.Conv2D):
                k = layer.kernel_size[0] * layer.kernel_size[1]
                total += out_shape[0] * out_shape[1] * k * in_shape[-1] * layer.filters
            elif isinstance(layer, layers.Dense):
                total += in_shape[-1] * out_shape[-1]

    walk(model)
    return int(total), skipped


class EpochTimer(keras.callbacks.Callback):
    """Records every epoch across both training phases."""
    def __init__(self):
        super().__init__()
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(float(time.perf_counter() - self.t0))


class BestWeights(keras.callbacks.Callback):
    """Keeps the weights with the lowest validation loss across BOTH phases."""
    def __init__(self):
        super().__init__()
        self.best = float("inf")
        self.weights = None
        self.epoch = None
        self.seen = 0

    def on_epoch_end(self, epoch, logs=None):
        self.seen += 1
        val = (logs or {}).get("val_loss")
        if val is not None and val < self.best:
            self.best = float(val)
            self.weights = self.model.get_weights()
            self.epoch = self.seen


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )


def lr_callback():
    return keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7,
        mode="min", verbose=1)
keras.utils.set_random_seed(SEED)
model, backbone = build_model()
model.summary()
summary_lines = []
model.summary(print_fn=lambda s, **kw: summary_lines.append(s), expand_nested=False)
(RESULTS_DIR / "model_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

total_params = int(model.count_params())
backbone_params = int(backbone.count_params())
head_params = total_params - backbone_params
macs, macs_skipped = count_macs(model)
float32_size_mb = total_params * 4 / (1024 ** 2)
print(f"\nBackbone parameters:      {backbone_params:,}")
print(f"Head parameters:          {head_params:,}")
print(f"Total parameters:         {total_params:,}")
print(f"MACs per inference:       {macs:,} at {MODEL_RES}x{MODEL_RES} input "
      f"(approx.; conv + dense layers; {macs_skipped} layers skipped)")
print(f"Float32 weight size:      {float32_size_mb:.2f} MB")
timer, best = EpochTimer(), BestWeights()
backbone.trainable = False
compile_model(model, args.head_lr)
head_trainable = count_trainable(model)
print(f"\n{'=' * 70}\nPhase 1: head only (backbone frozen), "
      f"{head_trainable:,} trainable parameters, lr={args.head_lr}\n{'=' * 70}")
t0 = time.perf_counter()
head_history = model.fit(train_ds, validation_data=val_ds,
                         epochs=args.head_epochs,
                         callbacks=[lr_callback(), best, timer], verbose=2)
set_fine_tune(backbone, args.fine_tune_layers)
compile_model(model, args.fine_lr)
fine_trainable = count_trainable(model)
print(f"\n{'=' * 70}\nPhase 2: fine-tuning top {args.fine_tune_layers} backbone layers, "
      f"{fine_trainable:,} trainable parameters, lr={args.fine_lr}\n{'=' * 70}")
fine_history = model.fit(train_ds, validation_data=val_ds,
                         initial_epoch=args.head_epochs, epochs=TOTAL_EPOCHS,
                         callbacks=[lr_callback(), best, timer], verbose=2)
total_time = time.perf_counter() - t0

if best.weights is not None:
    model.set_weights(best.weights)
print(f"\nRestored best weights: val loss {best.best:.4f} @ epoch {best.epoch}")
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
logits = model.predict(test_ds, verbose=0)
main_preds = logits.argmax(axis=1)

history = {
    key: [float(v) for v in head_history.history.get(key, [])]
    + [float(v) for v in fine_history.history.get(key, [])]
    for key in set(head_history.history) | set(fine_history.history)
}
n_epochs = len(history["val_loss"])
train_loss, val_loss = history["loss"], history["val_loss"]
val_acc, train_acc = history["val_accuracy"], history["accuracy"]
et = timer.epoch_times
fine_et = et[args.head_epochs:]

model_path = RESULTS_DIR / "model.keras"
model.save(model_path)
disk_size_kb = model_path.stat().st_size / 1024
disk_size_mb = disk_size_kb / 1024
print(f"\nSaved model file size: {disk_size_mb:.2f} MB ({disk_size_kb:.1f} KB)")

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
    "weights": args.weights,
    "backbone_input_resolution": MODEL_RES,
    "augmentation": AUG_MODE,
    "test_loss": float(test_loss),
    "test_accuracy": float(test_acc),
    "test_precision_macro": float(precision_score(y_test, main_preds, average="macro", zero_division=0)),
    "test_recall_macro": float(recall_score(y_test, main_preds, average="macro", zero_division=0)),
    "test_f1_macro": float(f1_score(y_test, main_preds, average="macro", zero_division=0)),
    "test_f1_weighted": float(f1_score(y_test, main_preds, average="weighted", zero_division=0)),
    "test_classification_report": report_dict,
    "confusion_matrix": cm.tolist(),
    "total_parameters": total_params,
    "backbone_parameters": backbone_params,
    "head_parameters": head_params,
    "trainable_parameters_phase1": head_trainable,
    "trainable_parameters_phase2": fine_trainable,
    "macs_per_inference": macs,
    "float32_weight_size_mb": float(float32_size_mb),
    "model_file_size_kb": float(disk_size_kb),
    "model_file_size_mb": float(disk_size_mb),
    "epochs_completed": n_epochs,
    "best_validation_loss": float(best.best),
    "best_validation_loss_epoch": int(best.epoch),
    "best_validation_accuracy": float(max(val_acc)),
    "best_validation_accuracy_epoch": int(np.argmax(val_acc)) + 1,
    "final_training_loss": float(train_loss[-1]),
    "final_training_accuracy": float(train_acc[-1]),
    "final_validation_loss": float(val_loss[-1]),
    "final_validation_accuracy": float(val_acc[-1]),
    "total_training_time_seconds": float(total_time),
    "average_epoch_time_seconds": float(np.mean(et)),
    "average_epoch_time_excl_first_seconds": float(np.mean(et[1:])) if len(et) > 1 else float(np.mean(et)),
    "median_epoch_time_seconds": float(np.median(et)),
    "median_epoch_time_fine_tune_seconds": float(np.median(fine_et)) if fine_et else None,
}
(RESULTS_DIR / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
(RESULTS_DIR / "training_history.json").write_text(
    json.dumps({**history, "epoch_time_seconds": et}, indent=2), encoding="utf-8")
(RESULTS_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
np.savetxt(RESULTS_DIR / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

(RESULTS_DIR / "run_metadata.json").write_text(json.dumps({
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "command": COMMAND_LINE,
    "arguments": vars(args),
    "model": MODEL_LABEL,
    "dataset": str(DATASET_DIR),
    "data_pipeline_image_size": list(IMG_SIZE),
    "backbone_input_resolution": MODEL_RES,
    "batch_size": BATCH_SIZE,
    "head_epochs": args.head_epochs,
    "fine_tune_epochs": args.fine_epochs,
    "fine_tune_layers": args.fine_tune_layers,
    "optimizer": f"Adam (head lr={args.head_lr}, fine-tune lr={args.fine_lr})",
    "loss": "SparseCategoricalCrossentropy(from_logits=True)",
    "lr_schedule": "ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7) per phase",
    "seed": SEED,
    "class_names": class_names,
    "class_to_idx": class_to_idx,
    "num_images": len(paths),
    "train_images": len(X_train),
    "validation_images": len(X_val),
    "test_images": len(X_test),
}, indent=2), encoding="utf-8")

print(f"\n{MODEL_LABEL}: test acc {test_acc:.4f} | macro F1 "
      f"{final_metrics['test_f1_macro']:.4f} | params {total_params:,} | "
      f"{disk_size_mb:.2f} MB | {final_metrics['median_epoch_time_seconds']:.2f}s/epoch (median)")
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

epochs_axis = range(1, n_epochs + 1)
boundary = args.head_epochs + 0.5

plt.figure(figsize=(10, 6))
plt.plot(epochs_axis, train_loss, label="Training Loss")
plt.plot(epochs_axis, val_loss, label="Validation Loss")
plt.axvline(boundary, color="gray", linestyle="--", label="unfreeze backbone")
plt.title(f"{MODEL_LABEL}: Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "loss_curve.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(epochs_axis, train_acc, label="Training Accuracy")
plt.plot(epochs_axis, val_acc, label="Validation Accuracy")
plt.axvline(boundary, color="gray", linestyle="--", label="unfreeze backbone")
plt.title(f"{MODEL_LABEL}: Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "accuracy_curve.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
    ax=ax, cmap="Blues", xticks_rotation=45, values_format="d", colorbar=False)
ax.set_title(f"{MODEL_LABEL}: Test Confusion Matrix")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close()

print(f"\nResults saved to: {RESULTS_DIR}")