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
)

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset6"
RESULTS_DIR = PROJECT_DIR / "results" / "modelA"

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 40

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
keras.utils.set_random_seed(SEED)

class_names = sorted(
    p.name for p in DATASET_DIR.iterdir() if p.is_dir()
)
class_to_idx = {
    name: index for index, name in enumerate(class_names)
}
num_classes = len(class_names)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

paths, labels = [], []

for class_name in class_names:
    class_dir = DATASET_DIR / class_name

    for path in sorted(class_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            paths.append(str(path))
            labels.append(class_to_idx[class_name])

if not paths:
    raise ValueError(f"No images found in {DATASET_DIR}")

class_counts = Counter(labels)

print("\nImages per class:")
for name, index in class_to_idx.items():
    print(f"{name:30s}: {class_counts[index]}")

print(f"\nTotal images: {len(paths)}")
print(f"Number of classes: {num_classes}")

# Stratified 70% / 15% / 15% split
X_train, X_temp, y_train, y_temp = train_test_split(
    paths,
    labels,
    test_size=0.30,
    stratify=labels,
    random_state=SEED,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=SEED,
)

print(f"\nTraining:   {len(X_train)} ({len(X_train)/len(paths):.1%})")
print(f"Validation: {len(X_val)} ({len(X_val)/len(paths):.1%})")
print(f"Testing:    {len(X_test)} ({len(X_test)/len(paths):.1%})")
print("Classes:", class_to_idx)


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


train_ds = make_dataset(X_train, y_train, training=True)
val_ds = make_dataset(X_val, y_val)
test_ds = make_dataset(X_test, y_test)

# Model A: conventional CNN
model = keras.Sequential([
    layers.Input(shape=(64, 64, 3)),

    # Convolution block 1
    layers.Conv2D(
        32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
    ),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Convolution block 2
    layers.Conv2D(
        64,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
    ),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.10),

    # Convolution block 3
    layers.Conv2D(
        128,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
    ),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Classification head
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.25),
    layers.Dense(num_classes, activation="softmax"),
], name="ModelA_Conventional_CNN")

trainable_params = sum(
    int(np.prod(weight.shape))
    for weight in model.trainable_weights
)
total_params = model.count_params()
non_trainable_params = total_params - trainable_params

print("\nModel A parameters:")
print(f"Trainable parameters:     {trainable_params:,}")
print(f"Non-trainable parameters: {non_trainable_params:,}")
print(f"Total parameters:         {total_params:,}")

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


class EpochTimer(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.perf_counter() - self.epoch_start
        self.epoch_times.append(float(elapsed))
        print(f"Epoch time: {elapsed:.2f} seconds")


epoch_timer = EpochTimer()

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(RESULTS_DIR / "best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        mode="min",
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        mode="min",
        verbose=1,
    ),
    epoch_timer,
]

print("\nTraining Model A...")

training_start = time.perf_counter()

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=2,
)

total_training_time = time.perf_counter() - training_start

# Evaluate the restored best-validation-loss weights
test_loss, test_accuracy = model.evaluate(test_ds, verbose=2)

test_probabilities = model.predict(test_ds, verbose=0)
test_predictions = test_probabilities.argmax(axis=1)

test_f1_macro = f1_score(
    y_test,
    test_predictions,
    average="macro",
    zero_division=0,
)
test_f1_weighted = f1_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0,
)

test_classification_report = classification_report(
    y_test,
    test_predictions,
    labels=list(range(num_classes)),
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)
report_text = classification_report(
    y_test,
    test_predictions,
    labels=list(range(num_classes)),
    target_names=class_names,
    zero_division=0,
)

average_epoch_time = float(np.mean(epoch_timer.epoch_times))
history_data = {
    metric: [float(value) for value in values]
    for metric, values in history.history.items()
}
history_data["epoch_time_seconds"] = epoch_timer.epoch_times

final_epoch = len(history.history["loss"])
best_val_loss_epoch = int(np.argmin(history.history["val_loss"])) + 1
best_val_accuracy_epoch = int(np.argmax(history.history["val_accuracy"])) + 1

print("\nModel A test results:")
print(f"Test loss:            {test_loss:.4f}")
print(f"Test accuracy:        {test_accuracy:.4f}")
print(f"Test Macro F1:        {test_f1_macro:.4f}")
print(f"Test Weighted F1:     {test_f1_weighted:.4f}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Total parameters:     {total_params:,}")
print(f"Total training time:  {total_training_time:.2f} seconds")
print(f"Average epoch time:   {average_epoch_time:.2f} seconds")

# Save the restored best model and training history
model.save(RESULTS_DIR / "model.keras")

(RESULTS_DIR / "training_history.json").write_text(
    json.dumps(history_data, indent=2),
    encoding="utf-8",
)

final_metrics = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "test_f1_macro": float(test_f1_macro),
    "test_f1_weighted": float(test_f1_weighted),
    "test_classification_report": test_classification_report,
    "trainable_parameters": int(trainable_params),
    "non_trainable_parameters": int(non_trainable_params),
    "total_parameters": int(total_params),
    "final_epoch": int(final_epoch),
    "best_validation_loss": float(min(history.history["val_loss"])),
    "best_validation_loss_epoch": best_val_loss_epoch,
    "best_validation_accuracy": float(
        max(history.history["val_accuracy"])
    ),
    "best_validation_accuracy_epoch": best_val_accuracy_epoch,
    "final_training_loss": float(history.history["loss"][-1]),
    "final_training_accuracy": float(history.history["accuracy"][-1]),
    "final_validation_loss": float(history.history["val_loss"][-1]),
    "final_validation_accuracy": float(
        history.history["val_accuracy"][-1]
    ),
    "total_training_time_seconds": float(total_training_time),
    "average_epoch_time_seconds": average_epoch_time,
}

(RESULTS_DIR / "metrics.json").write_text(
    json.dumps(final_metrics, indent=2),
    encoding="utf-8",
)

run_metadata = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "model": "Model A - Conventional CNN",
    "dataset": str(DATASET_DIR),
    "image_size": list(IMG_SIZE),
    "batch_size": BATCH_SIZE,
    "epochs_requested": EPOCHS,
    "epochs_completed": final_epoch,
    "seed": SEED,
    "optimizer": "Adam",
    "initial_learning_rate": 1e-4,
    "class_names": class_names,
    "class_to_idx": class_to_idx,
    "num_images": len(paths),
    "train_images": len(X_train),
    "validation_images": len(X_val),
    "test_images": len(X_test),
    "trainable_parameters": int(trainable_params),
    "non_trainable_parameters": int(non_trainable_params),
    "total_parameters": int(total_params),
    "total_training_time_seconds": float(total_training_time),
    "epoch_time_seconds": epoch_timer.epoch_times,
}

(RESULTS_DIR / "run_metadata.json").write_text(
    json.dumps(run_metadata, indent=2),
    encoding="utf-8",
)

# Plot class distribution
plt.figure(figsize=(12, 6))
plt.bar(
    class_names,
    [class_counts[class_to_idx[name]] for name in class_names],
)
plt.title("Images per Class")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=150)
plt.close()

# Plot training and validation loss
epochs = range(1, final_epoch + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, history.history["loss"], label="Training Loss")
plt.plot(epochs, history.history["val_loss"], label="Validation Loss")
plt.title("Model A: Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "loss_curve.png", dpi=150)
plt.close()

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(epochs, history.history["accuracy"], label="Training Accuracy")
plt.plot(
    epochs,
    history.history["val_accuracy"],
    label="Validation Accuracy",
)
plt.title("Model A: Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "accuracy_curve.png", dpi=150)
plt.close()

# Plot confusion matrix
cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=list(range(num_classes)),
)

fig, ax = plt.subplots(figsize=(10, 8))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)
display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format="d",
    colorbar=False,
)
ax.set_title("Model A: Test Confusion Matrix")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()

(RESULTS_DIR / "classification_report.txt").write_text(
    report_text,
    encoding="utf-8",
)

print(f"\nResults saved to: {RESULTS_DIR}")