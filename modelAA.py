
import json
import random
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)


# ============================================================
# 1. Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset6"
RESULTS_DIR = PROJECT_DIR / "results" / "modelA"

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 30                 # At least 20 epochs as required

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# ============================================================
# 2. Discover dataset and create stratified splits
# ============================================================

if not DATASET_DIR.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")

class_names = sorted(
    p.name for p in DATASET_DIR.iterdir() if p.is_dir()
)

if len(class_names) < 2:
    raise ValueError("At least two class directories are required.")

class_to_idx = {
    name: idx for idx, name in enumerate(class_names)
}
num_classes = len(class_names)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

paths, labels = [], []

for class_name in class_names:
    for path in sorted((DATASET_DIR / class_name).rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            paths.append(str(path))
            labels.append(class_to_idx[class_name])

if not paths:
    raise ValueError(f"No supported images found in {DATASET_DIR}")

labels = np.asarray(labels, dtype=np.int32)

print("\nOriginal dataset distribution:")
for name, idx in class_to_idx.items():
    print(f"{name:30s}: {int(np.sum(labels == idx))}")

# 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    paths,
    labels,
    test_size=0.30,
    stratify=labels,
    random_state=SEED,
)

# Split the remaining 30% equally into validation and test
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


# ============================================================
# 3. Efficient TensorFlow input pipeline
# ============================================================

def load_image(path, label):
    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])

    # All images are resized to 64 x 64 or smaller
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def make_dataset(image_paths, image_labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices(
        (image_paths, image_labels)
    )

    if training:
        ds = ds.shuffle(
            buffer_size=len(image_paths),
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


# ============================================================
# 4. Model A: Standard CNN
# ============================================================

def build_model_a():
    # Mild augmentation, applied during training only
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=SEED),
            layers.RandomRotation(0.05, seed=SEED + 1),
            layers.RandomContrast(0.10, seed=SEED + 2),
        ],
        name="data_augmentation",
    )

    model = keras.Sequential(
        [
            layers.Input(shape=(64, 64, 3)),
            augmentation,

            # Block 1: standard convolutions
            layers.Conv2D(
                32, 3, padding="same", use_bias=False
            ),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2D(
                32, 3, padding="same", use_bias=False
            ),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.MaxPooling2D(pool_size=2),
            layers.Dropout(0.10),

            # Block 2
            layers.Conv2D(
                64, 3, padding="same", use_bias=False
            ),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2D(
                64, 3, padding="same", use_bias=False
            ),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.MaxPooling2D(pool_size=2),
            layers.Dropout(0.15),

            # Block 3
            layers.Conv2D(
                128, 3, padding="same", use_bias=False
            ),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.MaxPooling2D(pool_size=2),
            layers.Dropout(0.20),

            # Compact classification head
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.30),

            # Softmax produces probabilities for all classes
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="ModelA_StandardCNN",
    )

    return model


# ============================================================
# 5. Optimizers to compare
# ============================================================

def create_optimizer(name):
    if name == "Adam":
        return keras.optimizers.Adam(learning_rate=3e-4)

    if name == "SGD":
        return keras.optimizers.SGD(learning_rate=0.01)

    if name == "SGD_Momentum":
        return keras.optimizers.SGD(
            learning_rate=0.01,
            momentum=0.9,
        )

    raise ValueError(f"Unknown optimizer: {name}")


# ============================================================
# 6. Record training time per epoch
# ============================================================

class EpochTimer(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.perf_counter() - self.epoch_start
        self.epoch_times.append(float(elapsed))

        print(f"Epoch time: {elapsed:.2f} seconds")


# ============================================================
# 7. Train, select best checkpoint and evaluate
# ============================================================

optimizer_names = ["Adam", "SGD", "SGD_Momentum"]
all_results = {}

for optimizer_name in optimizer_names:
    print("\n" + "=" * 65)
    print(f"Training Model A with {optimizer_name}")
    print("=" * 65)

    run_dir = RESULTS_DIR / optimizer_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Reset the random seeds for a reproducible initialization
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    model = build_model_a()
    model.compile(
        optimizer=create_optimizer(optimizer_name),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    if optimizer_name == "Adam":
        model.summary()

    timer = EpochTimer()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best_model.keras"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
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

        timer,
    ]

    # No EarlyStopping: guarantees the requested minimum
    # of 20 epochs. EPOCHS is set to 30 above.
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

    history_data = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }

    (run_dir / "training_history.json").write_text(
        json.dumps(history_data, indent=2),
        encoding="utf-8",
    )

    # Load the best validation-loss checkpoint
    model = keras.models.load_model(
        run_dir / "best_model.keras"
    )

    # Evaluate the selected model on the test set
    test_loss, _ = model.evaluate(test_ds, verbose=0)

    test_probabilities = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(test_probabilities, axis=1)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    test_recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    test_f1_macro = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    test_f1_weighted = f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_test,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
    )

    print(f"\nOptimizer: {optimizer_name}")
    print(f"Test loss:           {test_loss:.4f}")
    print(f"Test accuracy:       {test_accuracy:.4f}")
    print(f"Macro precision:     {test_precision:.4f}")
    print(f"Macro recall:        {test_recall:.4f}")
    print(f"Macro F1:            {test_f1_macro:.4f}")
    print(f"Weighted F1:         {test_f1_weighted:.4f}")
    print("\nClassification report:\n", report_text)

    # Save classification report
    (run_dir / "classification_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    # Save confusion matrix as both CSV and PNG
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=list(range(num_classes)),
    )

    np.savetxt(
        run_dir / "confusion_matrix.csv",
        cm,
        delimiter=",",
        fmt="%d",
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    ).plot(
        ax=ax,
        xticks_rotation=45,
        cmap="Blues",
        colorbar=False,
    )

    ax.set_title(f"Model A Test Confusion Matrix — {optimizer_name}")
    fig.tight_layout()
    fig.savefig(
        run_dir / "confusion_matrix.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Save training/validation curves
    epochs_ran = len(history_data["loss"])
    epoch_numbers = range(1, epochs_ran + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        epoch_numbers,
        history_data["loss"],
        label="Training loss",
    )
    ax.plot(
        epoch_numbers,
        history_data["val_loss"],
        label="Validation loss",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Model A Loss — {optimizer_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        epoch_numbers,
        history_data["accuracy"],
        label="Training accuracy",
    )
    ax.plot(
        epoch_numbers,
        history_data["val_accuracy"],
        label="Validation accuracy",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Model A Accuracy — {optimizer_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "accuracy_curve.png", dpi=150)
    plt.close(fig)

    # Save the exact evaluated model
    evaluated_model_path = run_dir / "model.keras"
    model.save(evaluated_model_path)

    total_params = model.count_params()
    trainable_params = sum(
        int(np.prod(weight.shape))
        for weight in model.trainable_weights
    )

    actual_size_kb = evaluated_model_path.stat().st_size / 1024
    fp32_weights_kb = trainable_params * 4 / 1024

    epoch_times = timer.epoch_times

    # Record best validation values and their epoch numbers
    best_val_loss_epoch = int(
        np.argmin(history_data["val_loss"]) + 1
    )
    best_val_accuracy_epoch = int(
        np.argmax(history_data["val_accuracy"]) + 1
    )

    result = {
        "optimizer": optimizer_name,
        "learning_rate_initial": (
            3e-4 if optimizer_name == "Adam" else 0.01
        ),
        "momentum": 0.9 if optimizer_name == "SGD_Momentum" else 0.0,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_macro_precision": float(test_precision),
        "test_macro_recall": float(test_recall),
        "test_macro_f1": float(test_f1_macro),
        "test_weighted_f1": float(test_f1_weighted),
        "classification_report": report_dict,
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "estimated_fp32_trainable_weights_kb": float(fp32_weights_kb),
        "saved_model_size_kb": float(actual_size_kb),
        "epochs_completed": epochs_ran,
        "average_epoch_time_seconds": float(np.mean(epoch_times)),
        "best_validation_loss": float(
            min(history_data["val_loss"])
        ),
        "best_validation_loss_epoch": best_val_loss_epoch,
        "best_validation_accuracy": float(
            max(history_data["val_accuracy"])
        ),
        "best_validation_accuracy_epoch": best_val_accuracy_epoch,
    }

    all_results[optimizer_name] = result

    (run_dir / "metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


# ============================================================
# 8. Compare optimizer results
# ============================================================

comparison = []

for name, result in all_results.items():
    comparison.append({
        "optimizer": name,
        "test_accuracy": result["test_accuracy"],
        "macro_precision": result["test_macro_precision"],
        "macro_recall": result["test_macro_recall"],
        "macro_f1": result["test_macro_f1"],
        "parameters": result["total_parameters"],
        "model_size_kb": result["saved_model_size_kb"],
        "avg_epoch_seconds": result["average_epoch_time_seconds"],
    })

comparison_path = RESULTS_DIR / "optimizer_comparison.json"
comparison_path.write_text(
    json.dumps(comparison, indent=2),
    encoding="utf-8",
)

print("\n" + "=" * 80)
print("OPTIMIZER COMPARISON")
print("=" * 80)

print(
    f"{'Optimizer':<18}"
    f"{'Accuracy':>12}"
    f"{'Macro F1':>12}"
    f"{'Params':>14}"
    f"{'Size KB':>12}"
    f"{'Sec/Epoch':>13}"
)

for row in comparison:
    print(
        f"{row['optimizer']:<18}"
        f"{row['test_accuracy']:>12.4f}"
        f"{row['macro_f1']:>12.4f}"
        f"{row['parameters']:>14,}"
        f"{row['model_size_kb']:>12.1f}"
        f"{row['avg_epoch_seconds']:>13.2f}"
    )

metadata = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "dataset": str(DATASET_DIR),
    "image_size": list(IMG_SIZE),
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "seed": SEED,
    "class_names": class_names,
    "class_to_idx": class_to_idx,
    "total_images": len(paths),
    "train_images": len(X_train),
    "validation_images": len(X_val),
    "test_images": len(X_test),
}

(RESULTS_DIR / "run_metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

print(f"\nResults saved to: {RESULTS_DIR}")
