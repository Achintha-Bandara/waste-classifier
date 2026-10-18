import json
from pathlib import Path
from datetime import datetime, timezone

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset6"
RESULTS_DIR = PROJECT_DIR / "results" / "modelA"
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 50

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class_names = sorted(
    p.name for p in DATASET_DIR.iterdir() if p.is_dir()
)
class_to_idx = {name: i for i, name in enumerate(class_names)}

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

paths, labels = [], []


for class_name in class_names:
    for path in (DATASET_DIR / class_name).rglob("*"):
        if path.suffix.lower() in extensions:
            paths.append(str(path))
            labels.append(class_to_idx[class_name])

from collections import Counter

class_counts = Counter(labels)

print("\nImages per class:")
for class_name, class_idx in class_to_idx.items():
    print(f"{class_name:30s}: {class_counts[class_idx]}")


# 70% train, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    paths, labels,
    test_size=0.30,
    stratify=labels,
    random_state=SEED
)

# Split the remaining 30% into 15% validation and 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=SEED
)

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image, label

def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(X_train, y_train, training=True)
val_ds = make_dataset(X_val, y_val)
test_ds = make_dataset(X_test, y_test)

print(f"Training:   {len(X_train)} ({len(X_train)/len(paths):.1%})")
print(f"Validation: {len(X_val)} ({len(X_val)/len(paths):.1%})")
print(f"Testing:    {len(X_test)} ({len(X_test)/len(paths):.1%})")
print("Classes:", class_to_idx)

num_classes = len(class_names)

model = keras.Sequential([
    layers.Input(shape=(64, 64, 3)),

    # Block 1
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Block 2
    layers.Conv2D(64, (3, 3), padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.10),

    # Block 3
    layers.Conv2D(128, (3, 3), padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2, 2)),

    # Classification head
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.25),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train using the existing training and validation datasets
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(RESULTS_DIR / "best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        mode="min",
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        mode="min",
        verbose=1
    )
]

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=2
)

# Evaluate on the test set after training
test_loss, test_accuracy = model.evaluate(test_ds, verbose=2)
test_probabilities = model.predict(test_ds, verbose=0)
test_predictions = test_probabilities.argmax(axis=1)
test_f1_macro = f1_score(y_test, test_predictions, average="macro", zero_division=0)
test_f1_weighted = f1_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)
test_classification_report = classification_report(
    y_test,
    test_predictions,
    labels=list(range(num_classes)),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1 Score (Macro): {test_f1_macro:.4f}")
print(f"Test F1 Score (Weighted): {test_f1_weighted:.4f}")

# Save the trained model, configuration, metrics, and training history.
model.save(RESULTS_DIR / "model.keras")

history_data = {
    metric: [float(value) for value in values]
    for metric, values in history.history.items()
}
(RESULTS_DIR / "training_history.json").write_text(
    json.dumps(history_data, indent=2),
    encoding="utf-8"
)

final_epoch = len(history_data["loss"])
final_metrics = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "test_f1_macro": float(test_f1_macro),
    "test_f1_weighted": float(test_f1_weighted),
    "test_classification_report": test_classification_report,
    "final_epoch": final_epoch,
    "final_training_loss": history_data["loss"][-1],
    "final_training_accuracy": history_data["accuracy"][-1],
    "final_validation_loss": history_data["val_loss"][-1],
    "final_validation_accuracy": history_data["val_accuracy"][-1],
    "best_validation_loss": min(history_data["val_loss"]),
    "best_validation_accuracy": max(history_data["val_accuracy"]),
}
(RESULTS_DIR / "metrics.json").write_text(
    json.dumps(final_metrics, indent=2),
    encoding="utf-8"
)

run_metadata = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "dataset": str(DATASET_DIR),
    "image_size": list(IMG_SIZE),
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "seed": SEED,
    "class_names": class_names,
    "class_to_idx": class_to_idx,
    "num_images": len(paths),
    "train_images": len(X_train),
    "validation_images": len(X_val),
    "test_images": len(X_test),
}
(RESULTS_DIR / "run_metadata.json").write_text(
    json.dumps(run_metadata, indent=2),
    encoding="utf-8"
)

# Save the class distribution plot.
plt.figure(figsize=(12, 6))
plt.bar(class_names, [class_counts[class_to_idx[name]] for name in class_names])
plt.title("Images per Class")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=150)
plt.close()

# Save training and validation loss and accuracy as separate plots.
epochs = range(1, final_epoch + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, history_data["loss"], label="Training Loss", marker="o")
plt.plot(epochs, history_data["val_loss"], label="Validation Loss", marker="o")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "loss_curve.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(epochs, history_data["accuracy"], label="Training Accuracy", marker="o")
plt.plot(epochs, history_data["val_accuracy"], label="Validation Accuracy", marker="o")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "accuracy_curve.png", dpi=150)
plt.close()

print(f"Results saved to: {RESULTS_DIR}")