from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

DATASET_DIR = Path("dataset")
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
SEED = 42

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

# Define the CNN model
num_classes = len(class_names)

model = keras.Sequential([
    layers.Input(shape=(64, 64, 3)),

    # First convolutional block
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Classification layers
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

learning_rate = 0.0001
optimizer = keras.optimizers.Adam(
    learning_rate=learning_rate
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# model = keras.Sequential([
#     layers.Input(shape=(64, 64, 3)),

#     layers.Conv2D(16, 3, padding="same", activation="relu"),
#     layers.MaxPooling2D(2),

#     layers.Conv2D(32, 3, padding="same", activation="relu"),
#     layers.MaxPooling2D(2),

#     layers.Conv2D(64, 3, padding="same", activation="relu"),
#     layers.MaxPooling2D(2),

#     layers.GlobalAveragePooling2D(),
#     layers.Dense(32, activation="relu"),
#     layers.Dropout(0.4),
#     layers.Dense(num_classes, activation="softmax")
# ])

# model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=0.0003),
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# Train using the existing training and validation datasets
history = model.fit(
    train_ds,
    epochs=20,
    validation_data=val_ds,
    verbose=2
)

# Evaluate on the test set after training
test_loss, test_accuracy = model.evaluate(test_ds, verbose=2)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


# Plot training, validation, and testing loss with respect to epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', linestyle='--', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()