from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset6"
RESULTS_DIR = PROJECT_DIR / "results" / "modelA"
MODEL_PATH = RESULTS_DIR / "model.keras"
OUTPUT_PATH = RESULTS_DIR / "confusion_matrix.png"
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
SEED = 42


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image, label


def make_dataset(paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


class_names = sorted(
    path.name for path in DATASET_DIR.iterdir() if path.is_dir()
)
class_to_idx = {name: index for index, name in enumerate(class_names)}
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

paths, labels = [], []
for class_name in class_names:
    for path in (DATASET_DIR / class_name).rglob("*"):
        if path.suffix.lower() in extensions:
            paths.append(str(path))
            labels.append(class_to_idx[class_name])

_, test_paths, _, test_labels = train_test_split(
    paths,
    labels,
    test_size=0.30,
    stratify=labels,
    random_state=SEED
)
_, test_paths, _, test_labels = train_test_split(
    test_paths,
    test_labels,
    test_size=0.50,
    stratify=test_labels,
    random_state=SEED
)

test_dataset = make_dataset(test_paths, test_labels)
model = keras.models.load_model(MODEL_PATH)
probabilities = model.predict(test_dataset, verbose=0)
predictions = np.argmax(probabilities, axis=1)
matrix = confusion_matrix(
    test_labels,
    predictions,
    labels=list(range(len(class_names)))
)

figure_width = max(10, len(class_names) * 1.2)
figure, axis = plt.subplots(figsize=(figure_width, figure_width))
display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=class_names
)
display.plot(
    ax=axis,
    cmap="Blues",
    values_format="d",
    xticks_rotation=45,
    colorbar=False
)
axis.set_title("Confusion Matrix - Model A Test Set")
axis.set_xlabel("Predicted Label")
axis.set_ylabel("True Label")
figure.tight_layout()
figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close(figure)

np.savetxt(
    RESULTS_DIR / "confusion_matrix.csv",
    matrix,
    delimiter=",",
    fmt="%d"
)

print(f"Confusion matrix saved to: {OUTPUT_PATH}")
print(f"Raw confusion matrix saved to: {RESULTS_DIR / 'confusion_matrix.csv'}")
