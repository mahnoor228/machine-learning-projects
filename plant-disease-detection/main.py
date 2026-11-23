"""
Plant Disease Detection
=======================
Train an image classifier that spots tomato leaf diseases from photos.

Dataset: PlantVillage (CC0, Hughes and Salathe 2015). This project uses
a small subset: 90 images each of 3 tomato classes (healthy, bacterial
spot, early blight), downloaded by download_data.py from the official
public GitHub repo. Run that script first, or it runs automatically
if the data folder is empty.

Model: a small convolutional neural network (TensorFlow/Keras),
3 conv blocks, trained for 12 epochs on 128x128 images, with
horizontal flips and small rotations as data augmentation.
The train/validation split is stratified, so every class appears
in both sets.
"""

import glob
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CLASSES = ["Tomato___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight"]
IMG_SIZE = 128
EPOCHS = 12


def ensure_data():
    have = len(glob.glob(os.path.join(DATA_DIR, "*", "*.JPG")))
    if have < 100:
        print("Data not found, running download_data.py ...")
        subprocess.run([sys.executable, "download_data.py"], check=True)


def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img / 255.0, label


def make_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=42)
    return ds.batch(32).prefetch(tf.data.AUTOTUNE)


def main():
    ensure_data()

    paths, labels = [], []
    for i, cls in enumerate(CLASSES):
        for p in sorted(glob.glob(os.path.join(DATA_DIR, cls, "*.JPG"))):
            paths.append(p)
            labels.append(i)
    print(f"Loaded {len(paths)} images across {len(CLASSES)} classes: {CLASSES}")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Training images: {len(train_paths)} | Validation images: {len(val_paths)}")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {train_labels.count(i)} train, {val_labels.count(i)} val")

    train_ds = make_dataset(train_paths, train_labels, shuffle=True)
    val_ds = make_dataset(val_paths, val_labels)

    model = tf.keras.Sequential(
        [
            # Augmentation only applies during training, not evaluation
            tf.keras.layers.RandomFlip("horizontal", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\nValidation accuracy: {acc:.4f} | Validation loss: {loss:.4f}")

    y_true, y_pred = [], []
    for images, labels in val_ds:
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(model.predict(images, verbose=0), axis=1))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4, zero_division=0))

    plt.figure(figsize=(7, 5))
    plt.plot(history.history["accuracy"], "o-", label="train accuracy")
    plt.plot(history.history["val_accuracy"], "o-", label="validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=120)
    print("Saved plot: training_history.png")


if __name__ == "__main__":
    main()
