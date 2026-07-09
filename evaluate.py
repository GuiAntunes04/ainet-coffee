import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import image_dataset_from_directory

from config import (
    BATCH_SIZE,
    IMAGE_SIZE_CNN,
    IMAGE_SIZE_MOBILENET,
    MODEL_CNN_BEST_PATH,
    MODEL_MOBILENET_PATH,
    TEST_DIR,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained image model.")
    parser.add_argument(
        "--model",
        choices=["cnn", "mobilenet"],
        default="mobilenet",
        help="Model pipeline to evaluate.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional path to a .keras model file.",
    )
    return parser.parse_args()


def get_pipeline(model_name, model_path):
    if model_name == "mobilenet":
        return {
            "path": model_path or MODEL_MOBILENET_PATH,
            "image_size": IMAGE_SIZE_MOBILENET,
            "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input,
        }

    return {
        "path": model_path or MODEL_CNN_BEST_PATH,
        "image_size": IMAGE_SIZE_CNN,
        "preprocess": lambda images: images / 255.0,
    }


def main():
    args = parse_args()
    pipeline = get_pipeline(args.model, args.model_path)

    model = tf.keras.models.load_model(pipeline["path"])

    test_ds = image_dataset_from_directory(
        str(TEST_DIR),
        image_size=pipeline["image_size"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="int",
    )

    class_names = test_ds.class_names
    print("Class order:", class_names)

    test_ds = test_ds.map(
        lambda x, y: (pipeline["preprocess"](x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    loss, accuracy = model.evaluate(test_ds)
    print(f"\nLoss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    ))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
