import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import image_dataset_from_directory

from config import (
    BATCH_SIZE,
    TEST_DIR,
)
from pipeline_utils import load_pipeline, resolve_path
from train_cnn import build_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained image model.")
    parser.add_argument(
        "--experiment",
        type=Path,
        default=None,
        help="Experiment folder, like experiments/cnn_exp002_img96.",
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "mobilenet"],
        default="cnn",
        help="Model pipeline to evaluate when --experiment is not provided.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional path to a .keras model file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = resolve_path(args.model_path) if args.model_path else None
    pipeline = load_pipeline(
        experiment=args.experiment,
        model_name=args.model,
        model_path=model_path,
    )

    if pipeline["experiment_name"]:
        print(f"Experiment: {pipeline['experiment_name']}")

    if not pipeline["path"].exists():
        raise FileNotFoundError(
            f"Model not found: {pipeline['path']}. "
            "Train the experiment first or choose another --experiment."
        )

    model = tf.keras.models.load_model(pipeline["path"], compile=False)

    test_ds = image_dataset_from_directory(
        str(TEST_DIR),
        image_size=pipeline["image_size"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="int",
    )

    class_names = test_ds.class_names
    print("Class order:", class_names)

    metadata = pipeline.get("metadata", {})
    loss_args = argparse.Namespace(
        label_smoothing=metadata.get("label_smoothing", 0.0)
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=build_loss(loss_args, class_count=len(class_names)),
        metrics=["accuracy"],
    )

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
