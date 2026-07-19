import argparse

import tensorflow as tf
from tensorflow.keras import layers, models

from config import BATCH_SIZE, IMAGE_SIZE_MOBILENET, SEED
from experiment_utils import (
    best_metric,
    create_experiment_dir,
    min_metric,
    save_experiment_metadata,
)
from utils import configure_reproducibility, load_datasets


def parse_image_size(value):
    size = int(value)
    if size <= 0:
        raise argparse.ArgumentTypeError("Image size must be positive.")
    return (size, size)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a MobileNetV2 experiment.")
    parser.add_argument(
        "--experiment-name",
        default="mobilenet_exp001_baseline",
        help="Folder name inside experiments/.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--image-size", type=parse_image_size, default=IMAGE_SIZE_MOBILENET)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def build_model(args, class_count):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*args.image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    return models.Sequential([
        layers.Input(shape=(*args.image_size, 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(args.dropout),
        layers.Dense(class_count, activation="softmax"),
    ])


def main():
    args = parse_args()
    configure_reproducibility(args.seed)

    experiment_dir = create_experiment_dir(args.experiment_name)
    model_path = experiment_dir / "model.keras"

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    train_ds, val_ds, class_names = load_datasets(
        image_size=args.image_size,
        batch_size=args.batch_size,
        preprocess=preprocess,
        seed=args.seed,
    )

    print("Class order:", class_names)
    print(f"Experiment: {args.experiment_name}")

    model = build_model(args, class_count=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(model_path),
        monitor="val_loss",
        save_best_only=True,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[checkpoint, early_stop],
    )

    metadata_path = save_experiment_metadata(
        experiment_dir,
        {
            "experiment_name": args.experiment_name,
            "architecture": "mobilenet",
            "model_path": str(model_path),
            "image_size": list(args.image_size),
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "dropout": args.dropout,
            "seed": args.seed,
            "best_val_accuracy": best_metric(history, "val_accuracy"),
            "best_val_loss": min_metric(history, "val_loss"),
            "class_names": class_names,
        },
    )

    print(f"Best MobileNet model saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
