import argparse

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from config import BATCH_SIZE, IMAGE_SIZE_CNN
from experiment_utils import (
    best_metric,
    create_experiment_dir,
    min_metric,
    save_experiment_metadata,
)
from utils import build_data_augmentation, load_datasets


def parse_filters(value):
    try:
        filters = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated integers, like 16,32,64") from exc

    if not filters or any(filter_count <= 0 for filter_count in filters):
        raise argparse.ArgumentTypeError("Filters must be positive integers.")

    return filters


def parse_image_size(value):
    size = int(value)
    if size <= 0:
        raise argparse.ArgumentTypeError("Image size must be positive.")
    return (size, size)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN experiment.")
    parser.add_argument(
        "--experiment-name",
        default="cnn_exp001_baseline",
        help="Folder name inside experiments/.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--image-size", type=parse_image_size, default=IMAGE_SIZE_CNN)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--filters", type=parse_filters, default=[16, 32, 64])
    parser.add_argument("--dense-units", type=int, default=32)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument(
        "--pooling",
        choices=["flatten", "gap"],
        default="flatten",
        help="Use flatten or global average pooling after convolution blocks.",
    )
    parser.add_argument(
        "--augmentation",
        choices=["none", "light", "medium", "strong"],
        default="medium",
    )
    return parser.parse_args()


def build_model(args, class_count):
    kernel_regularizer = (
        regularizers.l2(args.l2)
        if args.l2 > 0
        else None
    )

    model_layers = [
        layers.Input(shape=(*args.image_size, 3)),
        build_data_augmentation(args.augmentation),
    ]

    for filter_count in args.filters:
        model_layers.extend([
            layers.Conv2D(
                filter_count,
                (3, 3),
                activation="relu",
                kernel_regularizer=kernel_regularizer,
            ),
            layers.MaxPooling2D((2, 2)),
        ])

    if args.pooling == "gap":
        model_layers.append(layers.GlobalAveragePooling2D())
    else:
        model_layers.append(layers.Flatten())

    model_layers.extend([
        layers.Dense(
            args.dense_units,
            activation="relu",
            kernel_regularizer=kernel_regularizer,
        ),
        layers.Dropout(args.dropout),
        layers.Dense(class_count, activation="softmax"),
    ])

    return models.Sequential(model_layers)


def main():
    args = parse_args()
    experiment_dir = create_experiment_dir(args.experiment_name)
    model_path = experiment_dir / "model.keras"

    train_ds, val_ds, class_names = load_datasets(
        image_size=args.image_size,
        batch_size=args.batch_size,
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
        patience=30,
        restore_best_weights=True,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=0.00001,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[checkpoint, early_stop, reduce_lr],
    )

    metadata_path = save_experiment_metadata(
        experiment_dir,
        {
            "experiment_name": args.experiment_name,
            "architecture": "cnn",
            "model_path": str(model_path),
            "image_size": list(args.image_size),
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "dropout": args.dropout,
            "filters": args.filters,
            "dense_units": args.dense_units,
            "l2": args.l2,
            "pooling": args.pooling,
            "augmentation": args.augmentation,
            "best_val_accuracy": best_metric(history, "val_accuracy"),
            "best_val_loss": min_metric(history, "val_loss"),
            "class_names": class_names,
        },
    )

    print(f"Best CNN model saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
