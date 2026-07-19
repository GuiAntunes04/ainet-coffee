import argparse

import tensorflow as tf
from tensorflow.keras import layers, models

from config import BATCH_SIZE, IMAGE_SIZE_CNN, SEED
from experiment_utils import (
    best_metric,
    create_experiment_dir,
    min_metric,
    save_experiment_metadata,
)
from utils import configure_reproducibility, load_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Train an MLP experiment.")
    parser.add_argument(
        "--experiment-name",
        default="mlp_exp001_baseline",
        help="Folder name inside experiments/.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE_CNN[0])
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_reproducibility(args.seed)

    experiment_dir = create_experiment_dir(args.experiment_name)
    model_path = experiment_dir / "model.keras"
    image_size = (args.image_size, args.image_size)

    train_ds, val_ds, class_names = load_datasets(
        image_size=image_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print("Class order:", class_names)
    print(f"Experiment: {args.experiment_name}")

    model = models.Sequential([
        layers.Input(shape=(*image_size, 3)),
        layers.Flatten(),

        layers.Dense(32, activation="relu"),
        layers.Dropout(args.dropout),

        layers.Dense(16, activation="relu"),
        layers.Dropout(args.dropout),

        layers.Dense(len(class_names), activation="softmax"),
    ])

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

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[checkpoint],
    )

    metadata_path = save_experiment_metadata(
        experiment_dir,
        {
            "experiment_name": args.experiment_name,
            "architecture": "mlp",
            "model_path": str(model_path),
            "image_size": list(image_size),
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "dropout": args.dropout,
            "seed": args.seed,
            "dense_units": [32, 16],
            "best_val_accuracy": best_metric(history, "val_accuracy"),
            "best_val_loss": min_metric(history, "val_loss"),
            "class_names": class_names,
        },
    )

    print(f"Best MLP model saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
