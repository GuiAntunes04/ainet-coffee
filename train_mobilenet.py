import argparse
from pathlib import Path
from types import SimpleNamespace

import tensorflow as tf
import yaml
from tensorflow.keras import layers, models

from config import BATCH_SIZE, IMAGE_SIZE_MOBILENET, SEED
from experiment_utils import (
    best_metric,
    create_experiment_dir,
    min_metric,
    save_experiment_metadata,
)
from train_cnn import build_loss, parse_bool, parse_manual_class_weights, resolve_class_weights
from utils import configure_reproducibility, load_datasets


DEFAULT_CONFIG = {
    "experiment_name": "mobilenet_exp001_baseline",
    "epochs": 40,
    "batch_size": BATCH_SIZE,
    "image_size": IMAGE_SIZE_MOBILENET,
    "learning_rate": 0.0003,
    "dropout": 0.4,
    "label_smoothing": 0.0,
    "class_weights": False,
    "manual_class_weights": None,
    "train_base_layers": 0,
    "patience": 8,
    "reduce_lr_factor": 0.5,
    "reduce_lr_patience": 4,
    "min_lr": 0.00001,
    "seed": SEED,
}


def parse_image_size(value):
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise argparse.ArgumentTypeError("Image size list must have two values.")
        width, height = int(value[0]), int(value[1])
    else:
        width = height = int(value)

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Image size must be positive.")

    return (width, height)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a MobileNetV2 experiment.")
    parser.add_argument("--config", type=Path, default=None, help="YAML file with MobileNet settings.")
    parser.add_argument("--experiment-name", default=None, help="Folder name inside experiments/.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=parse_image_size, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--class-weights", type=parse_bool, default=None)
    parser.add_argument("--train-base-layers", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--reduce-lr-factor", type=float, default=None)
    parser.add_argument("--reduce-lr-patience", type=int, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_yaml_config(config_path):
    if config_path is None:
        return {}

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"YAML config must be a mapping: {config_path}")

    return loaded


def normalize_config(config):
    normalized = {**config}
    normalized["image_size"] = parse_image_size(normalized["image_size"])
    normalized["class_weights"] = parse_bool(normalized["class_weights"])
    normalized["manual_class_weights"] = parse_manual_class_weights(
        normalized.get("manual_class_weights")
    )

    if normalized["epochs"] <= 0:
        raise ValueError("epochs must be positive.")
    if normalized["batch_size"] <= 0:
        raise ValueError("batch_size must be positive.")
    if normalized["train_base_layers"] < 0:
        raise ValueError("train_base_layers must be >= 0.")
    if normalized["patience"] <= 0:
        raise ValueError("patience must be positive.")
    if normalized["reduce_lr_patience"] <= 0:
        raise ValueError("reduce_lr_patience must be positive.")
    if not 0 <= normalized["label_smoothing"] < 1:
        raise ValueError("label_smoothing must be >= 0 and < 1.")

    return normalized


def build_config(cli_args):
    config = {**DEFAULT_CONFIG}
    config.update(load_yaml_config(cli_args.config))

    for key, value in vars(cli_args).items():
        if key == "config" or value is None:
            continue
        config[key] = value

    return SimpleNamespace(**normalize_config(config))


def set_base_trainability(base_model, train_base_layers):
    if train_base_layers <= 0:
        base_model.trainable = False
        return

    base_model.trainable = True
    trainable_from = max(0, len(base_model.layers) - train_base_layers)

    for index, layer in enumerate(base_model.layers):
        layer.trainable = index >= trainable_from and not isinstance(layer, layers.BatchNormalization)


def build_model(args, class_count):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*args.image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    set_base_trainability(base_model, args.train_base_layers)

    return models.Sequential([
        layers.Input(shape=(*args.image_size, 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(args.dropout),
        layers.Dense(class_count, activation="softmax"),
    ])


def config_metadata(args):
    return {
        "experiment_name": args.experiment_name,
        "architecture": "mobilenet",
        "image_size": list(args.image_size),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "class_weights": args.class_weights,
        "manual_class_weights": args.manual_class_weights,
        "train_base_layers": args.train_base_layers,
        "patience": args.patience,
        "reduce_lr_factor": args.reduce_lr_factor,
        "reduce_lr_patience": args.reduce_lr_patience,
        "min_lr": args.min_lr,
        "seed": args.seed,
    }


def main():
    args = build_config(parse_args())
    configure_reproducibility(args.seed)

    experiment_dir = create_experiment_dir(args.experiment_name)
    model_path = experiment_dir / "model.keras"

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    train_ds, val_ds, class_names, automatic_class_weights = load_datasets(
        image_size=args.image_size,
        batch_size=args.batch_size,
        preprocess=preprocess,
        seed=args.seed,
        return_class_weights=True,
    )
    class_weights = resolve_class_weights(
        class_names,
        automatic_class_weights,
        args.manual_class_weights,
    )

    print("Class order:", class_names)
    print(f"Experiment: {args.experiment_name}")
    if args.class_weights:
        print("Class weights:", {
            class_names[class_index]: weight
            for class_index, weight in class_weights.items()
        })

    model = build_model(args, class_count=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=build_loss(args, class_count=len(class_names)),
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
        patience=args.patience,
        restore_best_weights=True,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=args.reduce_lr_factor,
        patience=args.reduce_lr_patience,
        min_lr=args.min_lr,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[checkpoint, early_stop, reduce_lr],
        class_weight=class_weights if args.class_weights else None,
    )

    metadata_path = save_experiment_metadata(
        experiment_dir,
        {
            **config_metadata(args),
            "model_path": str(model_path),
            "best_val_accuracy": best_metric(history, "val_accuracy"),
            "best_val_loss": min_metric(history, "val_loss"),
            "class_names": class_names,
            "class_weight_values": {
                class_names[class_index]: weight
                for class_index, weight in class_weights.items()
            } if args.class_weights else None,
        },
    )

    print(f"Best MobileNet model saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
