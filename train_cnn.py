import argparse
from pathlib import Path
from types import SimpleNamespace

import tensorflow as tf
import yaml
from tensorflow.keras import layers, models, regularizers

from config import BATCH_SIZE, IMAGE_SIZE_CNN, SEED
from experiment_utils import (
    best_metric,
    create_experiment_dir,
    min_metric,
    save_experiment_metadata,
)
from utils import configure_reproducibility, load_datasets


DEFAULT_CONFIG = {
    "experiment_name": "cnn_exp001_baseline",
    "epochs": 40,
    "batch_size": BATCH_SIZE,
    "image_size": IMAGE_SIZE_CNN,
    "learning_rate": 0.001,
    "dropout": 0.5,
    "filters": [16, 32, 64],
    "dense_units": 32,
    "l2": 0.0,
    "pooling": "flatten",
    "augmentation": "medium",
    "kernel_size": 3,
    "conv_dropout": 0.0,
    "batch_normalization": False,
    "activation": "relu",
    "optimizer": "adam",
    "weight_decay": 0.0,
    "label_smoothing": 0.0,
    "patience": 20,
    "reduce_lr_factor": 0.5,
    "reduce_lr_patience": 4,
    "min_lr": 0.00001,
    "random_brightness": 0.0,
    "random_contrast": None,
    "random_translation": 0.0,
    "random_zoom": None,
    "seed": SEED,
    "class_weights": False,
    "manual_class_weights": None,
}


AUGMENTATION_DEFAULTS = {
    "none": {"rotation": 0.0, "zoom": 0.0, "contrast": 0.0},
    "light": {"rotation": 0.05, "zoom": 0.05, "contrast": 0.0},
    "medium": {"rotation": 0.10, "zoom": 0.10, "contrast": 0.0},
    "strong": {"rotation": 0.15, "zoom": 0.15, "contrast": 0.15},
}


def parse_filters(value):
    if isinstance(value, list):
        filters = [int(item) for item in value]
    else:
        try:
            filters = [int(item.strip()) for item in str(value).split(",")]
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Use comma-separated integers, like 16,32,64") from exc

    if not filters or any(filter_count <= 0 for filter_count in filters):
        raise argparse.ArgumentTypeError("Filters must be positive integers.")

    return filters


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


def parse_optional_float(value):
    if value is None or str(value).lower() == "none":
        return None
    return float(value)


def parse_bool(value):
    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "sim"}:
        return True
    if value in {"0", "false", "no", "n", "nao"}:
        return False

    raise argparse.ArgumentTypeError("Use true or false.")


def parse_manual_class_weights(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("manual_class_weights must be a mapping of class names to weights.")

    weights = {}
    for class_name, weight in value.items():
        weight = float(weight)
        if weight <= 0:
            raise argparse.ArgumentTypeError("manual_class_weights values must be positive.")
        weights[str(class_name)] = weight

    return weights


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN experiment.")
    parser.add_argument("--config", type=Path, default=None, help="YAML file with CNN training settings.")
    parser.add_argument("--experiment-name", default=None, help="Folder name inside experiments/.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=parse_image_size, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--filters", type=parse_filters, default=None)
    parser.add_argument("--dense-units", type=int, default=None)
    parser.add_argument("--l2", type=float, default=None)
    parser.add_argument("--pooling", choices=["flatten", "gap"], default=None)
    parser.add_argument("--augmentation", choices=["none", "light", "medium", "strong"], default=None)
    parser.add_argument("--kernel-size", type=int, default=None)
    parser.add_argument("--conv-dropout", type=float, default=None)
    parser.add_argument("--batch-normalization", type=parse_bool, default=None)
    parser.add_argument("--activation", choices=["relu", "elu", "swish"], default=None)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "rmsprop", "sgd"], default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--reduce-lr-factor", type=float, default=None)
    parser.add_argument("--reduce-lr-patience", type=int, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--random-brightness", type=float, default=None)
    parser.add_argument("--random-contrast", type=parse_optional_float, default=None)
    parser.add_argument("--random-translation", type=float, default=None)
    parser.add_argument("--random-zoom", type=parse_optional_float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--class-weights", type=parse_bool, default=None)
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
    normalized["filters"] = parse_filters(normalized["filters"])
    normalized["batch_normalization"] = parse_bool(normalized["batch_normalization"])
    normalized["class_weights"] = parse_bool(normalized["class_weights"])
    normalized["manual_class_weights"] = parse_manual_class_weights(
        normalized.get("manual_class_weights")
    )

    for key in ("random_contrast", "random_zoom"):
        normalized[key] = parse_optional_float(normalized[key])

    if normalized["kernel_size"] <= 0:
        raise ValueError("kernel_size must be positive.")
    if normalized["patience"] <= 0:
        raise ValueError("patience must be positive.")
    if normalized["reduce_lr_patience"] <= 0:
        raise ValueError("reduce_lr_patience must be positive.")
    if not 0 <= normalized["label_smoothing"] < 1:
        raise ValueError("label_smoothing must be >= 0 and < 1.")

    return normalized


def resolve_class_weights(class_names, automatic_class_weights, manual_class_weights):
    if manual_class_weights is None:
        return automatic_class_weights

    unknown_classes = sorted(set(manual_class_weights) - set(class_names))
    if unknown_classes:
        raise ValueError(
            "Unknown classes in manual_class_weights: "
            f"{unknown_classes}. Valid classes: {class_names}"
        )

    return {
        class_index: float(manual_class_weights.get(class_name, 1.0))
        for class_index, class_name in enumerate(class_names)
    }


def build_config(cli_args):
    config = {**DEFAULT_CONFIG}
    config.update(load_yaml_config(cli_args.config))

    for key, value in vars(cli_args).items():
        if key == "config" or value is None:
            continue
        config[key] = value

    return SimpleNamespace(**normalize_config(config))


def build_cnn_augmentation(args):
    if args.augmentation == "none":
        return None

    if args.augmentation not in AUGMENTATION_DEFAULTS:
        raise ValueError(f"Unknown augmentation level: {args.augmentation}")

    defaults = AUGMENTATION_DEFAULTS[args.augmentation]
    zoom = defaults["zoom"] if args.random_zoom is None else args.random_zoom
    contrast = defaults["contrast"] if args.random_contrast is None else args.random_contrast

    augmentation_layers = [
        layers.RandomFlip("horizontal", seed=args.seed),
    ]

    if defaults["rotation"] > 0:
        augmentation_layers.append(layers.RandomRotation(defaults["rotation"], seed=args.seed + 1))
    if zoom and zoom > 0:
        augmentation_layers.append(layers.RandomZoom(zoom, seed=args.seed + 2))
    if args.random_translation > 0:
        augmentation_layers.append(
            layers.RandomTranslation(
                args.random_translation,
                args.random_translation,
                seed=args.seed + 3,
            )
        )
    if contrast and contrast > 0:
        augmentation_layers.append(layers.RandomContrast(contrast, seed=args.seed + 4))
    if args.random_brightness > 0:
        brightness_layer = getattr(layers, "RandomBrightness", None)
        if brightness_layer is None:
            raise RuntimeError("RandomBrightness is not available in this TensorFlow/Keras version.")
        augmentation_layers.append(
            brightness_layer(
                args.random_brightness,
                value_range=(0.0, 1.0),
                seed=args.seed + 5,
            )
        )

    return tf.keras.Sequential(augmentation_layers, name="data_augmentation")


def build_optimizer(args):
    if args.optimizer == "adamw":
        return tf.keras.optimizers.AdamW(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=args.learning_rate)
    if args.optimizer == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=0.9)
    return tf.keras.optimizers.Adam(learning_rate=args.learning_rate)


@tf.keras.utils.register_keras_serializable(package="AINET")
class SparseCategoricalCrossentropyWithLabelSmoothing(tf.keras.losses.Loss):
    def __init__(self, class_count, label_smoothing=0.0, name=None):
        super().__init__(name=name or "sparse_categorical_crossentropy_with_label_smoothing")
        self.class_count = int(class_count)
        self.label_smoothing = float(label_smoothing)
        self.categorical_loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=self.label_smoothing
        )

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true = tf.one_hot(y_true, depth=self.class_count)
        return self.categorical_loss(y_true, y_pred)

    def get_config(self):
        return {
            "class_count": self.class_count,
            "label_smoothing": self.label_smoothing,
            "name": self.name,
        }


def build_loss(args, class_count):
    if args.label_smoothing <= 0:
        return "sparse_categorical_crossentropy"

    return SparseCategoricalCrossentropyWithLabelSmoothing(
        class_count=class_count,
        label_smoothing=args.label_smoothing,
    )


def build_model(args, class_count):
    kernel_regularizer = (
        regularizers.l2(args.l2)
        if args.l2 > 0
        else None
    )

    model_layers = [
        layers.Input(shape=(*args.image_size, 3)),
    ]

    augmentation = build_cnn_augmentation(args)
    if augmentation is not None:
        model_layers.append(augmentation)

    for filter_count in args.filters:
        model_layers.append(
            layers.Conv2D(
                filter_count,
                (args.kernel_size, args.kernel_size),
                padding="valid",
                use_bias=not args.batch_normalization,
                kernel_regularizer=kernel_regularizer,
            )
        )
        if args.batch_normalization:
            model_layers.append(layers.BatchNormalization())

        model_layers.extend([
            layers.Activation(args.activation),
            layers.MaxPooling2D((2, 2)),
        ])

        if args.conv_dropout > 0:
            model_layers.append(layers.Dropout(args.conv_dropout))

    if args.pooling == "gap":
        model_layers.append(layers.GlobalAveragePooling2D())
    else:
        model_layers.append(layers.Flatten())

    model_layers.extend([
        layers.Dense(
            args.dense_units,
            activation=args.activation,
            kernel_regularizer=kernel_regularizer,
        ),
        layers.Dropout(args.dropout),
        layers.Dense(class_count, activation="softmax"),
    ])

    return models.Sequential(model_layers)


def config_metadata(args):
    return {
        "experiment_name": args.experiment_name,
        "architecture": "cnn",
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
        "kernel_size": args.kernel_size,
        "conv_dropout": args.conv_dropout,
        "batch_normalization": args.batch_normalization,
        "activation": args.activation,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "patience": args.patience,
        "reduce_lr_factor": args.reduce_lr_factor,
        "reduce_lr_patience": args.reduce_lr_patience,
        "min_lr": args.min_lr,
        "random_brightness": args.random_brightness,
        "random_contrast": args.random_contrast,
        "random_translation": args.random_translation,
        "random_zoom": args.random_zoom,
        "seed": args.seed,
        "class_weights": args.class_weights,
        "manual_class_weights": args.manual_class_weights,
    }


def main():
    args = build_config(parse_args())
    configure_reproducibility(args.seed)

    experiment_dir = create_experiment_dir(args.experiment_name)
    model_path = experiment_dir / "model.keras"

    train_ds, val_ds, class_names, automatic_class_weights = load_datasets(
        image_size=args.image_size,
        batch_size=args.batch_size,
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
        optimizer=build_optimizer(args),
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

    print(f"Best CNN model saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
