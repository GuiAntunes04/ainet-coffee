import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

from config import BATCH_SIZE, IMAGE_SIZE_CNN, SEED, TRAIN_DIR


AUTOTUNE = tf.data.AUTOTUNE

normalization = layers.Rescaling(1.0 / 255)

def build_data_augmentation(level="medium"):
    if level == "none":
        return None

    settings = {
        "light": {"rotation": 0.05, "zoom": 0.05, "contrast": None},
        "medium": {"rotation": 0.10, "zoom": 0.10, "contrast": None},
        "strong": {"rotation": 0.15, "zoom": 0.15, "contrast": 0.15},
    }

    if level not in settings:
        raise ValueError(f"Unknown augmentation level: {level}")

    config = settings[level]
    augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(config["rotation"]),
        layers.RandomZoom(config["zoom"]),
    ]

    if config["contrast"] is not None:
        augmentation_layers.append(layers.RandomContrast(config["contrast"]))

    return tf.keras.Sequential(augmentation_layers, name="data_augmentation")


data_augmentation = build_data_augmentation("medium")


def load_datasets(image_size=IMAGE_SIZE_CNN, batch_size=BATCH_SIZE, preprocess=None):
    train_ds = image_dataset_from_directory(
        str(TRAIN_DIR),
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
    )

    val_ds = image_dataset_from_directory(
        str(TRAIN_DIR),
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
    )

    class_names = train_ds.class_names
    transform = preprocess if preprocess is not None else normalization

    train_ds = train_ds.map(
        lambda x, y: (transform(x), y),
        num_parallel_calls=AUTOTUNE,
    )

    val_ds = val_ds.map(
        lambda x, y: (transform(x), y),
        num_parallel_calls=AUTOTUNE,
    )

    return train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE), class_names
