import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from config import BATCH_SIZE, IMAGE_SIZE_CNN, SEED, TRAIN_DIR


AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

normalization = layers.Rescaling(1.0 / 255)


def configure_reproducibility(seed=SEED):
    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as exc:
        print(f"Could not enable TensorFlow deterministic ops: {exc}")


def with_deterministic_options(dataset):
    options = tf.data.Options()
    options.deterministic = True
    return dataset.with_options(options)


def list_image_paths_and_labels(data_dir=TRAIN_DIR):
    class_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    class_names = [path.name for path in class_dirs]
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}

    image_paths = []
    labels = []

    for class_dir in class_dirs:
        class_images = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        image_paths.extend(str(path) for path in class_images)
        labels.extend([class_to_index[class_dir.name]] * len(class_images))

    return image_paths, labels, class_names


def load_image(path, label, image_size):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size)
    return image, label


def build_dataset(image_paths, labels, image_size, batch_size, preprocess, shuffle, seed):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda path, label: load_image(path, label, image_size),
        num_parallel_calls=AUTOTUNE,
    )
    dataset = dataset.batch(batch_size)

    transform = preprocess if preprocess is not None else normalization
    dataset = dataset.map(
        lambda images, labels: (transform(images), labels),
        num_parallel_calls=AUTOTUNE,
    )

    return with_deterministic_options(dataset).prefetch(AUTOTUNE)


def compute_class_weights(labels, class_count):
    total_count = len(labels)
    class_weights = {}

    for class_index in range(class_count):
        class_count_in_split = labels.count(class_index)
        if class_count_in_split == 0:
            class_weights[class_index] = 1.0
            continue

        class_weights[class_index] = total_count / (class_count * class_count_in_split)

    return class_weights


def load_datasets(
    image_size=IMAGE_SIZE_CNN,
    batch_size=BATCH_SIZE,
    preprocess=None,
    seed=SEED,
    return_class_weights=False,
):
    image_paths, labels, class_names = list_image_paths_and_labels()

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )

    print(f"Found {len(image_paths)} files belonging to {len(class_names)} classes.")
    print(f"Using {len(train_paths)} files for training.")
    print(f"Using {len(val_paths)} files for validation.")
    class_weights = compute_class_weights(train_labels, len(class_names))

    train_ds = build_dataset(
        train_paths,
        train_labels,
        image_size=image_size,
        batch_size=batch_size,
        preprocess=preprocess,
        shuffle=True,
        seed=seed,
    )

    val_ds = build_dataset(
        val_paths,
        val_labels,
        image_size=image_size,
        batch_size=batch_size,
        preprocess=preprocess,
        shuffle=False,
        seed=seed,
    )

    if return_class_weights:
        return train_ds, val_ds, class_names, class_weights

    return train_ds, val_ds, class_names
