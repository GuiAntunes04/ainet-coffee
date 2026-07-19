import json

import tensorflow as tf

from config import (
    CLASS_NAMES,
    IMAGE_SIZE_CNN,
    IMAGE_SIZE_MOBILENET,
    MODEL_CNN_BEST_PATH,
    MODEL_MOBILENET_PATH,
    PROJECT_ROOT,
    SUBMISSION_PATH,
)


def resolve_path(path):
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def preprocess_for_architecture(architecture):
    if architecture == "mobilenet":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    return lambda images: images / 255.0


def load_pipeline(experiment=None, model_name="cnn", model_path=None, output_path=None):
    if experiment is not None:
        experiment_dir = resolve_path(experiment)
        metadata_path = experiment_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        architecture = metadata["architecture"]
        return {
            "path": experiment_dir / "model.keras",
            "image_size": tuple(metadata["image_size"]),
            "preprocess": preprocess_for_architecture(architecture),
            "experiment_name": metadata.get("experiment_name", experiment_dir.name),
            "metadata": metadata,
            "output": output_path or experiment_dir / "submission.csv",
            "class_names": metadata.get("class_names", CLASS_NAMES),
        }

    if model_name == "mobilenet":
        return {
            "path": model_path or MODEL_MOBILENET_PATH,
            "image_size": IMAGE_SIZE_MOBILENET,
            "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input,
            "experiment_name": None,
            "metadata": {},
            "output": output_path or SUBMISSION_PATH,
            "class_names": CLASS_NAMES,
        }

    return {
        "path": model_path or MODEL_CNN_BEST_PATH,
        "image_size": IMAGE_SIZE_CNN,
        "preprocess": lambda images: images / 255.0,
        "experiment_name": None,
        "metadata": {},
        "output": output_path or SUBMISSION_PATH,
        "class_names": CLASS_NAMES,
    }
