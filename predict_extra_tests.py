import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from config import CLASS_NAMES, CLASS_TO_LABEL, PROJECT_ROOT


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict images from extra-tests.")
    parser.add_argument(
        "--experiment",
        type=Path,
        required=True,
        help="Experiment folder, like experiments/cnn_exp002_img96.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "extra-tests",
        help="Folder with external test images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to extra-tests/predictions.csv.",
    )
    return parser.parse_args()


def resolve_path(path):
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_experiment(experiment_path):
    experiment_dir = resolve_path(experiment_path)
    metadata_path = experiment_dir / "metadata.json"
    model_path = experiment_dir / "model.keras"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    architecture = metadata["architecture"]
    if architecture == "mobilenet":
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    else:
        preprocess = lambda img: img / 255.0

    return {
        "model_path": model_path,
        "image_size": tuple(metadata["image_size"]),
        "preprocess": preprocess,
        "class_names": metadata.get("class_names", CLASS_NAMES),
    }


def iter_images(input_dir):
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def predict_one(model, image_path, image_size, preprocess, class_names):
    img = image.load_img(image_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = preprocess(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0]
    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    class_name = class_names[class_index]

    return class_name, CLASS_TO_LABEL[class_name], confidence


def main():
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_path = resolve_path(args.output) if args.output else input_dir / "predictions.csv"

    experiment = load_experiment(args.experiment)
    model = tf.keras.models.load_model(experiment["model_path"])

    rows = []
    for image_path in iter_images(input_dir):
        class_name, label, confidence = predict_one(
            model=model,
            image_path=image_path,
            image_size=experiment["image_size"],
            preprocess=experiment["preprocess"],
            class_names=experiment["class_names"],
        )
        rows.append({
            "image": image_path.relative_to(input_dir).as_posix(),
            "class_name": class_name,
            "class": label,
            "confidence": confidence,
        })
        print(f"{image_path.name:<60} -> {class_name:<10} label={label} confidence={confidence:.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pd.DataFrame(rows).to_csv(output_path, index=False)
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")
        pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    main()
