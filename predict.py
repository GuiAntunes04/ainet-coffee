import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from config import (
    CLASS_TO_LABEL,
    SUBMISSION_TEMPLATE_PATH,
    TEST_DIR,
)
from pipeline_utils import load_pipeline, resolve_path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Kaggle submission file.")
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
        help="Model pipeline to use when --experiment is not provided.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional path to a .keras model file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path.",
    )
    return parser.parse_args()


def find_image_by_id(image_id):
    matches = [
        path
        for path in TEST_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.stem == str(image_id)
    ]

    if not matches:
        raise FileNotFoundError(f"Image id {image_id} was not found under {TEST_DIR}")

    return matches[0]


def load_template_ids():
    if SUBMISSION_TEMPLATE_PATH.exists():
        template = pd.read_csv(SUBMISSION_TEMPLATE_PATH)
        return template["id"].tolist()

    return sorted(
        path.stem
        for path in TEST_DIR.rglob("*")
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

    model_path = resolve_path(args.model_path) if args.model_path else None
    output_path = resolve_path(args.output) if args.output else None
    pipeline = load_pipeline(
        experiment=args.experiment,
        model_name=args.model,
        model_path=model_path,
        output_path=output_path,
    )

    if not pipeline["path"].exists():
        raise FileNotFoundError(
            f"Model not found: {pipeline['path']}. "
            "Train the experiment first or choose another --experiment."
        )

    model = tf.keras.models.load_model(pipeline["path"], compile=False)
    rows = []

    for image_id in load_template_ids():
        image_path = find_image_by_id(image_id)
        class_name, label, confidence = predict_one(
            model=model,
            image_path=image_path,
            image_size=pipeline["image_size"],
            preprocess=pipeline["preprocess"],
            class_names=pipeline["class_names"],
        )

        rows.append({"id": image_id, "class": label})
        print(f"{image_id:>5} -> {class_name:<10} label={label} confidence={confidence:.4f}")

    submission = pd.DataFrame(rows, columns=["id", "class"])
    pipeline["output"].parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(pipeline["output"], index=False)
    print(f"\nSubmission saved to {pipeline['output']}")


if __name__ == "__main__":
    main()
