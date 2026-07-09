import json
from datetime import datetime

from config import EXPERIMENTS_DIR


def create_experiment_dir(experiment_name):
    experiment_dir = EXPERIMENTS_DIR / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def save_experiment_metadata(experiment_dir, metadata):
    metadata_path = experiment_dir / "metadata.json"
    metadata = {
        **metadata,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return metadata_path


def best_metric(history, metric_name):
    values = history.history.get(metric_name, [])
    if not values:
        return None
    return max(values)


def min_metric(history, metric_name):
    values = history.history.get(metric_name, [])
    if not values:
        return None
    return min(values)
