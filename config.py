from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_DIR = PROJECT_ROOT / "Training set-kaggle"
TEST_DIR = PROJECT_ROOT / "test-kaggle"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

SUBMISSION_TEMPLATE_PATH = PROJECT_ROOT / "submission_template.csv"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"

MODEL_CNN_BEST_PATH = EXPERIMENTS_DIR / "cnn_exp001_baseline" / "model.keras"
MODEL_MLP_PATH = EXPERIMENTS_DIR / "mlp_exp001_baseline" / "model.keras"
MODEL_MOBILENET_PATH = EXPERIMENTS_DIR / "mobilenet_exp001_baseline" / "model.keras"

IMAGE_SIZE_CNN = (64, 64)
IMAGE_SIZE_MOBILENET = (160, 160)
BATCH_SIZE = 32
SEED = 42

CLASS_TO_LABEL = {
    "Verde": 1,
    "Verde cana": 2,
    "Cereja": 3,
    "Passa": 4,
    "Seco": 5,
}

CLASS_NAMES = ["Cereja", "Passa", "Seco", "Verde", "Verde cana"]
LABEL_TO_CLASS = {label: class_name for class_name, label in CLASS_TO_LABEL.items()}
