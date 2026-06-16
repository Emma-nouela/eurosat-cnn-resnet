"""Central configuration: paths, class names, normalization, and hyperparameters.

Keeping every "magic number" in one place makes the experiments reproducible and
easy to tweak from the notebook or the command line.
"""
from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"          # local fallback if kagglehub is unavailable
RESULTS_DIR = ROOT / "results"    # figures + metrics are written here
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
KAGGLE_DATASET = "apollo2506/eurosat-dataset"

# EuroSAT (RGB) classes — alphabetical, matching torchvision.ImageFolder ordering.
CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
NUM_CLASSES = len(CLASSES)

IMG_SIZE_CNN = 64       # native EuroSAT tile size — used by the custom CNN
IMG_SIZE_RESNET = 224   # ImageNet input size — used by the pretrained ResNet50

# Normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]   # for the pretrained ResNet50
IMAGENET_STD = [0.229, 0.224, 0.225]
EUROSAT_MEAN = [0.3444, 0.3803, 0.4078]  # approximate EuroSAT RGB stats
EUROSAT_STD = [0.2037, 0.1366, 0.1148]

# --------------------------------------------------------------------------- #
# Reproducible stratified split
# --------------------------------------------------------------------------- #
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2

# Sensible per-model defaults (the notebook can override these with Optuna's best params).
DEFAULTS = {
    "cnn": {
        "batch_size": 64,
        "epochs": 25,
        "lr": 1e-3,
        "optimizer": "adam",
        "dropout": 0.4,
        "img_size": IMG_SIZE_CNN,
    },
    "resnet": {
        "batch_size": 64,
        "epochs": 12,
        "lr": 1e-4,
        "optimizer": "adam",
        "dropout": 0.0,  # not used by the ResNet head, kept for a uniform interface
        "img_size": IMG_SIZE_RESNET,
    },
}
