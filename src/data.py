"""Data pipeline: download EuroSAT, build transforms, and create stratified loaders.

The dataset is fetched with ``kagglehub`` (works on Colab and locally once Kaggle
credentials are configured). We then build a *reproducible* stratified
train/val/test split with scikit-learn so every run sees the same partition.
"""
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from . import config


def find_image_root(base_path):
    """Return the directory that directly contains the 10 EuroSAT class subfolders.

    The Kaggle archive nests the images a few levels deep, so we search downwards
    for the folder whose subdirectories match our expected class names.
    """
    base = Path(base_path)
    expected = set(config.CLASSES)
    candidates = [base] + [p for p in base.rglob("*") if p.is_dir()]
    for cand in candidates:
        subdirs = {p.name for p in cand.iterdir() if p.is_dir()}
        if expected.issubset(subdirs):
            return cand
    raise FileNotFoundError(
        f"Could not locate the EuroSAT class folders under {base}. "
        f"Expected subfolders: {sorted(expected)}"
    )


def download_dataset():
    """Download the dataset via kagglehub and return its image root.

    Falls back to ``config.DATA_DIR`` if the data was placed there manually.
    """
    try:
        import kagglehub

        path = kagglehub.dataset_download(config.KAGGLE_DATASET)
    except Exception as exc:  # noqa: BLE001 - we want a friendly fallback message
        if config.DATA_DIR.exists():
            path = config.DATA_DIR
        else:
            raise RuntimeError(
                "Dataset download failed and no local data was found. Configure your "
                "Kaggle credentials, or place the extracted dataset under ./data. "
                f"Original error: {exc}"
            ) from exc
    return find_image_root(path)


def build_transforms(img_size, train, mean, std):
    """Compose torchvision transforms; augmentation is applied only for training."""
    steps = [transforms.Resize((img_size, img_size))]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),     # satellite tiles are orientation-agnostic
            transforms.RandomRotation(15),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(steps)


def get_dataloaders(model_type="cnn", batch_size=None, img_root=None):
    """Build train/val/test dataloaders for the requested model type.

    Returns
    -------
    (train_loader, val_loader, test_loader, class_names)
    """
    cfg = config.DEFAULTS[model_type]
    batch_size = batch_size or cfg["batch_size"]
    img_size = cfg["img_size"]

    if model_type == "resnet":
        mean, std = config.IMAGENET_MEAN, config.IMAGENET_STD
    else:
        mean, std = config.EUROSAT_MEAN, config.EUROSAT_STD

    if img_root is None:
        img_root = download_dataset()

    # A transform-free dataset just to read targets and define the split indices.
    base = datasets.ImageFolder(str(img_root))
    targets = np.array(base.targets)
    indices = np.arange(len(targets))

    # Stratified 70 / 15 / 15 split (val+test carved out first, then halved).
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(config.VAL_RATIO + config.TEST_RATIO),
        stratify=targets,
        random_state=config.SEED,
    )
    rel_test = config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=rel_test,
        stratify=targets[temp_idx],
        random_state=config.SEED,
    )

    # Separate datasets so train gets augmentation while val/test stay deterministic.
    train_ds = datasets.ImageFolder(str(img_root), transform=build_transforms(img_size, True, mean, std))
    eval_ds = datasets.ImageFolder(str(img_root), transform=build_transforms(img_size, False, mean, std))

    train_loader = DataLoader(
        Subset(train_ds, train_idx), batch_size=batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(eval_ds, val_idx), batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(eval_ds, test_idx), batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    return train_loader, val_loader, test_loader, base.classes
