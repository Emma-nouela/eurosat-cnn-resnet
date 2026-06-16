"""Data pipeline: download EuroSAT, build transforms, and create stratified loaders.

By default the dataset is fetched **token-free** from the public Zenodo mirror
(``EuroSAT_RGB.zip``) — no Kaggle account or API token is required. A ``kagglehub``
fallback is kept for anyone who prefers it (``source="kaggle"``). We then build a
*reproducible* stratified train/val/test split with scikit-learn so every run sees the
same partition.
"""
import urllib.request
import zipfile
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


def _download_with_progress(url, dest):
    """Download ``url`` to ``dest`` showing a tqdm progress bar (best-effort)."""
    try:
        from tqdm import tqdm

        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc="EuroSAT_RGB.zip") as bar:
            def hook(block_num, block_size, total_size):
                if total_size > 0:
                    bar.total = total_size
                bar.update(block_size)

            urllib.request.urlretrieve(url, dest, reporthook=hook)
    except ImportError:
        urllib.request.urlretrieve(url, dest)


def download_zenodo():
    """Token-free download of the public EuroSAT RGB archive from Zenodo.

    Idempotent: skips the download/extraction if the data is already present.
    Returns the directory that contains the 10 class subfolders.
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Already extracted? Reuse it.
    try:
        return find_image_root(config.DATA_DIR)
    except FileNotFoundError:
        pass

    zip_path = config.DATA_DIR / "EuroSAT_RGB.zip"
    if not zip_path.exists():
        print(f"Downloading EuroSAT (RGB) from Zenodo — no token required:\n  {config.ZENODO_RGB_URL}")
        _download_with_progress(config.ZENODO_RGB_URL, zip_path)

    print("Extracting EuroSAT_RGB.zip ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(config.DATA_DIR)

    return find_image_root(config.DATA_DIR)


def download_kaggle():
    """Optional fallback: download via kagglehub (requires a Kaggle API token)."""
    import kagglehub

    path = kagglehub.dataset_download(config.KAGGLE_DATASET)
    return find_image_root(path)


def download_dataset(source="zenodo"):
    """Return the image root for EuroSAT, downloading if necessary.

    Parameters
    ----------
    source : {"zenodo", "kaggle"}
        ``"zenodo"`` (default) is token-free. ``"kaggle"`` uses kagglehub and
        requires Kaggle credentials.
    """
    if source == "kaggle":
        return download_kaggle()
    if source == "zenodo":
        try:
            return download_zenodo()
        except Exception as exc:  # noqa: BLE001 - friendly fallback to a manual copy
            if config.DATA_DIR.exists():
                return find_image_root(config.DATA_DIR)
            raise RuntimeError(
                "Zenodo download failed and no local data was found under ./data. "
                f"Original error: {exc}"
            ) from exc
    raise ValueError(f"Unknown source: {source!r} (expected 'zenodo' or 'kaggle').")


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
