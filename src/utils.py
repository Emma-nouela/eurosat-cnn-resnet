"""Reproducibility and plotting helpers shared across the project."""
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from . import config


def set_seed(seed=config.SEED):
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_curves(history, title, save_path=None):
    """Plot training/validation loss and accuracy side by side."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title(f"{title} — Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title(f"{title} — Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_confusion_matrix(y_true, y_pred, classes, title, save_path=None, normalize=True):
    """Render a (optionally row-normalized) confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-9)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=ax, cbar=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def denormalize(img_tensor, mean, std):
    """Undo Normalize so an image tensor can be displayed."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (img_tensor.cpu() * std + mean).clamp(0, 1)


def show_predictions(images, y_true, y_pred, classes, mean, std, title,
                     save_path=None, n=16):
    """Grid of sample test images with predicted vs. true labels (green=correct)."""
    n = min(n, len(images))
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        img = denormalize(images[i], mean, std).permute(1, 2, 0).numpy()
        ax = axes[i]
        ax.imshow(img)
        correct = y_true[i] == y_pred[i]
        ax.set_title(
            f"P: {classes[y_pred[i]]}\nT: {classes[y_true[i]]}",
            color="green" if correct else "red", fontsize=8,
        )
        ax.axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
