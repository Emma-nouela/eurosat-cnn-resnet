"""Evaluate a trained checkpoint on the test set.

Produces, in ``results/``:
  * ``<model>_metrics.json``            — accuracy + full classification report
  * ``<model>_confusion_matrix.png``    — normalized confusion matrix heatmap
  * ``<model>_predictions.png``         — grid of sample predictions (green=correct)

Usage::

    python -m src.evaluate --model cnn
    python -m src.evaluate --model resnet
"""
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report

from . import config, utils
from .data import get_dataloaders
from .models import build_model


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run the model over a loader; return (y_true, y_pred, images)."""
    model.eval()
    y_true, y_pred, images = [], [], []
    for x, y in loader:
        out = model(x.to(device))
        y_pred.extend(out.argmax(1).cpu().numpy().tolist())
        y_true.extend(y.numpy().tolist())
        images.append(x.cpu())
    return np.array(y_true), np.array(y_pred), torch.cat(images, dim=0)


def evaluate(model_type="cnn", ckpt_path=None, device=None):
    """Load a checkpoint, score the test set, and save metrics + figures."""
    device = device or config.DEVICE
    _, _, test_loader, classes = get_dataloaders(model_type)

    model = build_model(model_type).to(device)
    ckpt_path = ckpt_path or (config.RESULTS_DIR / f"{model_type}_best.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    y_true, y_pred, images = collect_predictions(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    print(f"{model_type} test accuracy: {acc:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=classes))

    with open(config.RESULTS_DIR / f"{model_type}_metrics.json", "w") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=2)

    utils.plot_confusion_matrix(
        y_true, y_pred, classes, f"{model_type.upper()} — Confusion Matrix",
        save_path=config.RESULTS_DIR / f"{model_type}_confusion_matrix.png",
    )

    mean, std = (
        (config.IMAGENET_MEAN, config.IMAGENET_STD)
        if model_type == "resnet"
        else (config.EUROSAT_MEAN, config.EUROSAT_STD)
    )
    utils.show_predictions(
        images, y_true, y_pred, classes, mean, std,
        f"{model_type.upper()} — Sample Predictions",
        save_path=config.RESULTS_DIR / f"{model_type}_predictions.png",
    )
    return acc, report


def main():
    parser = argparse.ArgumentParser(description="Evaluate an EuroSAT classifier.")
    parser.add_argument("--model", choices=["cnn", "resnet"], default="cnn")
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    utils.set_seed()
    evaluate(args.model, args.ckpt)


if __name__ == "__main__":
    main()
