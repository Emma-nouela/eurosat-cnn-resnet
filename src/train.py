"""Generic training loop with history tracking and best-checkpoint saving.

Run from the command line, e.g.::

    python -m src.train --model cnn
    python -m src.train --model resnet --epochs 12
"""
import argparse
import copy
import time

import torch
import torch.nn as nn

from . import config, utils
from .data import get_dataloaders
from .models import build_model


def make_optimizer(model, name, lr):
    """Build an optimizer over the model's trainable parameters."""
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name!r}")


def run_epoch(model, loader, criterion, optimizer, device, train):
    """Run a single epoch; returns (avg_loss, accuracy)."""
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)

    return loss_sum / total, correct / total


def train_model(model, train_loader, val_loader, epochs, lr, optimizer_name="adam",
                device=None, ckpt_path=None, verbose=True):
    """Train ``model`` and return (best_model, history, best_val_acc).

    The model with the highest validation accuracy is restored before returning,
    and optionally written to ``ckpt_path``.
    """
    device = device or config.DEVICE
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, optimizer_name, lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step(va_acc)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if va_acc > best_acc:
            best_acc = va_acc
            best_state = copy.deepcopy(model.state_dict())
            if ckpt_path:
                torch.save(best_state, ckpt_path)

        if verbose:
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
                f"{time.time() - t0:.1f}s"
            )

    model.load_state_dict(best_state)
    return model, history, best_acc


def main():
    parser = argparse.ArgumentParser(description="Train an EuroSAT classifier.")
    parser.add_argument("--model", choices=["cnn", "resnet"], default="cnn")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    utils.set_seed()
    cfg = config.DEFAULTS[args.model]
    epochs = args.epochs or cfg["epochs"]
    lr = args.lr or cfg["lr"]

    print(f"Device: {config.DEVICE} | model: {args.model} | epochs: {epochs} | lr: {lr}")
    train_loader, val_loader, _, _ = get_dataloaders(args.model, batch_size=args.batch_size)
    model = build_model(args.model)

    ckpt = config.RESULTS_DIR / f"{args.model}_best.pth"
    _, history, best_acc = train_model(
        model, train_loader, val_loader, epochs, lr, cfg["optimizer"], ckpt_path=ckpt
    )
    utils.plot_curves(
        history, args.model.upper(),
        save_path=config.RESULTS_DIR / f"{args.model}_curves.png",
    )
    print(f"Best val accuracy: {best_acc:.4f} | checkpoint saved to {ckpt}")


if __name__ == "__main__":
    main()
