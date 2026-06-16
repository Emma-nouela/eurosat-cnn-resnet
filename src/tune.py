"""Optuna hyperparameter search (5 trials by default).

Optimizes validation accuracy over learning rate, dropout, batch size and
optimizer. The best configuration is written to ``results/`` and can then be used
to retrain the final model.

Usage::

    python -m src.tune --model cnn               # 5 trials, short epochs
    python -m src.tune --model resnet --trials 5
"""
import argparse
import json

import optuna

from . import config, utils
from .data import get_dataloaders
from .models import build_model
from .train import train_model


def objective(trial, model_type, tune_epochs):
    """One Optuna trial: sample hyperparameters, train briefly, return val accuracy."""
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])

    train_loader, val_loader, _, _ = get_dataloaders(model_type, batch_size=batch_size)

    if model_type == "cnn":
        model = build_model("cnn", dropout=dropout)
    else:
        model = build_model("resnet")  # dropout not applicable to the ResNet head

    _, _, best_acc = train_model(
        model, train_loader, val_loader, epochs=tune_epochs,
        lr=lr, optimizer_name=optimizer_name, verbose=False,
    )
    return best_acc


def run_study(model_type="cnn", n_trials=5, tune_epochs=5):
    """Run the Optuna study and persist best params, the trials table and a plot."""
    utils.set_seed()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.SEED),
    )
    study.optimize(lambda t: objective(t, model_type, tune_epochs), n_trials=n_trials)

    best = dict(study.best_params)
    best["val_acc"] = study.best_value

    with open(config.RESULTS_DIR / f"optuna_{model_type}_best_params.json", "w") as f:
        json.dump(best, f, indent=2)
    study.trials_dataframe().to_csv(
        config.RESULTS_DIR / f"optuna_{model_type}_trials.csv", index=False
    )

    # Optimization-history plot (best-effort: needs matplotlib backend for Optuna viz).
    try:
        from optuna.visualization.matplotlib import plot_optimization_history

        ax = plot_optimization_history(study)
        ax.figure.savefig(
            config.RESULTS_DIR / f"optuna_{model_type}_history.png",
            dpi=120, bbox_inches="tight",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Could not render the Optuna history plot: {exc}")

    print(f"Best params for {model_type}: {best}")
    return study


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for EuroSAT models.")
    parser.add_argument("--model", choices=["cnn", "resnet"], default="cnn")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--tune-epochs", type=int, default=5,
                        help="Epochs per trial (kept small to keep the search fast).")
    args = parser.parse_args()
    run_study(args.model, args.trials, args.tune_epochs)


if __name__ == "__main__":
    main()
