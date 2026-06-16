# 🛰️ EuroSAT Land-Use Classification — Custom CNN vs. ResNet50 (Transfer Learning)

> MSc in **Artificial Intelligence & Visual Computing (AIVC)** — University of West Attica
> Assignment (*Ergasia*) 2026 — Image Classification

## 1. Problem description

[EuroSAT](https://github.com/phelber/EuroSAT) is a benchmark dataset of **Sentinel-2**
satellite images for **land-use / land-cover (LULC) classification**. It contains
**27,000 RGB images** of **64×64** pixels, evenly spread across **10 classes**:

`AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`,
`PermanentCrop`, `Residential`, `River`, `SeaLake`.

The goal of this project is to **train, analyse and compare at least two different deep
learning models** for this classification task, evaluating their performance,
capabilities and limitations.

Dataset source (Kaggle): [`apollo2506/eurosat-dataset`](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)

## 2. The Idea (methodology)

To contrast two of the approaches suggested in the assignment, we implement and compare:

| # | Model | Approach | Why |
|---|-------|----------|-----|
| 1 | **Custom CNN** | Trained **from scratch** | A compact, purpose-built VGG-style network (4 conv blocks) — a strong, lightweight baseline designed specifically for 64×64 tiles. |
| 2 | **ResNet50** | **Transfer learning + fine-tuning** | A deep backbone pre-trained on ImageNet, with a new classifier head and full fine-tuning — leverages learned visual features for higher accuracy and faster convergence. |

**Pipeline (identical for both models for a fair comparison):**

1. **Data** — download via `kagglehub`, build a **reproducible stratified 70/15/15**
   train/validation/test split (seed = 42).
2. **Augmentation** — random horizontal/vertical flips and ±15° rotation (satellite
   tiles are orientation-agnostic). The CNN uses native 64×64 with EuroSAT
   normalization; ResNet50 resizes to 224×224 with ImageNet normalization.
3. **Training** — `CrossEntropyLoss`, Adam, `ReduceLROnPlateau` scheduler, best
   checkpoint selected by validation accuracy.
4. **Hyperparameter optimization** — **Optuna (5 trials)** tunes learning rate,
   dropout, batch size and optimizer; the best configuration is used to retrain the
   final model.
5. **Evaluation** — accuracy + per-class precision/recall/F1, **confusion matrix**,
   and a **sample-prediction grid** on the held-out test set.

## 3. Repository structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── config.py      # paths, class names, normalization, hyperparameters
│   ├── data.py        # download, transforms, stratified split, dataloaders
│   ├── models.py      # CustomCNN + ResNet50 (transfer learning)
│   ├── train.py       # training loop + history tracking + checkpointing
│   ├── tune.py        # Optuna hyperparameter search (5 trials)
│   ├── evaluate.py    # metrics, confusion matrix, prediction grids
│   └── utils.py       # reproducibility + plotting helpers
├── notebooks/
│   └── EuroSAT_Classification.ipynb   # end-to-end, Colab-ready
└── results/           # generated figures + metrics (committed)
```

## 4. Installation & how to run

### Option A — Google Colab (recommended)

1. Open `notebooks/EuroSAT_Classification.ipynb` in
   [Google Colab](https://colab.research.google.com/) and select a **GPU** runtime
   (*Runtime → Change runtime type → GPU*).
2. Provide your Kaggle credentials when prompted (upload `kaggle.json`, available from
   your Kaggle account → *Settings → API → Create New Token*).
3. **Runtime → Run all.** The notebook downloads the data, trains both models, runs the
   Optuna search, and renders all curves, confusion matrices and prediction grids.

### Option B — Local

```bash
# 1. (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. configure Kaggle credentials (so kagglehub can download the dataset)
#    place kaggle.json in ~/.kaggle/  (or %USERPROFILE%\.kaggle\ on Windows)

# 4. train each model
python -m src.train --model cnn
python -m src.train --model resnet

# 5. tune hyperparameters with Optuna (5 trials)
python -m src.tune --model cnn

# 6. evaluate on the test set (writes metrics + figures to results/)
python -m src.evaluate --model cnn
python -m src.evaluate --model resnet
```

> **Note on compute:** training the full models and the Optuna search is GPU-intensive.
> A GPU (Colab or local CUDA) is strongly recommended. For a quick pipeline check on
> CPU, run with `--epochs 1`.

## 5. Results

> The figures below are generated into `results/` when you run the notebook/scripts.
> Replace the placeholder numbers in the table with your actual results after running.

### Training curves

| Custom CNN | ResNet50 |
|---|---|
| ![CNN curves](results/cnn_curves.png) | ![ResNet curves](results/resnet_curves.png) |

### Confusion matrices

| Custom CNN | ResNet50 |
|---|---|
| ![CNN confusion matrix](results/cnn_confusion_matrix.png) | ![ResNet confusion matrix](results/resnet_confusion_matrix.png) |

### Sample predictions (green = correct, red = wrong)

| Custom CNN | ResNet50 |
|---|---|
| ![CNN predictions](results/cnn_predictions.png) | ![ResNet predictions](results/resnet_predictions.png) |

### Optuna hyperparameter search (5 trials)

![Optuna history](results/optuna_cnn_history.png)

### Summary table

| Model | Approach | Test accuracy | Macro F1 | Trainable params | Epoch time* |
|-------|----------|---------------|----------|------------------|-------------|
| Custom CNN | from scratch | ~0.90–0.94 | ~0.90 | ~1.2M | fast |
| ResNet50 | transfer learning | ~0.96–0.98 | ~0.97 | ~23.5M | slower |

\* indicative — fill in with your measured values.

## 6. Comparison, capabilities & limitations

**Custom CNN (from scratch)**
- ✅ Lightweight (~1.2M params), fast to train, low memory — works on modest hardware.
- ✅ Operates on native 64×64 tiles (no upscaling needed).
- ⚠️ Lower ceiling on accuracy; must learn all visual features from limited data.
- ⚠️ More sensitive to augmentation/regularization choices.

**ResNet50 (transfer learning + fine-tuning)**
- ✅ Highest accuracy; ImageNet features transfer well to satellite imagery.
- ✅ Converges in fewer epochs.
- ⚠️ Many more parameters; heavier compute/memory and slower inference.
- ⚠️ Requires upscaling 64×64 → 224×224 and ImageNet normalization.

**Typical confusions (both models):** semantically/visually similar classes such as
`Highway` ↔ `River`, `PermanentCrop` ↔ `AnnualCrop`, and `HerbaceousVegetation` ↔
`Pasture` — visible in the confusion matrices.

## 7. Reproducibility

All randomness is seeded (`SEED = 42` in `src/config.py`): the stratified split and
training are deterministic, so reruns produce the same partition and comparable metrics.

## 8. License & acknowledgements

- EuroSAT: Helber et al., *"EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land
  Use and Land Cover Classification"*, IEEE JSTARS, 2019.
- Built with [PyTorch](https://pytorch.org/), [torchvision](https://pytorch.org/vision/),
  [scikit-learn](https://scikit-learn.org/) and [Optuna](https://optuna.org/).
