"""EuroSAT land-use classification package.

Modules
-------
config    : paths, class names, hyperparameters
data      : dataset download, transforms, stratified splits, dataloaders
models    : CustomCNN (from scratch) + ResNet50 (transfer learning)
train     : generic training loop with history tracking
tune      : Optuna hyperparameter search (5 trials)
evaluate  : test-set metrics, confusion matrix, prediction grids
utils     : reproducibility + plotting helpers
"""
