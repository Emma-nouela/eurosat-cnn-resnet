"""Model definitions.

Two contrasting approaches from the assignment brief:

* ``CustomCNN``      — a compact convolutional network trained *from scratch*.
* ``build_resnet50`` — a ResNet50 pre-trained on ImageNet, adapted via *transfer
  learning* (replace the classifier head) and fine-tuned on EuroSAT.
"""
import torch.nn as nn
from torchvision import models

from . import config


class CustomCNN(nn.Module):
    """A small VGG-style CNN for 64x64 RGB EuroSAT tiles.

    Four Conv-BN-ReLU-MaxPool blocks progressively halve the spatial resolution
    (64 -> 32 -> 16 -> 8 -> 4) while doubling the channel count, followed by global
    average pooling and a two-layer classifier with dropout.
    """

    def __init__(self, num_classes=config.NUM_CLASSES, dropout=0.4, width=32):
        super().__init__()
        c1, c2, c3, c4 = width, width * 2, width * 4, width * 8

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, c1),    # 64 -> 32
            block(c1, c2),   # 32 -> 16
            block(c2, c3),   # 16 -> 8
            block(c3, c4),   # 8  -> 4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_resnet50(num_classes=config.NUM_CLASSES, pretrained=True, freeze_backbone=False):
    """ResNet50 with ImageNet weights and a fresh classifier head for EuroSAT.

    Parameters
    ----------
    pretrained      : load ImageNet-pretrained weights (the transfer-learning case).
    freeze_backbone : if True, train only the new head (pure feature extraction);
                      if False (default), fine-tune the whole network.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)  # always trainable
    return model


def build_model(model_type, **kwargs):
    """Factory used by the training / tuning / evaluation entrypoints."""
    if model_type == "cnn":
        return CustomCNN(**kwargs)
    if model_type == "resnet":
        return build_resnet50(**kwargs)
    raise ValueError(f"Unknown model_type: {model_type!r}")


def count_parameters(model):
    """Total number of trainable parameters — handy for the comparison table."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
