# =============================================================
# model.py - DenseNet-121 Classification Model
# Project: Multi-Explainability Deep Learning Framework
#          for Reliable Early Pneumonia Detection
# =============================================================

import torch
import torch.nn as nn
from torchvision import models


# -------------------------------------------------------------
# 1. DENSENET-121 MODEL
# -------------------------------------------------------------
class PneumoniaDetector(nn.Module):
    """
    DenseNet-121 based pneumonia detection model.
    Pretrained on ImageNet, fine-tuned for binary classification.

    Architecture changes:
    - Original classifier replaced with custom head
    - Dropout added for regularization
    - Output: 2 classes (Normal / Pneumonia)
    """

    def __init__(self, num_classes=2, dropout_rate=0.5, pretrained=True):
        super(PneumoniaDetector, self).__init__()

        # Load pretrained DenseNet-121
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.densenet = models.densenet121(weights=weights)

        # Get number of features from original classifier
        in_features = self.densenet.classifier.in_features  # 1024

        # Replace classifier with custom head
        self.densenet.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(256, num_classes),
        )

        # Store config
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

    def forward(self, x):
        return self.densenet(x)

    def get_features(self, x):
        """Extract features before classifier (for XAI)."""
        features = self.densenet.features(x)
        features = torch.nn.functional.relu(features, inplace=True)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        return features

    def get_last_conv_layer(self):
        """Return last convolutional layer (for Grad-CAM++)."""
        return self.densenet.features.denseblock4.denselayer16.conv2


# -------------------------------------------------------------
# 2. MODEL FACTORY
# -------------------------------------------------------------
def build_model(num_classes=2, dropout_rate=0.5, pretrained=True, device=None):
    """
    Build and return the PneumoniaDetector model.

    Args:
        num_classes:   Number of output classes (default: 2)
        dropout_rate:  Dropout probability (default: 0.5)
        pretrained:    Use ImageNet pretrained weights (default: True)
        device:        Target device (cuda/mps/cpu)

    Returns:
        model: PneumoniaDetector on target device
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model = PneumoniaDetector(
        num_classes=num_classes, dropout_rate=dropout_rate, pretrained=pretrained
    ).to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model: DenseNet-121 based PneumoniaDetector")
    print(f"[INFO] Total parameters:     {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")
    print(f"[INFO] Device: {device}")

    return model, device


# -------------------------------------------------------------
# 3. FREEZE / UNFREEZE LAYERS (for fine-tuning strategy)
# -------------------------------------------------------------
def freeze_backbone(model):
    """Freeze DenseNet backbone — only train classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Backbone frozen — Trainable params: {trainable:,}")


def unfreeze_all(model):
    """Unfreeze all layers for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] All layers unfrozen — Trainable params: {trainable:,}")


def unfreeze_last_n_blocks(model, n=2):
    """
    Unfreeze last n dense blocks for gradual fine-tuning.
    Useful for paper's ablation study.
    """
    # First freeze all
    freeze_backbone(model)

    # Then unfreeze last n denseblocks
    blocks_to_unfreeze = [f"denseblock{4 - i}" for i in range(n)]
    for name, param in model.named_parameters():
        for block in blocks_to_unfreeze:
            if block in name:
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Last {n} blocks unfrozen — Trainable params: {trainable:,}")
