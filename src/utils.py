# =============================================================
# utils.py - Helper Functions, Metrics & Visualization
# Project: Multi-Explainability Deep Learning Framework
#          for Reliable Early Pneumonia Detection
# =============================================================

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)


# -------------------------------------------------------------
# 1. REPRODUCIBILITY
# -------------------------------------------------------------
def set_seed(seed=42):
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed set to {seed}")


# -------------------------------------------------------------
# 2. DEVICE CONFIGURATION
# -------------------------------------------------------------
def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] Using Apple MPS (Metal)")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU")
    return device


# -------------------------------------------------------------
# 3. DIRECTORY SETUP
# -------------------------------------------------------------
def create_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"[INFO] Directory ready: {d}")


# -------------------------------------------------------------
# 4. MODEL CHECKPOINT
# -------------------------------------------------------------
def save_checkpoint(model, optimizer, epoch, val_acc, path):
    """Save model checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
        },
        path,
    )
    print(f"[INFO] Checkpoint saved → {path}")


def load_checkpoint(model, optimizer, path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    val_acc = checkpoint["val_acc"]
    print(f"[INFO] Checkpoint loaded from epoch {epoch} | Val Acc: {val_acc:.4f}")
    return model, optimizer, epoch, val_acc


# -------------------------------------------------------------
# 5. METRICS
# -------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_prob):
    """Compute classification metrics."""
    report = classification_report(
        y_true, y_pred, target_names=["Normal", "Pneumonia"], output_dict=True
    )
    auc = roc_auc_score(y_true, y_prob)
    metrics = {
        "accuracy": report["accuracy"],
        "precision": report["Pneumonia"]["precision"],
        "recall": report["Pneumonia"]["recall"],
        "f1": report["Pneumonia"]["f1-score"],
        "auc_roc": auc,
        "sensitivity": report["Pneumonia"]["recall"],
        "specificity": report["Normal"]["recall"],
    }
    return metrics


def compute_ece(y_prob, y_true, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    Measures alignment between confidence and accuracy.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    ece /= len(y_true)
    return ece


# -------------------------------------------------------------
# 6. VISUALIZATION
# -------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot and optionally save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Pneumonia"],
        yticklabels=["Normal", "Pneumonia"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Confusion matrix saved → {save_path}")
    plt.show()


def plot_roc_curve(y_true, y_prob, save_path=None):
    """Plot ROC curve with AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] ROC curve saved → {save_path}")
    plt.show()


def plot_training_history(
    train_losses, val_losses, train_accs, val_accs, save_path=None
):
    """Plot training and validation loss/accuracy curves."""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, train_losses, "b-o", label="Train Loss")
    ax1.plot(epochs, val_losses, "r-o", label="Val Loss")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(epochs, train_accs, "b-o", label="Train Acc")
    ax2.plot(epochs, val_accs, "r-o", label="Val Acc")
    ax2.set_title("Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Training history saved → {save_path}")
    plt.show()


def plot_calibration_curve(y_prob, y_true, n_bins=10, save_path=None):
    """Plot reliability/calibration diagram."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs = [], []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_accs.append(y_true[mask].mean())
        bin_confs.append(y_prob[mask].mean())

    plt.figure(figsize=(6, 5))
    plt.plot(bin_confs, bin_accs, "b-o", label="Model Calibration")
    plt.plot([0, 1], [0, 1], "r--", label="Perfect Calibration")
    plt.xlabel("Mean Confidence")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve (Reliability Diagram)")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Calibration curve saved → {save_path}")
    plt.show()
