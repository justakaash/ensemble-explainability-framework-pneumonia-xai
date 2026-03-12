# =============================================================
# train.py - Training Loop with ECE Calibration
# Project: Multi-Explainability Deep Learning Framework
#          for Reliable Early Pneumonia Detection
# =============================================================

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    set_seed,
    get_device,
    create_dirs,
    save_checkpoint,
    compute_metrics,
    compute_ece,
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_calibration_curve,
)
from preprocessing import get_dataloaders
from model import build_model, freeze_backbone, unfreeze_all


# -------------------------------------------------------------
# 1. CONFIGURATION
# -------------------------------------------------------------
CONFIG = {
    # Paths
    "data_root": "data/chest_xray",
    "output_dir": "outputs",
    "checkpoint_dir": "outputs/checkpoints",
    # Training
    "seed": 42,
    "num_epochs": 30,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "patience": 7,  # Early stopping patience
    # Model
    "num_classes": 2,
    "dropout_rate": 0.5,
    "pretrained": True,
    # Fine-tuning strategy
    "freeze_epochs": 5,  # Freeze backbone for first N epochs
}


# -------------------------------------------------------------
# 2. WEIGHTED LOSS (handles class imbalance)
# -------------------------------------------------------------
def get_loss_function(class_counts, device):
    """
    Compute class weights to handle pneumonia/normal imbalance.
    Pneumonia samples >> Normal samples in Kaggle dataset.
    """
    total = class_counts["total"]
    w_normal = total / (2 * class_counts["NORMAL"])
    w_pneumo = total / (2 * class_counts["PNEUMONIA"])
    weights = torch.tensor([w_normal, w_pneumo], dtype=torch.float).to(device)
    print(f"[INFO] Class weights → Normal: {w_normal:.4f} | Pneumonia: {w_pneumo:.4f}")
    return nn.CrossEntropyLoss(weight=weights)


# -------------------------------------------------------------
# 3. TRAINING STEP
# -------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{correct / total:.4f}"}
        )

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# -------------------------------------------------------------
# 4. VALIDATION STEP
# -------------------------------------------------------------
def validate(model, loader, criterion, device):
    """Run validation/test epoch."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Pneumonia probability
            preds = outputs.argmax(dim=1)

            running_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    return epoch_loss, epoch_acc, all_labels, all_preds, all_probs


# -------------------------------------------------------------
# 5. MAIN TRAINING LOOP
# -------------------------------------------------------------
def train(config=CONFIG):
    """Full training pipeline with early stopping."""

    # Setup
    set_seed(config["seed"])
    device = get_device()
    create_dirs(config["output_dir"], config["checkpoint_dir"])

    # Data
    print("\n[INFO] Loading dataset...")
    dataloaders, class_counts = get_dataloaders(
        config["data_root"], config["batch_size"]
    )

    # Model
    print("\n[INFO] Building model...")
    model, device = build_model(
        num_classes=config["num_classes"],
        dropout_rate=config["dropout_rate"],
        pretrained=config["pretrained"],
        device=device,
    )

    # Loss & Optimizer
    criterion = get_loss_function(class_counts, device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"], eta_min=1e-6)

    # Fine-tuning strategy
    print(f"\n[INFO] Freezing backbone for first {config['freeze_epochs']} epochs...")
    freeze_backbone(model)

    # Training tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    patience_counter = 0
    best_checkpoint = os.path.join(config["checkpoint_dir"], "best_model.pth")

    print("\n" + "=" * 60)
    print("   STARTING TRAINING")
    print("=" * 60)

    for epoch in range(1, config["num_epochs"] + 1):
        # Unfreeze backbone after freeze_epochs
        if epoch == config["freeze_epochs"] + 1:
            print(
                f"\n[INFO] Epoch {epoch}: Unfreezing all layers for full fine-tuning..."
            )
            unfreeze_all(model)
            # Reset optimizer with lower LR for fine-tuning
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config["learning_rate"] * 0.1,
                weight_decay=config["weight_decay"],
            )
            scheduler = CosineAnnealingLR(
                optimizer, T_max=config["num_epochs"] - epoch, eta_min=1e-7
            )

        # Train
        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_labels, val_preds, val_probs = validate(
            model, dataloaders["val"], criterion, device
        )

        scheduler.step()

        # Track history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # ECE
        ece = compute_ece(val_probs, val_labels)

        print(
            f"\nEpoch [{epoch:02d}/{config['num_epochs']}] "
            f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
            f"| ECE: {ece:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_acc, best_checkpoint)
            print(f"   ✓ New best model saved! Val Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            print(
                f"   No improvement. Patience: {patience_counter}/{config['patience']}"
            )

        # Early stopping
        if patience_counter >= config["patience"]:
            print(f"\n[INFO] Early stopping triggered at epoch {epoch}")
            break

    print("\n" + "=" * 60)
    print(f"   TRAINING COMPLETE | Best Val Acc: {best_val_acc:.4f}")
    print("=" * 60)

    # -------------------------------------------------------------
    # 6. FINAL EVALUATION ON TEST SET
    # -------------------------------------------------------------
    print("\n[INFO] Evaluating on test set...")

    # Load best model
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, test_acc, test_labels, test_preds, test_probs = validate(
        model, dataloaders["test"], criterion, device
    )

    # Compute all metrics
    metrics = compute_metrics(test_labels, test_preds, test_probs)
    ece = compute_ece(test_probs, test_labels)

    print("\n" + "=" * 60)
    print("   TEST SET RESULTS")
    print("=" * 60)
    print(f"   Accuracy:    {metrics['accuracy']:.4f}")
    print(f"   Precision:   {metrics['precision']:.4f}")
    print(f"   Recall:      {metrics['recall']:.4f}")
    print(f"   F1-Score:    {metrics['f1']:.4f}")
    print(f"   AUC-ROC:     {metrics['auc_roc']:.4f}")
    print(f"   Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"   Specificity: {metrics['specificity']:.4f}")
    print(f"   ECE:         {ece:.4f}")
    print("=" * 60)

    # Save plots
    plot_training_history(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        save_path=os.path.join(config["output_dir"], "training_history.png"),
    )
    plot_confusion_matrix(
        test_labels,
        test_preds,
        save_path=os.path.join(config["output_dir"], "confusion_matrix.png"),
    )
    plot_roc_curve(
        test_labels,
        test_probs,
        save_path=os.path.join(config["output_dir"], "roc_curve.png"),
    )
    plot_calibration_curve(
        test_probs,
        test_labels,
        save_path=os.path.join(config["output_dir"], "calibration_curve.png"),
    )

    return model, metrics, ece


# -------------------------------------------------------------
# 7. ENTRY POINT
# -------------------------------------------------------------
if __name__ == "__main__":
    model, metrics, ece = train()
