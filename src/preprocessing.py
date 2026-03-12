# =============================================================
# preprocessing.py - Data Loading & Augmentation
# Project: Multi-Explainability Deep Learning Framework
#          for Reliable Early Pneumonia Detection
# =============================================================

import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


# -------------------------------------------------------------
# 1. CONSTANTS
# -------------------------------------------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # Set to 0 for Windows compatibility
MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
STD = [0.229, 0.224, 0.225]  # ImageNet std

CLASSES = ["NORMAL", "PNEUMONIA"]
CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}


# -------------------------------------------------------------
# 2. ALBUMENTATIONS AUGMENTATION PIPELINES
# -------------------------------------------------------------
def get_train_transforms():
    """Strong augmentations for training set."""
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=(-10, 10), p=0.5),
            A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.3),  # Enhances early pneumonia patterns
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms():
    """Minimal transforms for validation/test set."""
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )


def get_inference_transforms():
    """Transforms for single image inference in Streamlit app."""
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


# -------------------------------------------------------------
# 3. CUSTOM DATASET CLASS
# -------------------------------------------------------------
class ChestXRayDataset(Dataset):
    """
    Custom Dataset for Kaggle Chest X-Ray (Pneumonia) Dataset.

    Expected folder structure:
        chest_xray/
            train/
                NORMAL/
                PNEUMONIA/
            val/
                NORMAL/
                PNEUMONIA/
            test/
                NORMAL/
                PNEUMONIA/
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = self.labels[idx]
        return image, label


# -------------------------------------------------------------
# 4. DATA LOADING HELPERS
# -------------------------------------------------------------
def load_image_paths_and_labels(split_dir):
    """
    Load all image paths and labels from a split directory.

    Args:
        split_dir: Path to train/val/test folder

    Returns:
        image_paths (list), labels (list)
    """
    image_paths, labels = [], []
    for class_name, class_idx in CLASS_TO_IDX.items():
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"[WARNING] Directory not found: {class_dir}")
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(class_idx)
    print(f"[INFO] Loaded {len(image_paths)} images from {split_dir}")
    return image_paths, labels


def get_dataloaders(data_root, batch_size=BATCH_SIZE):
    """
    Build train, val, and test DataLoaders.

    Args:
        data_root: Path to chest_xray/ folder
        batch_size: Batch size for DataLoader

    Returns:
        dict with 'train', 'val', 'test' DataLoaders
        dict with class counts for imbalance handling
    """
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    # Load paths and labels
    train_paths, train_labels = load_image_paths_and_labels(train_dir)
    val_paths, val_labels = load_image_paths_and_labels(val_dir)
    test_paths, test_labels = load_image_paths_and_labels(test_dir)

    # Handle small val set (Kaggle val only has 16 images)
    # Merge val into train and re-split if too small
    if len(val_paths) < 100:
        print("[INFO] Val set too small — merging with train and re-splitting (80/20)")
        all_paths = train_paths + val_paths
        all_labels = train_labels + val_labels
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )

    # Build datasets
    train_dataset = ChestXRayDataset(train_paths, train_labels, get_train_transforms())
    val_dataset = ChestXRayDataset(val_paths, val_labels, get_val_transforms())
    test_dataset = ChestXRayDataset(test_paths, test_labels, get_val_transforms())

    # Build dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Class counts for weighted loss
    class_counts = {
        "NORMAL": train_labels.count(0),
        "PNEUMONIA": train_labels.count(1),
        "total": len(train_labels),
    }
    print(
        f"[INFO] Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}"
    )
    print(f"[INFO] Class counts → {class_counts}")

    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    return dataloaders, class_counts


# -------------------------------------------------------------
# 5. SINGLE IMAGE LOADER (for Streamlit app)
# -------------------------------------------------------------
def load_single_image(image_path):
    """
    Load and preprocess a single image for inference.

    Args:
        image_path: Path to the image file

    Returns:
        tensor: Preprocessed image tensor (1, 3, 224, 224)
        pil_image: Original PIL image for display
    """
    pil_image = Image.open(image_path).convert("RGB")
    transform = get_inference_transforms()
    tensor = transform(pil_image).unsqueeze(0)  # Add batch dimension
    return tensor, pil_image


def load_image_from_upload(uploaded_file):
    """
    Load image from Streamlit uploaded file object.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        tensor: Preprocessed image tensor (1, 3, 224, 224)
        pil_image: Original PIL image for display
    """
    pil_image = Image.open(uploaded_file).convert("RGB")
    transform = get_inference_transforms()
    tensor = transform(pil_image).unsqueeze(0)
    return tensor, pil_image
