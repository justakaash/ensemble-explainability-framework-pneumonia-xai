# =============================================================
# explain.py - Multi-Explainability XAI Framework
# Project: Multi-Explainability Deep Learning Framework
#          for Reliable Early Pneumonia Detection
#
# XAI Methods:
#   1. Grad-CAM++ (gradient-weighted class activation maps)
#   2. Integrated Gradients (attribution-based)
#   3. SHAP (game-theory-based feature importance)
#
# NOVEL CONTRIBUTION:
#   - XAI Consistency Score (cross-method agreement metric)
#   - Multi-map ensemble overlay
# =============================================================

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from PIL import Image

# Captum XAI
from captum.attr import IntegratedGradients, GradientShap, NoiseTunnel, LayerGradCam
import shap

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# -------------------------------------------------------------
# 1. GRAD-CAM++ IMPLEMENTATION
# -------------------------------------------------------------
class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation for DenseNet-121.
    Produces class-discriminative localization maps.
    Improvement over Grad-CAM: better localization of
    multiple object instances and partial occluded objects.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM++ heatmap.

        Args:
            input_tensor: Preprocessed image tensor (1, 3, H, W)
            target_class: Class index (None = predicted class)

        Returns:
            cam: Normalized heatmap (H, W) numpy array
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Grad-CAM++ weight computation
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Grad-CAM++ specific: second and third order gradients
        alpha_num = gradients**2
        alpha_denom = 2 * (gradients**2) + (activations * gradients**3).sum(
            dim=(1, 2), keepdim=True
        )
        alpha_denom = torch.where(
            alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom)
        )
        alpha = alpha_num / alpha_denom
        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2), keepdim=True)

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, target_class


# -------------------------------------------------------------
# 2. INTEGRATED GRADIENTS
# -------------------------------------------------------------
def generate_integrated_gradients(model, input_tensor, target_class=None, n_steps=50):
    """
    Generate Integrated Gradients attribution map using Captum.
    Assigns importance scores by integrating gradients along
    a straight path from baseline (black image) to input.

    Args:
        model:        Trained PneumoniaDetector
        input_tensor: Preprocessed image tensor (1, 3, H, W)
        target_class: Target class index
        n_steps:      Integration steps (higher = more accurate)

    Returns:
        attr_map: Normalized attribution map (H, W) numpy array
    """
    model.eval()

    if target_class is None:
        with torch.no_grad():
            output = model(input_tensor)
            target_class = output.argmax(dim=1).item()

    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(input_tensor)  # Black image baseline

    attributions = ig.attribute(
        input_tensor,
        baseline,
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=False,
    )

    # Convert to single-channel heatmap
    attr_map = attributions[0].cpu().detach().numpy()
    attr_map = np.mean(np.abs(attr_map), axis=0)  # Average over RGB channels
    attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)

    return attr_map, target_class


# -------------------------------------------------------------
# 3. SHAP (GradientSHAP via Captum)
# -------------------------------------------------------------
def generate_shap(model, input_tensor, target_class=None, n_samples=10):
    """
    Generate SHAP attribution map using GradientSHAP (Captum).
    Combines SHAP values with Integrated Gradients for efficiency.

    Args:
        model:        Trained PneumoniaDetector
        input_tensor: Preprocessed image tensor (1, 3, H, W)
        target_class: Target class index
        n_samples:    Number of noise samples

    Returns:
        shap_map: Normalized SHAP map (H, W) numpy array
    """
    model.eval()

    if target_class is None:
        with torch.no_grad():
            output = model(input_tensor)
            target_class = output.argmax(dim=1).item()

    gradient_shap = GradientShap(model)

    # Baselines: random Gaussian noise
    baselines = torch.randn(n_samples, *input_tensor.shape[1:]).unsqueeze(0)
    baselines = baselines.squeeze(0).to(input_tensor.device)

    attributions = gradient_shap.attribute(
        input_tensor, baselines, target=target_class, n_samples=n_samples, stdevs=0.09
    )

    shap_map = attributions[0].cpu().detach().numpy()
    shap_map = np.mean(np.abs(shap_map), axis=0)
    shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min() + 1e-8)

    return shap_map, target_class


# -------------------------------------------------------------
# 4. XAI CONSISTENCY SCORE ← NOVEL CONTRIBUTION
# -------------------------------------------------------------
def compute_consistency_score(map1, map2, map3, threshold=0.5):
    """
    Compute pairwise and overall XAI consistency score.

    NOVEL METRIC: Measures spatial agreement between explanation
    maps from different XAI methods. High consistency = more
    trustworthy and reliable explanations.

    Method:
        1. Binarize each map at threshold
        2. Compute Intersection over Union (IoU) for each pair
        3. Average pairwise IoUs = Overall Consistency Score

    Args:
        map1: Grad-CAM++ heatmap (H, W) normalized [0,1]
        map2: Integrated Gradients map (H, W) normalized [0,1]
        map3: SHAP map (H, W) normalized [0,1]
        threshold: Binarization threshold (default: 0.5)

    Returns:
        scores: dict with pairwise and overall consistency scores
    """

    def iou(a, b, thresh):
        bin_a = (a >= thresh).astype(np.float32)
        bin_b = (b >= thresh).astype(np.float32)
        intersection = (bin_a * bin_b).sum()
        union = np.clip(bin_a + bin_b, 0, 1).sum()
        return float(intersection / (union + 1e-8))

    def pearson_corr(a, b):
        a_flat = a.flatten()
        b_flat = b.flatten()
        corr = np.corrcoef(a_flat, b_flat)[0, 1]
        return float(corr)

    # Resize all maps to same size if needed
    h, w = map1.shape
    map2_r = cv2.resize(map2, (w, h))
    map3_r = cv2.resize(map3, (w, h))

    # Pairwise IoU scores
    iou_12 = iou(map1, map2_r, threshold)
    iou_13 = iou(map1, map3_r, threshold)
    iou_23 = iou(map2_r, map3_r, threshold)

    # Pairwise Pearson correlation
    corr_12 = pearson_corr(map1, map2_r)
    corr_13 = pearson_corr(map1, map3_r)
    corr_23 = pearson_corr(map2_r, map3_r)

    # Overall scores
    overall_iou = (iou_12 + iou_13 + iou_23) / 3
    overall_corr = (corr_12 + corr_13 + corr_23) / 3

    scores = {
        "iou_gradcam_ig": iou_12,
        "iou_gradcam_shap": iou_13,
        "iou_ig_shap": iou_23,
        "corr_gradcam_ig": corr_12,
        "corr_gradcam_shap": corr_13,
        "corr_ig_shap": corr_23,
        "overall_iou": overall_iou,
        "overall_correlation": overall_corr,
        "consistency_score": (overall_iou + max(overall_corr, 0)) / 2,
    }

    return scores


# -------------------------------------------------------------
# 5. HEATMAP OVERLAY UTILITIES
# -------------------------------------------------------------
def overlay_heatmap(pil_image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay a heatmap on the original image.

    Args:
        pil_image: Original PIL image
        heatmap:   Normalized heatmap array (H, W) [0, 1]
        alpha:     Overlay transparency
        colormap:  OpenCV colormap

    Returns:
        overlaid: PIL image with heatmap overlay
    """
    # Resize image to standard size
    img = np.array(pil_image.resize((224, 224)))

    # Convert heatmap to color
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(cv2.resize(heatmap_uint8, (224, 224)), colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend
    overlaid = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return Image.fromarray(overlaid)


def resize_heatmap(heatmap, size=(224, 224)):
    """Resize heatmap to target size."""
    return cv2.resize(heatmap, size)


# -------------------------------------------------------------
# 6. MULTI-XAI ENSEMBLE MAP ← NOVEL CONTRIBUTION
# -------------------------------------------------------------
def generate_ensemble_map(gradcam_map, ig_map, shap_map, weights=(0.4, 0.3, 0.3)):
    """
    Generate weighted ensemble explanation map from all 3 XAI methods.

    NOVEL: Combines multiple explanation maps into one
    consensus visualization, weighted by typical reliability
    of each method for medical imaging.

    Args:
        gradcam_map: Grad-CAM++ heatmap (H, W)
        ig_map:      Integrated Gradients map (H, W)
        shap_map:    SHAP map (H, W)
        weights:     Tuple of (w_gradcam, w_ig, w_shap)

    Returns:
        ensemble: Normalized ensemble map (H, W)
    """
    h, w = gradcam_map.shape

    # Resize all to same size
    ig_r = cv2.resize(ig_map, (w, h))
    shap_r = cv2.resize(shap_map, (w, h))

    # Weighted combination
    ensemble = weights[0] * gradcam_map + weights[1] * ig_r + weights[2] * shap_r

    # Normalize
    ensemble = (ensemble - ensemble.min()) / (ensemble.max() - ensemble.min() + 1e-8)
    return ensemble


# -------------------------------------------------------------
# 7. FULL EXPLANATION PIPELINE
# -------------------------------------------------------------
def explain_prediction(model, input_tensor, pil_image, device, save_dir=None):
    """
    Run full multi-XAI explanation pipeline on one image.

    Steps:
        1. Generate Grad-CAM++ map
        2. Generate Integrated Gradients map
        3. Generate SHAP map
        4. Compute XAI Consistency Score
        5. Generate Ensemble map
        6. Visualize all maps

    Args:
        model:        Trained PneumoniaDetector
        input_tensor: Preprocessed image tensor (1, 3, H, W)
        pil_image:    Original PIL image for display
        device:       Torch device
        save_dir:     Directory to save plots (optional)

    Returns:
        results: dict with all maps, scores, prediction
    """
    model.eval()
    input_tensor = input_tensor.to(device)

    # --- Prediction ---
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item()

    class_names = ["Normal", "Pneumonia"]
    print(
        f"\n[XAI] Prediction: {class_names[pred_class]} | Confidence: {confidence:.4f}"
    )

    # --- 1. Grad-CAM++ ---
    print("[XAI] Generating Grad-CAM++...")
    target_layer = model.get_last_conv_layer()
    gradcam = GradCAMPlusPlus(model, target_layer)
    gradcam_map, _ = gradcam.generate(input_tensor, pred_class)
    gradcam_map = resize_heatmap(gradcam_map, (224, 224))

    # --- 2. Integrated Gradients ---
    print("[XAI] Generating Integrated Gradients...")
    ig_map, _ = generate_integrated_gradients(
        model, input_tensor, pred_class, n_steps=50
    )
    ig_map = resize_heatmap(ig_map, (224, 224))

    # --- 3. SHAP ---
    print("[XAI] Generating SHAP...")
    shap_map, _ = generate_shap(model, input_tensor, pred_class, n_samples=10)
    shap_map = resize_heatmap(shap_map, (224, 224))

    # --- 4. Consistency Score ---
    print("[XAI] Computing consistency scores...")
    consistency = compute_consistency_score(gradcam_map, ig_map, shap_map)
    print(f"[XAI] Consistency Score: {consistency['consistency_score']:.4f}")
    print(f"      IoU  (GradCAM++ vs IG):   {consistency['iou_gradcam_ig']:.4f}")
    print(f"      IoU  (GradCAM++ vs SHAP): {consistency['iou_gradcam_shap']:.4f}")
    print(f"      IoU  (IG vs SHAP):        {consistency['iou_ig_shap']:.4f}")

    # --- 5. Ensemble Map ---
    print("[XAI] Generating Ensemble Map...")
    ensemble_map = generate_ensemble_map(gradcam_map, ig_map, shap_map)

    # --- 6. Overlays ---
    gradcam_overlay = overlay_heatmap(pil_image, gradcam_map)
    ig_overlay = overlay_heatmap(pil_image, ig_map, colormap=cv2.COLORMAP_PLASMA)
    shap_overlay = overlay_heatmap(pil_image, shap_map, colormap=cv2.COLORMAP_VIRIDIS)
    ensemble_overlay = overlay_heatmap(
        pil_image, ensemble_map, colormap=cv2.COLORMAP_HOT
    )

    # --- 7. Visualization ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        f"Multi-XAI Explanation | Prediction: {class_names[pred_class]} "
        f"({confidence * 100:.1f}%) | Consistency: {consistency['consistency_score']:.4f}",
        fontsize=14,
        fontweight="bold",
    )

    titles = ["Original X-Ray", "Grad-CAM++", "Integrated Gradients", "SHAP"]
    images1 = [pil_image.resize((224, 224)), gradcam_overlay, ig_overlay, shap_overlay]

    for ax, title, img in zip(axes[0], titles, images1):
        ax.imshow(img, cmap="gray" if title == "Original X-Ray" else None)
        ax.set_title(title, fontweight="bold")
        ax.axis("off")

    # Row 2: Raw heatmaps + ensemble
    raw_maps = [gradcam_map, ig_map, shap_map, ensemble_map]
    raw_titles = ["Grad-CAM++ Map", "IG Map", "SHAP Map", "Ensemble Map"]
    cmaps = ["jet", "plasma", "viridis", "hot"]

    for ax, title, hmap, cmap in zip(axes[1], raw_titles, raw_maps, cmaps):
        im = ax.imshow(hmap, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "explanation.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[XAI] Explanation saved → {save_path}")

    plt.show()

    # Package results
    results = {
        "prediction": class_names[pred_class],
        "confidence": confidence,
        "pred_class": pred_class,
        "probs": probs.cpu().numpy(),
        "gradcam_map": gradcam_map,
        "ig_map": ig_map,
        "shap_map": shap_map,
        "ensemble_map": ensemble_map,
        "gradcam_overlay": gradcam_overlay,
        "ig_overlay": ig_overlay,
        "shap_overlay": shap_overlay,
        "ensemble_overlay": ensemble_overlay,
        "consistency": consistency,
        "figure": fig,
    }

    return results
