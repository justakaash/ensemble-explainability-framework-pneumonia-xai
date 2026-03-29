# =============================================================
# app.py - Streamlit Clinical Interface (Liquid Glass v3)
# Project: Multi-Explainability Deep Learning Framework
#          for Reliable Early Pneumonia Detection
# =============================================================

import os
import sys
import torch
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import load_image_from_upload
from model import build_model
from explain import explain_prediction

# -------------------------------------------------------------
# 1. PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="Lung Health AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------------------
# 2. CSS — Liquid Glass
# -------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

/* ── Animated Tech Panel ── */
    div[data-testid="stVerticalBlock"] > div:has(.tech-item) {
        animation: slideDown 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        transform-origin: top;
        overflow: hidden;
    }

    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-12px) scaleY(0.95);
        }
        to {
            opacity: 1;
            transform: translateY(0) scaleY(1);
        }
    }

    /* ── Button styling ── */
    div[data-testid="stButton"] button {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(0,212,255,0.2) !important;
        border-radius: 12px !important;
        color: rgba(255,255,255,0.7) !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        transition: all 0.3s ease !important;
        padding: 0.5rem 1.2rem !important;
    }
    div[data-testid="stButton"] button:hover {
        background: rgba(0,212,255,0.08) !important;
        border-color: rgba(0,212,255,0.4) !important;
        color: white !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(0,212,255,0.15) !important;
    }

/* ── Floating Team Button ── */
    #team-float {
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 99999;
    }
    #team-float .tf-btn {
        background: rgba(13,17,35,0.85);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0,212,255,0.25);
        border-radius: 50px;
        padding: 0.5rem 1.2rem;
        color: rgba(255,255,255,0.7);
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        transition: all 0.3s ease;
        white-space: nowrap;
    }
    #team-float .tf-btn:hover {
        border-color: rgba(0,212,255,0.6);
        color: white;
        box-shadow: 0 4px 24px rgba(0,212,255,0.2);
    }
    #team-float .tf-dot {
        width: 6px; height: 6px;
        background: #00d4ff;
        border-radius: 50%;
        box-shadow: 0 0 6px rgba(0,212,255,0.8);
        animation: pulse 2s infinite;
    }
    #team-float .tf-panel {
        display: none;
        position: absolute;
        top: calc(100% + 0.6rem);
        right: 0;
        background: rgba(13,17,35,0.95);
        backdrop-filter: blur(30px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 1.2rem;
        min-width: 560px;
        box-shadow: 0 8px 40px rgba(0,0,0,0.5),
                    inset 0 1px 0 rgba(255,255,255,0.08);
        animation: fadeSlide 0.35s cubic-bezier(0.16,1,0.3,1);
    }
    #team-float[open] .tf-panel { display: block; }
    #team-float[open] > .tf-btn {
        border-color: rgba(0,212,255,0.5);
        background: rgba(0,212,255,0.08);
        color: white;
    }
    @keyframes fadeSlide {
        from { opacity:0; transform:translateY(-8px) scale(0.97); }
        to   { opacity:1; transform:translateY(0) scale(1); }
    }
    .tf-title {
        font-size: 0.62rem;
        font-weight: 700;
        color: rgba(0,212,255,0.6);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.8rem;
        text-align: center;
    }
    .tf-grid {
        display: grid;
        grid-template-columns: repeat(3,1fr);
        gap: 0.8rem;
        margin-bottom: 0.8rem;
    }
    .tf-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 0.9rem 0.7rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .tf-card:hover {
        background: rgba(0,212,255,0.06);
        border-color: rgba(0,212,255,0.2);
        transform: translateY(-2px);
    }
    .tf-av {
        width: 34px; height: 34px;
        border-radius: 50%;
        background: linear-gradient(135deg,#00d4ff,#7c3aed);
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 0.5rem auto;
        font-size: 0.9rem; font-weight: 800; color: white;
        box-shadow: 0 0 12px rgba(0,212,255,0.3);
    }
    .tf-name {
        font-size: 0.75rem; font-weight: 700;
        color: rgba(255,255,255,0.85);
        margin-bottom: 0.2rem; line-height: 1.3;
    }
    .tf-reg {
        font-size: 0.65rem; color: rgba(0,212,255,0.65);
        font-family: monospace; margin-bottom: 0.3rem;
    }
    .tf-mail {
        font-size: 0.65rem; color: rgba(255,255,255,0.35);
        word-break: break-all; line-height: 1.4;
    }
    .tf-dept {
        text-align: center; font-size: 0.63rem;
        color: rgba(255,255,255,0.2); letter-spacing: 0.03em;
    }
    /* Remove default <details>/<summary> browser styling */
    details#team-float { list-style: none; }
    details#team-float > summary { list-style: none; }
    details#team-float > summary::-webkit-details-marker { display: none; }
    details#team-float > summary::marker { display: none; }

/* Background */
.stApp {
    background: radial-gradient(ellipse at 0% 0%, #0d1f3c 0%, #0a0a1a 40%,
                #0d0d1f 70%, #0a1628 100%) !important;
    min-height: 100vh;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 3rem !important;
    max-width: 1400px !important;
}

/* ── Glass Card ── */
.g-card {
    background: linear-gradient(135deg,
        rgba(255,255,255,0.07) 0%,
        rgba(255,255,255,0.03) 100%);
    backdrop-filter: blur(30px) saturate(150%);
    -webkit-backdrop-filter: blur(30px) saturate(150%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 24px;
    padding: 1.5rem 2rem;
    margin-bottom: 1rem;
    box-shadow:
        0 8px 32px rgba(0,0,0,0.5),
        inset 0 1px 0 rgba(255,255,255,0.12);
    position: relative;
    overflow: hidden;
}
.g-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg,
        transparent 0%,
        rgba(255,255,255,0.25) 50%,
        transparent 100%);
}

/* ── Small glass pill ── */
.g-pill {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 0.6rem;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg,
        #ffffff 0%, #a8d8ea 35%,
        #00d4ff 65%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    letter-spacing: -0.03em;
    margin-bottom: 0.8rem;
}
.hero-sub {
    font-size: 1rem;
    color: rgba(255,255,255,0.4);
    font-weight: 300;
    max-width: 600px;
    margin: 0 auto 0.5rem auto;
    line-height: 1.6;
}

/* ── Section label ── */
.sec-label {
    font-size: 0.7rem;
    font-weight: 700;
    color: rgba(0,212,255,0.7);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.4rem;
}
.sec-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: rgba(255,255,255,0.9);
    margin-bottom: 1rem;
}

/* ── Prediction ── */
.pred-pneumonia {
    background: linear-gradient(135deg,
        rgba(255,68,68,0.25) 0%,
        rgba(200,30,30,0.15) 100%);
    border: 1px solid rgba(255,100,100,0.35);
    border-radius: 18px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(255,68,68,0.15),
                inset 0 1px 0 rgba(255,255,255,0.08);
}
.pred-pneumonia h2 {
    font-size: 1.6rem;
    font-weight: 800;
    color: #ff6b6b;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
}
.pred-normal {
    background: linear-gradient(135deg,
        rgba(0,200,81,0.25) 0%,
        rgba(0,150,60,0.15) 100%);
    border: 1px solid rgba(0,220,100,0.35);
    border-radius: 18px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(0,200,81,0.15),
                inset 0 1px 0 rgba(255,255,255,0.08);
}
.pred-normal h2 {
    font-size: 1.6rem;
    font-weight: 800;
    color: #00e676;
    margin: 0 0 0.5rem 0;
}
.pred-desc {
    font-size: 0.85rem;
    line-height: 1.6;
    color: rgba(255,255,255,0.5);
    margin-top: 0.5rem;
}

/* ── XAI Cards ── */
.xai-wrap {
    background: linear-gradient(135deg,
        rgba(255,255,255,0.05) 0%,
        rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.xai-method {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.xai-what {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.4);
    line-height: 1.5;
    margin-top: 0.6rem;
}
.xai-find {
    font-size: 0.78rem;
    color: rgba(255,210,80,0.9);
    font-weight: 500;
    margin-top: 0.5rem;
    padding: 0.35rem 0.7rem;
    background: rgba(255,200,50,0.07);
    border-radius: 8px;
    border-left: 2px solid rgba(255,200,50,0.35);
    line-height: 1.5;
}

/* ── Badges ── */
.badge {
    display: inline-block;
    border-radius: 50px;
    padding: 0.35rem 1rem;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.03em;
}
.badge-high {
    background: linear-gradient(135deg,#00c851,#00e676);
    color: #001a0a;
    box-shadow: 0 0 16px rgba(0,200,81,0.4);
}
.badge-med {
    background: linear-gradient(135deg,#ff8c00,#ffbb33);
    color: #1a0a00;
    box-shadow: 0 0 16px rgba(255,140,0,0.4);
}
.badge-low {
    background: linear-gradient(135deg,#ff4444,#ff6b6b);
    color: #1a0000;
    box-shadow: 0 0 16px rgba(255,68,68,0.4);
}

/* ── Tech box ── */
.tech-item {
    background: rgba(0,212,255,0.04);
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}
.tech-item .t-label {
    font-size: 0.65rem;
    font-weight: 700;
    color: rgba(0,212,255,0.6);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
}
.tech-item .t-val {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.75);
    line-height: 1.6;
}

/* ── Disclaimer ── */
.disc {
    background: rgba(255,140,0,0.05);
    border: 1px solid rgba(255,140,0,0.18);
    border-radius: 14px;
    padding: 1rem 1.4rem;
    color: rgba(255,200,100,0.75);
    font-size: 0.82rem;
    line-height: 1.6;
    margin-top: 1.5rem;
}

/* ── Streamlit overrides ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg,#00d4ff,#7c3aed) !important;
    border-radius: 10px !important;
}
div[data-testid="stMetricValue"] {
    color: #00d4ff !important;
    font-weight: 700 !important;
}
.stExpander {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
}
.stExpander summary {
    color: rgba(255,255,255,0.7) !important;
    font-weight: 500 !important;
}
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 2px dashed rgba(0,212,255,0.25) !important;
    border-radius: 20px !important;
    padding: 1rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# -------------------------------------------------------------
# 3. MODEL LOADING
# -------------------------------------------------------------
@st.cache_resource
def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, device = build_model(num_classes=2, pretrained=False, device=device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, device, True
    return model, device, False


# -------------------------------------------------------------
# 4. CHEST X-RAY VALIDATOR
# -------------------------------------------------------------
def is_chest_xray(pil_image):
    """
    Heuristic check to reject obviously non-X-ray images.
    Returns (is_valid: bool, reason: str).
    """
    img_rgb = pil_image.convert("RGB")
    arr = np.array(img_rgb).astype(float)
    gray = np.mean(arr, axis=2)

    # --- Check 1: X-rays are grayscale (R≈G≈B) ---
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    color_diff = np.mean(
        [
            np.mean(np.abs(r - g)),
            np.mean(np.abs(r - b)),
            np.mean(np.abs(g - b)),
        ]
    )
    if color_diff > 15:
        return False, (
            "This looks like a color photograph, not a chest X-ray. "
            "X-rays are always grayscale (black & white). Please upload a real chest X-ray image."
        )

    # --- Check 2: Aspect ratio — chest X-rays are roughly square (PA/AP view) ---
    w, h = pil_image.size
    ratio = w / h
    if ratio < 0.6 or ratio > 1.5:
        return False, (
            f"Unusual image dimensions ({w}×{h} px, ratio {ratio:.2f}). "
            "Chest X-rays are typically square or slightly rectangular (ratio 0.6–1.5). "
            "This image looks like a wide-format or portrait photo."
        )

    # --- Check 3: Minimum resolution ---
    if w < 64 or h < 64:
        return False, (
            f"Image is too small ({w}×{h} px). "
            "Please upload a higher-resolution chest X-ray."
        )

    # --- Check 4: Must not be blank / near-uniform ---
    if np.std(gray) < 8:
        return False, (
            "Image appears blank or nearly uniform. "
            "Please upload a real chest X-ray with visible lung structure."
        )

    # --- Check 5: X-rays must have significant dark regions ---
    # Real X-rays always have dark background + dark lung fields
    dark_fraction = np.mean(gray < 30)
    if dark_fraction < 0.08:
        return False, (
            f"No significant dark regions found (only {dark_fraction * 100:.1f}% dark pixels). "
            "Chest X-rays always have a dark background and dark lung fields. "
            "This image looks like a regular photograph."
        )

    # --- Check 6: X-rays must also have bright regions (bone/tissue) ---
    bright_fraction = np.mean(gray > 180)
    if bright_fraction < 0.03:
        return False, (
            "No bright regions found. Chest X-rays always have bright areas "
            "(ribs, spine, heart shadow). This does not look like an X-ray."
        )

    return True, ""


# -------------------------------------------------------------
# 5. CONFIDENCE GAUGE
# -------------------------------------------------------------
def plot_confidence_gauge(confidence, prediction):
    fig, ax = plt.subplots(figsize=(3.5, 2.8), subplot_kw=dict(polar=True))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    color = "#ff6b6b" if prediction == "Pneumonia" else "#00e676"
    theta_bg = np.linspace(0, np.pi, 100)
    theta = np.linspace(0, np.pi * confidence, 100)

    # Background track
    ax.plot(theta_bg, [1] * 100, color="#2a2a4a", linewidth=10, solid_capstyle="round")
    # Value arc
    ax.plot(
        theta, [1] * 100, color=color, linewidth=10, solid_capstyle="round", alpha=0.95
    )

    ax.set_ylim(0, 1.3)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)

    ax.text(
        np.pi / 2,
        0.25,
        f"{confidence * 100:.1f}%",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="800",
        color=color,
        fontfamily="Inter",
    )
    ax.text(
        np.pi / 2,
        -0.15,
        "AI Certainty",
        ha="center",
        va="center",
        fontsize=8,
        color="#888888",
        fontfamily="Inter",
    )
    plt.tight_layout(pad=0)
    return fig


# -------------------------------------------------------------
# 6. CONSISTENCY BARS
# -------------------------------------------------------------
def plot_consistency_bars(consistency):
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_alpha(0)
    ax.set_facecolor("#0d1117")

    methods = [
        "GradCAM++\nvs IG",
        "GradCAM++\nvs SHAP",
        "IG vs\nSHAP",
        "Overall\nScore",
    ]
    values = [
        consistency["iou_gradcam_ig"],
        consistency["iou_gradcam_shap"],
        consistency["iou_ig_shap"],
        consistency["consistency_score"],
    ]
    colors = ["#00d4ff", "#7c3aed", "#00c851", "#ff8c00"]

    bars = ax.bar(
        methods, values, color=colors, edgecolor="none", width=0.45, alpha=0.9, zorder=3
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Agreement", color="#666688", fontsize=8)
    ax.tick_params(colors="#888899", labelsize=8)
    ax.grid(axis="y", color="#1e1e2e", linewidth=1, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#1e1e2e")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            color="white",
            fontsize=8,
            fontweight="700",
        )
    plt.tight_layout()
    return fig


# -------------------------------------------------------------
# 7. XAI DESCRIPTIONS
# -------------------------------------------------------------
def get_xai_info(prediction, confidence):
    p = "pneumonia" if prediction == "Pneumonia" else "no pneumonia"
    c = f"{confidence * 100:.1f}%"
    return {
        "gradcam": {
            "color": "#ff6b6b",
            "label": "🔴 Grad-CAM++ — Where AI Looked",
            "what": "Red/yellow = high attention zones. Blue/green = low attention. Shows which lung areas the AI focused on.",
            "find": f"AI focused on specific lung zones to detect {p} ({c} confidence). Red hotspots = suspicious regions.",
        },
        "ig": {
            "color": "#b794f4",
            "label": "🟣 Integrated Gradients — Pixel Evidence",
            "what": "Shows individual pixels that most influenced the decision. Bright = strong evidence. Dark = less relevant.",
            "find": "Bright scattered regions = fine-grained lung texture patterns used as diagnostic evidence.",
        },
        "shap": {
            "color": "#68d391",
            "label": "🟢 SHAP — Feature Voting Map",
            "what": 'Based on game theory — shows how much each region "voted" for the diagnosis. More glow = stronger vote.',
            "find": "Highlighted zones are statistically the most significant contributors to the final prediction.",
        },
        "ensemble": {
            "color": "#fbd38d",
            "label": "⭐ Ensemble — AI Consensus Map",
            "what": "Combines all 3 methods above. Where they all agree, it glows brighter. Most reliable map to interpret.",
            "find": "Warm/bright areas = strongest cross-method consensus. These regions are most confidently linked to the result.",
        },
    }


# -------------------------------------------------------------
# 8. MAIN
# -------------------------------------------------------------
def main():
    checkpoint_path = "outputs/checkpoints/best_model.pth"
    model, device, model_loaded = load_model(checkpoint_path)

    # ── Hero ──
    st.markdown(
        """
    <div class="hero">
        <div class="hero-title">🫁 Is Your Lung Healthy?</div>
        <div class="hero-sub">
            Upload a chest X-ray and our AI instantly checks for
            pneumonia — then shows you <em>exactly</em> what it
            found, in plain language.
        </div>
        <div style="margin-top:0.6rem;display:flex;align-items:center;
             justify-content:center;gap:0.5rem;flex-wrap:wrap;">
            <span style="font-size:0.72rem;color:rgba(255,255,255,0.2);
                  letter-spacing:0.04em;">Powered by</span>
            <span style="font-size:0.72rem;font-weight:600;
                  color:rgba(0,212,255,0.5);letter-spacing:0.03em;">DenseNet-121</span>
            <span style="color:rgba(255,255,255,0.1);font-size:0.65rem;">·</span>
            <span style="font-size:0.72rem;font-weight:600;
                  color:rgba(0,212,255,0.5);letter-spacing:0.03em;">Grad-CAM++</span>
            <span style="color:rgba(255,255,255,0.1);font-size:0.65rem;">·</span>
            <span style="font-size:0.72rem;font-weight:600;
                  color:rgba(0,212,255,0.5);letter-spacing:0.03em;">Integrated Gradients</span>
            <span style="color:rgba(255,255,255,0.1);font-size:0.65rem;">·</span>
            <span style="font-size:0.72rem;font-weight:600;
                  color:rgba(0,212,255,0.5);letter-spacing:0.03em;">SHAP</span>
            <span style="color:rgba(255,255,255,0.1);font-size:0.65rem;">·</span>
            <span style="font-size:0.72rem;font-weight:600;
                  color:rgba(124,58,237,0.6);letter-spacing:0.03em;">PyTorch</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Technical Details Expander ──
    # ── Technical Details Toggle ──
    if "show_tech" not in st.session_state:
        st.session_state.show_tech = False

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        if st.button("📋 Technical Details", use_container_width=True):
            st.session_state.show_tech = not st.session_state.show_tech

    if st.session_state.show_tech:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                """
            <div class="tech-item">
                <div class="t-label">AI Model Architecture</div>
                <div class="t-val">
                    DenseNet-121 CNN<br>
                    7,610,498 parameters<br>
                    Pretrained: ImageNet<br>
                    Fine-tuned: 5,232 X-rays<br>
                    Dropout: 0.5 | Optimizer: AdamW
                </div>
            </div>
            <div class="tech-item">
                <div class="t-label">XAI Methods</div>
                <div class="t-val">
                    • Grad-CAM++ (gradient activation)<br>
                    • Integrated Gradients (Captum)<br>
                    • GradientSHAP (Captum)<br>
                    • Ensemble Map (novel contribution)<br>
                    • XAI Consistency Score (novel metric)
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
            <div class="tech-item">
                <div class="t-label">Test Set Performance</div>
                <div class="t-val">
                    Accuracy:    85.42%<br>
                    AUC-ROC:     96.90%<br>
                    Sensitivity: 99.49%<br>
                    Specificity: 61.97%<br>
                    F1-Score:    89.50%<br>
                    Precision:   81.34%<br>
                    ECE:         0.1422
                </div>
            </div>
            <div class="tech-item">
                <div class="t-label">Training Configuration</div>
                <div class="t-val">
                    Epochs: 30 (early stopping)<br>
                    Batch size: 32<br>
                    LR: 1e-4 → 1e-5 (cosine)<br>
                    Augmentation: Albumentations<br>
                    CLAHE preprocessing: Yes<br>
                    Class weighting: Yes
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                """
            <div class="tech-item">
                <div class="t-label">Software Stack</div>
                <div class="t-val">
                    Python 3.12<br>
                    PyTorch 2.0 + CUDA 12.4<br>
                    GPU: NVIDIA RTX 4070<br>
                    Captum (XAI library)<br>
                    Albumentations, OpenCV<br>
                    scikit-learn, Streamlit
                </div>
            </div>
            <div class="tech-item">
                <div class="t-label">Paper</div>
                <div class="t-val">
                    Ensemble Explainability Framework
                    with XAI Consistency Score for
                    Pneumonia Diagnosis Using
                    DenseNet_121<br><br>
                    Dataset: Kaggle Chest X-Ray
                    (Pneumonia) — 5,232 images
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Upload ──
    st.markdown(
        """
    <div class="g-card">
        <div class="sec-label">Step 1</div>
        <div class="sec-title">📤 Upload Your Chest X-Ray</div>
        <p style="color:rgba(255,255,255,0.35);font-size:0.85rem;margin:0;">
        Supported: JPG, JPEG, PNG &nbsp;·&nbsp; Max 200MB
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload chest X-ray", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )

    if not model_loaded:
        st.markdown(
            """
        <div class="disc">
        ⚠️ <strong>Model not found.</strong>
        Run <code>python src/train.py</code> first.
        </div>
        """,
            unsafe_allow_html=True,
        )
        return

    if uploaded_file and model_loaded:
        input_tensor, pil_image = load_image_from_upload(uploaded_file)

        # ── Validate: reject non-X-ray images ──
        valid, reason = is_chest_xray(pil_image)
        if not valid:
            st.markdown(
                f"""
            <div style="
                background: linear-gradient(135deg,
                    rgba(255,140,0,0.18) 0%,
                    rgba(180,80,0,0.10) 100%);
                border: 1px solid rgba(255,160,0,0.4);
                border-radius: 20px;
                padding: 1.8rem 2rem;
                margin-top: 1rem;
                box-shadow: 0 0 40px rgba(255,140,0,0.12);
            ">
                <div style="font-size:1.4rem;font-weight:800;
                     color:#ffaa33;margin-bottom:0.6rem;">
                    ⚠️ Invalid Image
                </div>
                <div style="font-size:0.9rem;color:rgba(255,220,150,0.85);
                     line-height:1.7;">
                    {reason}
                </div>
                <div style="margin-top:1rem;font-size:0.8rem;
                     color:rgba(255,255,255,0.3);line-height:1.6;
                     padding:0.8rem 1rem;
                     background:rgba(255,255,255,0.03);
                     border-radius:10px;">
                    💡 <strong style="color:rgba(255,255,255,0.5);">
                    Tip:</strong> Use a standard front-facing chest
                    X-ray (PA or AP view) saved as JPG or PNG.
                    X-ray images are always grayscale (black &amp; white).
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.stop()

        input_tensor = input_tensor.to(device)

        with st.spinner("🧠 Analyzing X-ray..."):
            results = explain_prediction(
                model, input_tensor, pil_image, device, save_dir="outputs/explanations"
            )

        pred = results["prediction"]
        conf = results["confidence"]
        probs = results["probs"]
        consistency = results["consistency"]
        xai_info = get_xai_info(pred, conf)

        # ── Results Row ──
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1.1, 1.2, 0.9])

        with col1:
            st.markdown(
                """
            <div class="g-card">
                <div class="sec-label">Original Image</div>
                <div class="sec-title">🩻 Your X-Ray</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.image(pil_image.resize((380, 380)), use_container_width=True)

        with col2:
            st.markdown(
                """
            <div class="g-card">
                <div class="sec-label">AI Diagnosis</div>
                <div class="sec-title">🎯 What We Found</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            if pred == "Pneumonia":
                st.markdown(
                    f"""
                <div class="pred-pneumonia">
                    <h2>🦠 Signs of Pneumonia Detected</h2>
                    <p class="pred-desc">
                    Our AI found patterns commonly linked to pneumonia —
                    such as cloudiness or white patches in the lung area.
                    <strong style="color:#ff9999;">
                    Please consult a doctor immediately.</strong>
                    </p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div class="pred-normal">
                    <h2>✅ Lungs Appear Normal</h2>
                    <p class="pred-desc">
                    No significant pneumonia patterns detected.
                    Lung patterns appear within normal range.
                    <strong style="color:#99ffbb;">
                    Always confirm with a qualified doctor.</strong>
                    </p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                """
            <div class="g-pill">
                <div class="sec-label">Probability Breakdown</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.caption(f"Normal lungs: {probs[0] * 100:.1f}%")
            st.progress(float(probs[0]))
            st.caption(f"Pneumonia signs: {probs[1] * 100:.1f}%")
            st.progress(float(probs[1]))

        with col3:
            st.markdown(
                """
            <div class="g-card">
                <div class="sec-label">Confidence Level</div>
                <div class="sec-title">📊 AI Certainty</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            gauge = plot_confidence_gauge(conf, pred)
            st.pyplot(gauge, use_container_width=True)
            plt.close()

            certainty = (
                "Very High ✅"
                if conf > 0.9
                else "High ✅"
                if conf > 0.75
                else "Moderate ⚠️"
                if conf > 0.5
                else "Low ❌"
            )
            st.markdown(
                f"""
            <div class="g-pill" style="text-align:center;">
                <div class="sec-label">Certainty Level</div>
                <div style="font-size:1rem;font-weight:700;
                    color:white;">{certainty}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # ── XAI Maps ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
        <div class="g-card">
            <div class="sec-label">Explainable AI</div>
            <div class="sec-title">
                🧠 How the AI Made Its Decision
            </div>
            <p style="color:rgba(255,255,255,0.35);
               font-size:0.85rem;margin:0;line-height:1.6;">
            We use 3 different AI methods to highlight suspicious
            lung regions. Each uses a different approach — when
            they agree, the result is more trustworthy.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        xai_cols = st.columns(4)
        xai_pairs = [
            ("gradcam", results["gradcam_overlay"]),
            ("ig", results["ig_overlay"]),
            ("shap", results["shap_overlay"]),
            ("ensemble", results["ensemble_overlay"]),
        ]

        for col, (key, overlay) in zip(xai_cols, xai_pairs):
            info = xai_info[key]
            with col:
                st.markdown(
                    f"""
                <div class="xai-wrap">
                    <div class="xai-method"
                         style="color:{info["color"]};">
                         {info["label"]}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
                st.image(overlay, use_container_width=True)
                st.markdown(
                    f"""
                <div class="xai-wrap">
                    <div class="xai-what">{info["what"]}</div>
                    <div class="xai-find">💡 {info["find"]}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # ── Consistency ──
        st.markdown("<br>", unsafe_allow_html=True)
        cs = consistency["consistency_score"]

        if cs >= 0.7:
            badge = "<span class='badge badge-high'>HIGH AGREEMENT ✅</span>"
            interp = "All three AI methods strongly agree on the suspicious regions — highly trustworthy result."
        elif cs >= 0.4:
            badge = "<span class='badge badge-med'>MODERATE AGREEMENT ⚠️</span>"
            interp = "Methods partially agree. Result is likely correct but view with some caution."
        else:
            badge = "<span class='badge badge-low'>COMPLEMENTARY VIEWS ℹ️</span>"
            interp = "Each method captures different aspects of the same finding — this is expected and shows a comprehensive multi-perspective analysis."

        st.markdown(
            f"""
        <div class="g-card">
            <div class="sec-label">Reliability Check</div>
            <div class="sec-title">
                📐 Do the AI Methods Agree?
            </div>
            <p style="color:rgba(255,255,255,0.35);
               font-size:0.85rem;margin-bottom:1rem;line-height:1.6;">
            We check if all 3 explanation methods point to the
            same lung areas. Higher agreement = more reliable result.
            </p>
            {badge}
            <span style="color:rgba(255,255,255,0.3);
                font-size:0.82rem;margin-left:0.8rem;">
                Score: <strong style="color:white;">
                {cs:.4f}</strong> / 1.0000
            </span>
            <p style="color:rgba(255,255,255,0.4);
               font-size:0.82rem;margin-top:0.8rem;
               line-height:1.6;">{interp}</p>
            <p style="color:rgba(255,255,255,0.35);
               font-size:0.82rem;margin-top:0.6rem;
               line-height:1.7;
               padding:0.8rem 1rem;
               background:rgba(255,255,255,0.03);
               border-radius:10px;
               border-left:3px solid rgba(0,212,255,0.3);">
               💬 <strong style="color:rgba(255,255,255,0.6);">
               In simple terms:</strong>
               A score of <strong style="color:white;">
               {cs:.4f}</strong> means the 3 AI methods
               agree on about <strong style="color:white;">
               {cs * 100:.1f}%</strong> of the lung regions
               they highlighted. This is like asking 3
               doctors to circle suspicious areas on the
               same X-ray — they may circle slightly
               different spots, but they all reached the
               same diagnosis. The <strong
               style="color:#fbd38d;">prediction itself
               remains highly reliable</strong> regardless
               of this score.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(
                """
            <div class="g-card">
                <div class="sec-label">Pairwise Scores</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.metric("GradCAM++ vs IG", f"{consistency['iou_gradcam_ig']:.4f}")
            st.metric("GradCAM++ vs SHAP", f"{consistency['iou_gradcam_shap']:.4f}")
            st.metric("IG vs SHAP", f"{consistency['iou_ig_shap']:.4f}")
        with col2:
            bar_fig = plot_consistency_bars(consistency)
            st.pyplot(bar_fig, use_container_width=True)
            plt.close()

        # ── Disclaimer ──
        st.markdown(
            """
        <div class="disc">
        ⚠️ <strong>Clinical Disclaimer:</strong>
        This tool is for <strong>research and educational
        purposes only</strong>. It is not a substitute for
        professional medical diagnosis. Always consult a
        qualified doctor or radiologist. Early pneumonia
        can be subtle — professional evaluation is essential.
        </div>
        """,
            unsafe_allow_html=True,
        )

    else:
        # ── Landing ──
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        cards = [
            (
                "📤",
                "Step 1",
                "Upload X-Ray",
                "Drag & drop a chest X-ray in JPG or PNG format above.",
            ),
            (
                "🧠",
                "Step 2",
                "AI Analyzes",
                "Deep learning scans for pneumonia patterns in seconds.",
            ),
            (
                "📊",
                "Step 3",
                "See Evidence",
                "4 visual maps show exactly what the AI examined and why.",
            ),
        ]
        for col, (icon, step, title, desc) in zip([c1, c2, c3], cards):
            with col:
                st.markdown(
                    f"""
                <div class="g-card" style="text-align:center;
                     padding:2rem 1.5rem;">
                    <div style="font-size:2.5rem;
                         margin-bottom:0.8rem;">{icon}</div>
                    <div class="sec-label">{step}</div>
                    <div style="font-size:1rem;font-weight:700;
                         color:rgba(255,255,255,0.9);
                         margin-bottom:0.5rem;">{title}</div>
                    <div style="font-size:0.83rem;
                         color:rgba(255,255,255,0.35);
                         line-height:1.6;">{desc}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )


# -------------------------------------------------------------
# 8. FLOATING TEAM BAR
# -------------------------------------------------------------
def render_team_bar():
    st.markdown(
        """
    <details id="team-float">
        <summary class="tf-btn">
            <span class="tf-dot"></span>
            Meet the Team
            <span style="font-size:0.65rem;opacity:0.5;">▼</span>
        </summary>
        <div class="tf-panel">
            <div class="tf-grid">
                <div class="tf-card">
                    <div class="tf-av">A</div>
                    <div class="tf-name">Akaash Srinivasan</div>
                    <div class="tf-reg">113222031006</div>
                    <div class="tf-mail">aksrini28@gmail.com</div>
                </div>
                <div class="tf-card">
                    <div class="tf-av">C</div>
                    <div class="tf-name">Chandn Sai Kumar S</div>
                    <div class="tf-reg">113222031027</div>
                    <div class="tf-mail">chandnveccseb@gmail.com</div>
                </div>
                <div class="tf-card">
                    <div class="tf-av">G</div>
                    <div class="tf-name">Gokul Raj V</div>
                    <div class="tf-reg">113222031041</div>
                    <div class="tf-mail">vgokulrajvec@gmail.com</div>
                </div>
            </div>
            <div class="tf-dept">
                VEC | 2022-2026
            </div>
        </div>
    </details>
    """,
        unsafe_allow_html=True,
    )


render_team_bar()


if __name__ == "__main__":
    main()
