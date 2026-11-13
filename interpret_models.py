"""
interpret_models.py
===================
Model Interpretation & Explainability for PPG Arrhythmia Detection.

Implements four complementary explainability techniques across all trained
model architectures (XGBoost, Edge CNN, BiLSTM, TCN):

  1. SHAP (SHapley Additive exPlanations)
     - TreeExplainer   -> XGBoost feature importances (global + local)
     - DeepExplainer   -> Deep-learning models (CNN, BiLSTM, TCN)
     - Summary plots, beeswarm plots, force plots, waterfall plots

  2. Grad-CAM (Gradient-weighted Class Activation Maps)
     - Highlights which TIME REGIONS of the raw PPG signal most influence
       each class prediction.
     - Works for all Keras 1-D convolutional models (Edge CNN, BiLSTM, TCN).
     - Produces per-class heatmaps overlaid on the raw waveform.

  3. Integrated Gradients
     - Attribution method that satisfies the completeness axiom (attributions
       sum to f(x) - f(baseline)).
     - Baseline = flat-line signal (zeros), representing "no information".
     - Produces sample-level attribution maps on raw PPG windows.

  4. LIME (Local Interpretable Model-agnostic Explanations)
     - Segment-based perturbation: divides the 10-second PPG window into
       ~25 segments, perturbs each by zeroing it out, and trains a local
       linear proxy model around the prediction.
     - Works for ANY black-box model.

Outputs (written to outputs/<model_name>/interpretation/)
---------
  shap_summary.png           -- SHAP feature importance summary
  shap_beeswarm.png          -- SHAP beeswarm (dot plot)
  gradcam_<class>_sample<n>.png  -- Grad-CAM heatmap per class
  intgrad_sample<n>.png      -- Integrated Gradients attribution
  lime_sample<n>.png         -- LIME local explanation
  interpretation_report.txt  -- Text summary of key findings

Usage
-----
  # All models (auto-discovers saved .h5 / .ubj files):
  python interpret_models.py

  # Specific model(s):
  python interpret_models.py --models xgb cnn

  # Control number of test samples to explain:
  python interpret_models.py --n-samples 20

  # Pick specific classes to explain:
  python interpret_models.py --classes Normal AFib

  # Disable heavy DeepSHAP (faster run):
  python interpret_models.py --no-deep-shap

  # Force CPU (avoids GPU OOM during gradient computations):
  python interpret_models.py --no-gpu
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import warnings
import time
from pathlib import Path
from typing import Optional, List
# Setting NUMBA_CUDA_DRIVER to a non-existent path prevents the probe
# entirely without disabling TensorFlow GPU.
# ---------------------------------------------------------------------------
os.environ.setdefault("NUMBA_CUDA_DRIVER", "0")          # skip CUDA driver probe
os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")          # numba: no GPU needed
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")        # silence TF C++ logs

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# -- Suppress noisy TF/SHAP/numba warnings ------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="shap")
warnings.filterwarnings("ignore", message=".*cublasLt.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Cannot find cuBLAS.*")
warnings.filterwarnings("ignore", message=".*CUDA.*", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# -- Project constants --------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs")
DATASET_PATH = os.path.join(OUTPUT_DIR, "dataset.npz")
FEATURES_TEST = os.path.join(OUTPUT_DIR, "features_test.parquet")

NUM_CLASSES  = 5
INPUT_LENGTH = 1250   # 10 s x 125 Hz
FS           = 125    # Hz

CLASS_NAMES = ["Normal", "AFib", "Bradycardia", "Tachycardia", "Other"]

# One vivid colour per class
CLASS_COLORS = ["#4CAF50", "#F44336", "#2196F3", "#FF9800", "#9C27B0"]


# =============================================================================
#  DATA LOADERS
# =============================================================================

def load_raw_test() -> tuple:
    """Load raw PPG test windows from dataset.npz."""
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Run: python preprocessing/fuse_datasets.py  first."
        )
    data = np.load(DATASET_PATH, allow_pickle=True)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int32)
    logger.info("Raw test set loaded: %s  labels=%s", X_test.shape, y_test.shape)
    return X_test, y_test


def load_feature_test() -> tuple:
    """Load tabular feature test set (for XGBoost SHAP)."""
    import pandas as pd

    path = FEATURES_TEST
    if not os.path.isfile(path):
        csv_path = path.replace(".parquet", ".csv")
        if os.path.isfile(csv_path):
            path = csv_path
        else:
            raise FileNotFoundError(
                f"Feature file not found: {FEATURES_TEST}\n"
                "Run: python feature_extraction.py  first."
            )

    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    y  = df["label"].values.astype(np.int32)
    X  = df.drop(columns=["label"])
    feature_names = list(X.columns)
    return X.values.astype(np.float32), y, feature_names


def sample_per_class(X, y, n_per_class=5, seed=42):
    """
    Return (X_sub, y_sub, indices) with up to n_per_class samples per class.
    Balances representation so minority classes are always visible.
    """
    rng = np.random.RandomState(seed)
    idx = []
    for c in range(NUM_CLASSES):
        cls_idx = np.where(y == c)[0]
        if len(cls_idx) == 0:
            continue
        chosen = rng.choice(cls_idx, size=min(n_per_class, len(cls_idx)), replace=False)
        idx.extend(chosen.tolist())
    idx = np.array(idx, dtype=int)
    return X[idx], y[idx], idx


# =============================================================================
#  MODEL LOADERS
# =============================================================================

def load_keras_model(model_name):
    """Load a saved Keras model (.h5) from outputs/<model_name>/."""
    import tensorflow as tf

    candidates = [
        os.path.join(OUTPUT_DIR, model_name, f"best_{model_name}.h5"),
        os.path.join(OUTPUT_DIR, f"best_{model_name}.h5"),
        os.path.join(OUTPUT_DIR, "best_model.h5"),
        os.path.join(OUTPUT_DIR, "final_model.h5"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            logger.info("Loading Keras model from %s", path)
            try:
                model = tf.keras.models.load_model(path, compile=False)
                return model
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)
    logger.error("No saved Keras model found for '%s'. Searched:\n  %s",
                 model_name, "\n  ".join(candidates))
    return None


def load_xgb_model():
    """Load a saved XGBoost model (.ubj) from outputs/xgb/."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("xgboost not installed. Run: pip install xgboost")
        return None

    candidates = [
        os.path.join(OUTPUT_DIR, "xgb", "xgb_model.ubj"),
        os.path.join(OUTPUT_DIR, "xgb_model.ubj"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            logger.info("Loading XGBoost model from %s", path)
            model = xgb.Booster()
            model.load_model(path)
            return model
    logger.error("No XGBoost model found.")
    return None


# =============================================================================
#  UTILITY HELPERS
# =============================================================================

def _save(fig, path):
    """Save figure and log path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved -> %s", path)


def _ppg_time_axis():
    """Return time axis in seconds for a single PPG window."""
    return np.linspace(0, INPUT_LENGTH / FS, INPUT_LENGTH)


# =============================================================================
#  1. SHAP -- XGBoost (TreeExplainer)
# =============================================================================

def interpret_xgb_shap(model, X_test, y_test, feature_names, out_dir,
                       n_background=200, n_explain=100):
    """
    SHAP TreeExplainer for XGBoost.

    Produces (SHAP path — XGBoost 2.x):
      - shap_summary_bar.png      : mean |SHAP| per feature (global ranking)
      - shap_beeswarm_<class>.png : per-class beeswarm
      - shap_dependence_<feat>.png: dependence plots for top-3 features

    Produces (native fallback — XGBoost 3.x):
      - xgb_importance_gain.png   : avg gain per feature
      - xgb_importance_weight.png : split count per feature
      - xgb_importance_cover.png  : avg cover per feature
      - xgb_scatter_<feat>.png    : feature value vs predicted probability
      - shap_report.txt           : ranked feature list
    """
    import xgboost as xgb

    logger.info("-- XGBoost Feature Importance --")

    rng     = np.random.RandomState(42)
    exp_idx = rng.choice(len(X_test), size=min(n_explain, len(X_test)), replace=False)
    X_exp   = X_test[exp_idx]
    y_exp   = y_test[exp_idx]

    # ------------------------------------------------------------------ #
    #  Path A: SHAP TreeExplainer (XGBoost 2.x)                          #
    # ------------------------------------------------------------------ #
    shap_array = None
    try:
        import shap
        logger.info("  Trying shap.TreeExplainer ...")
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_exp)
        if isinstance(shap_values, list):
            shap_array = np.stack(shap_values, axis=2)   # (n, feats, classes)
        else:
            shap_array = shap_values
        logger.info("  TreeExplainer succeeded.")
    except Exception as e:
        logger.warning(
            "  SHAP TreeExplainer failed (%s).\n"
            "  XGBoost 3.x is not yet supported by SHAP TreeExplainer.\n"
            "  Using XGBoost native importance (gain/weight/cover) instead.", e
        )

    # ------------------------------------------------------------------ #
    #  Path A continued: render SHAP plots                               #
    # ------------------------------------------------------------------ #
    if shap_array is not None:
        mean_abs  = np.mean(np.abs(shap_array), axis=(0, 2))
        top_k     = min(20, len(feature_names))
        top_idx   = np.argsort(mean_abs)[-top_k:][::-1]
        top_names = [feature_names[i] for i in top_idx]
        top_vals  = mean_abs[top_idx]

        fig, ax = plt.subplots(figsize=(10, 7))
        colors  = plt.cm.viridis(np.linspace(0.2, 0.9, top_k))
        bars    = ax.barh(range(top_k), top_vals[::-1], color=colors[::-1])
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(top_names[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("XGBoost -- Top Feature Importances (SHAP)", fontsize=12)
        ax.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, top_vals[::-1]):
            ax.text(bar.get_width() + 0.0005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=7)
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, "shap_summary_bar.png"))

        for c in np.unique(y_exp):
            cls_name = CLASS_NAMES[c]
            sv_c = shap_array[:, :, c] if shap_array.ndim == 3 else shap_array
            try:
                exp_obj = shap.Explanation(
                    values=sv_c, base_values=np.zeros(len(sv_c)),
                    data=X_exp, feature_names=feature_names,
                )
                shap.plots.beeswarm(exp_obj, max_display=15, show=False)
                plt.title(f"SHAP Beeswarm -- Class: {cls_name}", fontsize=12)
                plt.tight_layout()
                path = os.path.join(out_dir, f"shap_beeswarm_{cls_name.lower()}.png")
                plt.savefig(path, dpi=130, bbox_inches="tight")
                plt.close()
                logger.info("  Saved -> %s", path)
            except Exception as e:
                logger.warning("  Beeswarm for %s failed: %s", cls_name, e)

        for rank, feat_idx in enumerate(top_idx[:3]):
            feat_name = feature_names[feat_idx]
            sv_c0     = shap_array[:, feat_idx, 0]
            fig, ax   = plt.subplots(figsize=(8, 5))
            sc = ax.scatter(X_exp[:, feat_idx], sv_c0, c=y_exp,
                            cmap="tab10", alpha=0.6, s=20)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_ticks(list(range(NUM_CLASSES)))
            cbar.set_ticklabels(CLASS_NAMES)
            ax.set_xlabel(feat_name)
            ax.set_ylabel("SHAP value")
            ax.set_title(f"SHAP Dependence -- {feat_name} (rank #{rank+1})", fontsize=11)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            safe = feat_name.replace("/", "_").replace(" ", "_")
            _save(fig, os.path.join(out_dir, f"shap_dependence_{safe}.png"))

        report_lines = ["XGBoost SHAP Report\n" + "=" * 50 + "\n",
                        "Top Features by Mean |SHAP|:\n"]
        for i, (name, val) in enumerate(zip(top_names, top_vals)):
            report_lines.append(f"  {i+1:2d}. {name:35s}  {val:.6f}\n")
        with open(os.path.join(out_dir, "shap_report.txt"), "w") as f:
            f.writelines(report_lines)
        logger.info("-- XGBoost SHAP done --")
        return   # SHAP path complete — skip native fallback

    # ------------------------------------------------------------------ #
    #  Path B: XGBoost native importance (XGBoost 3.x fallback)          #
    #  gain   = avg information gain when feature used in a split         #
    #  weight = number of times feature appears in trees (split count)    #
    #  cover  = avg number of samples affected by splits on feature       #
    # ------------------------------------------------------------------ #
    logger.info("  Using XGBoost native importance (gain / weight / cover).")

    importance_types = {
        "gain":   "Average Gain per Split\n(how much each feature improves the model)",
        "weight": "Split Count (Weight)\n(how often each feature is used for splitting)",
        "cover":  "Average Cover\n(avg samples routed through feature splits)",
    }
    all_top_names, all_top_vals = None, None

    for imp_type, ylabel in importance_types.items():
        scores_dict = model.get_score(importance_type=imp_type)
        if not scores_dict:
            logger.warning("  No scores for importance_type=%s", imp_type)
            continue

        sorted_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
        top_k  = min(20, len(sorted_items))
        names  = [item[0] for item in sorted_items[:top_k]]
        vals   = np.array([item[1] for item in sorted_items[:top_k]])

        if imp_type == "gain":
            all_top_names, all_top_vals = names, vals

        vals_norm = vals / (vals.max() + 1e-9)
        colors    = plt.cm.plasma(np.linspace(0.15, 0.9, top_k))

        fig, ax = plt.subplots(figsize=(10, 7))
        bars    = ax.barh(range(top_k), vals_norm[::-1], color=colors[::-1])
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(names[::-1], fontsize=9)
        ax.set_xlabel(f"Normalised {imp_type.capitalize()} (0-1)")
        ax.set_title(
            f"XGBoost Feature Importance  [{imp_type.upper()}]\n{ylabel}",
            fontsize=11
        )
        ax.grid(axis="x", alpha=0.3)
        for bar, raw in zip(bars, vals[::-1]):
            ax.text(bar.get_width() + 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{raw:.3f}", va="center", fontsize=7)
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, f"xgb_importance_{imp_type}.png"))

    # Scatter: feature value vs predicted probability (top-3 features)
    if all_top_names:
        dm_exp = xgb.DMatrix(X_exp, feature_names=feature_names)
        probs  = model.predict(dm_exp).reshape(-1, NUM_CLASSES)
        for feat_name in all_top_names[:3]:
            if feat_name not in feature_names:
                continue
            feat_idx = feature_names.index(feat_name)
            fig, ax  = plt.subplots(figsize=(8, 5))
            sc = ax.scatter(X_exp[:, feat_idx], probs[:, 0],
                            c=y_exp, cmap="tab10", alpha=0.5, s=18)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_ticks(list(range(NUM_CLASSES)))
            cbar.set_ticklabels(CLASS_NAMES)
            ax.set_xlabel(feat_name)
            ax.set_ylabel("P(Normal) -- predicted probability")
            ax.set_title(
                f"XGBoost Feature Scatter -- {feat_name}\n(coloured by true class)",
                fontsize=11
            )
            ax.grid(alpha=0.25)
            fig.tight_layout()
            safe = feat_name.replace("/", "_").replace(" ", "_")
            _save(fig, os.path.join(out_dir, f"xgb_scatter_{safe}.png"))

    # Text report
    report_lines = [
        "XGBoost Native Feature Importance Report\n",
        "(SHAP TreeExplainer unavailable -- XGBoost 3.x incompatibility)\n",
        "=" * 55 + "\n",
        "Top Features by Gain (avg information gain per split):\n",
    ]
    if all_top_names:
        for i, (name, val) in enumerate(zip(all_top_names, all_top_vals)):
            report_lines.append(f"  {i+1:2d}. {name:35s}  {val:.4f}\n")
    report_lines += [
        "\nNote: To enable SHAP, downgrade XGBoost:\n",
        "  pip install \"xgboost>=2.0.0,<3.0.0\"\n",
        "  python train_models.py --models xgb\n",
        "  python interpret_models.py --models xgb\n",
    ]
    with open(os.path.join(out_dir, "shap_report.txt"), "w") as f:
        f.writelines(report_lines)
    logger.info("  Report -> %s", os.path.join(out_dir, "shap_report.txt"))
    logger.info("-- XGBoost native importance done --")


# =============================================================================
#  2. SHAP -- DeepExplainer (CNN / BiLSTM / TCN)
# =============================================================================

def interpret_dl_shap(model, X_test, y_test, model_name, out_dir,
                      n_background=100, n_explain=20):
    """
    SHAP DeepExplainer for Keras 1-D sequence models.

    Produces:
      - shap_deep_sample<n>.png  : per-class 1-D SHAP attribution on signal
      - shap_deep_mean_abs.png   : mean |SHAP| across time
    """
    try:
        import shap
        import tensorflow as tf
    except ImportError as e:
        logger.error("Missing dependency for DeepSHAP: %s", e)
        return

    logger.info("-- %s DeepSHAP --", model_name)
    rng    = np.random.RandomState(42)
    bg_idx = rng.choice(len(X_test), size=min(n_background, len(X_test)), replace=False)
    X_bg   = X_test[bg_idx]
    X_sub, y_sub, _ = sample_per_class(X_test, y_test,
                                       n_per_class=max(1, n_explain // NUM_CLASSES))

    try:
        explainer   = shap.DeepExplainer(model, X_bg)
        shap_values = explainer.shap_values(X_sub)
    except Exception as e:
        logger.error("DeepExplainer failed for %s: %s", model_name, e)
        logger.info("  Tip: use --no-deep-shap to skip.")
        return

    if not isinstance(shap_values, list):
        shap_values = [shap_values]

    t = _ppg_time_axis()

    for sample_idx in range(min(5, len(X_sub))):
        fig, axes = plt.subplots(NUM_CLASSES, 1, figsize=(14, 3 * NUM_CLASSES),
                                 sharex=True)
        raw_signal  = X_sub[sample_idx, :, 0]
        true_label  = CLASS_NAMES[y_sub[sample_idx]]

        for c, ax in enumerate(axes):
            shap_1d = shap_values[c][sample_idx, :, 0]
            ax.plot(t, raw_signal, color="#777", linewidth=0.7, alpha=0.7)
            ax.fill_between(t, 0, shap_1d, where=(shap_1d >= 0),
                            interpolate=True, color="#E53935", alpha=0.55,
                            label="Positive")
            ax.fill_between(t, 0, shap_1d, where=(shap_1d < 0),
                            interpolate=True, color="#1E88E5", alpha=0.55,
                            label="Negative")
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.set_ylabel(CLASS_NAMES[c], fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2)
            if c == 0:
                ax.legend(fontsize=7, loc="upper right")

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(
            f"{model_name} -- DeepSHAP Attributions\n"
            f"Sample {sample_idx} | True label: {true_label}",
            fontsize=12, y=1.01
        )
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, f"shap_deep_sample{sample_idx}.png"))

    # Mean |SHAP| across time
    mean_abs_time = np.mean(
        [np.mean(np.abs(sv[:, :, 0]), axis=0) for sv in shap_values], axis=0
    )
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, mean_abs_time, color="#5C6BC0", linewidth=1.2)
    ax.fill_between(t, 0, mean_abs_time, alpha=0.3, color="#5C6BC0")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title(f"{model_name} -- Mean Absolute DeepSHAP by Time Position")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "shap_deep_mean_abs.png"))
    logger.info("-- %s DeepSHAP done --", model_name)


# =============================================================================
#  3. Grad-CAM (1-D CNN / BiLSTM / TCN)
# =============================================================================

def _find_last_conv_layer(model):
    """Find the name of the last Conv1D or DepthwiseConv1D layer."""
    import tensorflow as tf
    conv_layers = []
    for layer in model.layers:
        if isinstance(layer, (
            tf.keras.layers.Conv1D,
            tf.keras.layers.DepthwiseConv1D,
            tf.keras.layers.SeparableConv1D,
        )):
            conv_layers.append(layer.name)
    return conv_layers[-1] if conv_layers else None


def gradcam_1d(model, signal, class_idx, layer_name=None):
    """
    Compute 1-D Grad-CAM activation map for a given class.
    Returns cam (np.ndarray, shape 1250) in range [0, 1].
    """
    import tensorflow as tf

    if layer_name is None:
        layer_name = _find_last_conv_layer(model)
        if layer_name is None:
            raise ValueError("No Conv1D layer found in model for Grad-CAM.")

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    x = tf.cast(signal[np.newaxis], tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        score = preds[:, class_idx]

    grads       = tape.gradient(score, conv_out)              # (1, T', filters)
    pooled_grads = tf.reduce_mean(grads, axis=1, keepdims=True)  # (1, 1, filters)
    weighted    = conv_out * pooled_grads                     # (1, T', filters)
    cam_raw     = tf.reduce_sum(weighted, axis=-1)[0]         # (T',)
    cam_raw     = tf.nn.relu(cam_raw).numpy()
    if cam_raw.max() == 0:
        cam_raw = cam_raw + 1e-9

    cam_up = np.interp(
        np.linspace(0, len(cam_raw) - 1, INPUT_LENGTH),
        np.arange(len(cam_raw)), cam_raw,
    )
    return (cam_up - cam_up.min()) / (cam_up.max() - cam_up.min() + 1e-9)


def interpret_gradcam(model, X_test, y_test, model_name, out_dir, n_per_class=3):
    """
    Generate Grad-CAM heatmaps for all present classes.

    Produces:
      - gradcam_<ClassName>_sample<n>.png
      - gradcam_grid.png
    """
    logger.info("-- %s Grad-CAM --", model_name)
    layer_name = _find_last_conv_layer(model)
    if layer_name is None:
        logger.warning("No Conv layer found -- skipping Grad-CAM for %s", model_name)
        return
    logger.info("  Grad-CAM target layer: %s", layer_name)

    t = _ppg_time_axis()
    present_classes = [c for c in range(NUM_CLASSES) if np.any(y_test == c)]

    for c in present_classes:
        cls_name = CLASS_NAMES[c]
        cls_idx  = np.where(y_test == c)[0]
        chosen   = cls_idx[:n_per_class]

        for i, sample_idx in enumerate(chosen):
            signal = X_test[sample_idx]
            try:
                cam = gradcam_1d(model, signal, class_idx=c, layer_name=layer_name)
            except Exception as e:
                logger.warning("  Grad-CAM failed for class %s sample %d: %s",
                               cls_name, i, e)
                continue

            raw = signal[:, 0]
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

            ax1.plot(t, raw, color="#263238", linewidth=1.0)
            ax1.set_ylabel("PPG Amplitude")
            ax1.set_title(
                f"{model_name} -- Grad-CAM | Class: {cls_name}\n"
                f"Sample index {sample_idx} | Layer: {layer_name}", fontsize=11
            )
            ax1.grid(alpha=0.25)
            extent = [t[0], t[-1], raw.min(), raw.max()]
            ax1.imshow(np.tile(cam, (20, 1)), aspect="auto", extent=extent,
                       cmap="hot", alpha=0.35, vmin=0, vmax=1,
                       origin="lower", zorder=2)

            ax2.fill_between(t, 0, cam, color="#E53935", alpha=0.7)
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel("Activation (Grad-CAM)")
            ax2.set_xlabel("Time (s)")
            ax2.grid(alpha=0.25)

            fig.tight_layout()
            _save(fig, os.path.join(out_dir, f"gradcam_{cls_name}_sample{i}.png"))

    # --- Multi-class grid ---
    n_cols = 2
    n_rows = (len(present_classes) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes_flat  = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, c in enumerate(present_classes):
        ax       = axes_flat[i]
        cls_name = CLASS_NAMES[c]
        cls_idx  = np.where(y_test == c)[0]
        if len(cls_idx) == 0:
            ax.axis("off")
            continue
        signal = X_test[cls_idx[0]]
        try:
            cam = gradcam_1d(model, signal, class_idx=c, layer_name=layer_name)
        except Exception:
            ax.axis("off")
            continue
        raw = signal[:, 0]
        ax.plot(t, raw, color="#37474F", linewidth=0.9, zorder=3)
        ax.fill_between(t, raw.min(), raw.min() + cam * (raw.max() - raw.min()),
                        alpha=0.4, color=CLASS_COLORS[c], zorder=2)
        ax.set_title(cls_name, fontsize=10, color=CLASS_COLORS[c], fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(f"{model_name} -- Grad-CAM Summary (per class)", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "gradcam_grid.png"))
    logger.info("-- %s Grad-CAM done --", model_name)


# =============================================================================
#  4. Integrated Gradients
# =============================================================================

def integrated_gradients_1d(model, signal, class_idx, n_steps=50, baseline=None):
    """
    Integrated Gradients attribution (baseline = flat zero signal).
    Returns attributions (np.ndarray, shape 1250), signed.
    """
    import tensorflow as tf

    if baseline is None:
        baseline = np.zeros_like(signal)

    signal   = signal.astype(np.float32)
    baseline = baseline.astype(np.float32)
    alphas   = np.linspace(0, 1, n_steps, dtype=np.float32)
    interp   = baseline[np.newaxis] + \
               alphas[:, np.newaxis, np.newaxis] * (signal - baseline)[np.newaxis]

    interp_tensor = tf.cast(interp, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(interp_tensor)
        preds  = model(interp_tensor, training=False)
        scores = preds[:, class_idx]

    grads     = tape.gradient(scores, interp_tensor)          # (steps, 1250, 1)
    avg_grads = tf.reduce_mean(grads, axis=0).numpy()         # (1250, 1)
    return ((signal - baseline) * avg_grads)[:, 0]            # (1250,)


def interpret_integrated_gradients(model, X_test, y_test, model_name, out_dir,
                                   n_per_class=2, n_steps=50):
    """
    Compute and plot Integrated Gradients for all classes.

    Produces:
      - intgrad_<ClassName>_sample<n>.png
      - intgrad_grid.png
    """
    logger.info("-- %s Integrated Gradients --", model_name)
    t = _ppg_time_axis()
    present_classes = [c for c in range(NUM_CLASSES) if np.any(y_test == c)]
    grid_data = []

    for c in present_classes:
        cls_name = CLASS_NAMES[c]
        cls_idx  = np.where(y_test == c)[0]
        chosen   = cls_idx[:n_per_class]

        for i, sample_idx in enumerate(chosen):
            signal = X_test[sample_idx]
            try:
                attr = integrated_gradients_1d(model, signal, class_idx=c,
                                               n_steps=n_steps)
            except Exception as e:
                logger.warning("  IG failed for class %s sample %d: %s",
                               cls_name, i, e)
                continue

            raw = signal[:, 0]
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(14, 6), sharex=True,
                gridspec_kw={"height_ratios": [2, 1]}
            )
            ax1.plot(t, raw, color="#1A237E", linewidth=1.0,
                     label="PPG signal", zorder=3)
            ax1.fill_between(t, 0, attr, where=(attr >= 0),
                             interpolate=True, color="#E53935",
                             alpha=0.55, label="+ attribution", zorder=2)
            ax1.fill_between(t, 0, attr, where=(attr < 0),
                             interpolate=True, color="#1565C0",
                             alpha=0.55, label="- attribution", zorder=2)
            ax1.set_ylabel("Amplitude / Attribution")
            ax1.legend(fontsize=8, loc="upper right")
            ax1.set_title(
                f"{model_name} -- Integrated Gradients\n"
                f"Class: {cls_name} | Sample {sample_idx}", fontsize=11
            )
            ax1.grid(alpha=0.25)

            ax2.fill_between(t, 0, np.abs(attr), color="#F57F17", alpha=0.75)
            ax2.set_ylabel("|Attribution|")
            ax2.set_xlabel("Time (s)")
            ax2.grid(alpha=0.25)

            fig.tight_layout()
            _save(fig, os.path.join(out_dir, f"intgrad_{cls_name}_sample{i}.png"))

            if i == 0:
                grid_data.append((cls_name, raw, attr, CLASS_COLORS[c]))

    if grid_data:
        n = len(grid_data)
        fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, (cls_name, raw, attr, color) in zip(axes, grid_data):
            ax.plot(t, raw, color="#263238", linewidth=0.8, alpha=0.8, zorder=3)
            ax.fill_between(t, 0, attr, where=(attr >= 0), interpolate=True,
                            color="#E53935", alpha=0.5, zorder=2)
            ax.fill_between(t, 0, attr, where=(attr < 0), interpolate=True,
                            color="#1565C0", alpha=0.5, zorder=2)
            ax.set_ylabel(cls_name, fontsize=9, color=color, fontweight="bold")
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=7)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"{model_name} -- Integrated Gradients per Class",
                     fontsize=13, y=1.01)
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, "intgrad_grid.png"))

    logger.info("-- %s Integrated Gradients done --", model_name)


# =============================================================================
#  5. LIME -- Local Interpretable Model-agnostic Explanations
# =============================================================================

def interpret_lime(model, X_test, y_test, model_name, out_dir,
                   n_per_class=2, n_segments=25, n_perturbations=1000):
    """
    LIME for 1-D time-series.

    Segments the PPG window into n_segments equal parts. Each perturbation
    zeros a random subset of segments. A Ridge regression gives a coefficient
    per segment (importance).

    Produces:
      - lime_<ClassName>_sample<n>.png
    """
    logger.info("-- %s LIME --", model_name)
    t       = _ppg_time_axis()
    seg_len = INPUT_LENGTH // n_segments
    present = [c for c in range(NUM_CLASSES) if np.any(y_test == c)]
    rng     = np.random.RandomState(42)

    def _apply_mask(signal_1d, mask):
        out = signal_1d.copy()
        for s_i, active in enumerate(mask):
            if not active:
                start = s_i * seg_len
                end   = start + seg_len if s_i < n_segments - 1 else INPUT_LENGTH
                out[start:end] = 0.0
        return out

    def _predict_batch(signals):
        X_batch = signals[:, :, np.newaxis].astype(np.float32)
        return model.predict(X_batch, verbose=0, batch_size=64)

    for c in present:
        cls_name = CLASS_NAMES[c]
        cls_idx  = np.where(y_test == c)[0]
        chosen   = cls_idx[:n_per_class]

        for i, sample_idx in enumerate(chosen):
            signal = X_test[sample_idx, :, 0]

            masks     = rng.randint(0, 2, size=(n_perturbations, n_segments))
            masks[0]  = 1
            perturbed = np.stack([_apply_mask(signal, m) for m in masks], axis=0)

            try:
                probs  = _predict_batch(perturbed)
            except Exception as e:
                logger.warning("  LIME predict failed for %s sample %d: %s",
                               cls_name, i, e)
                continue

            y_lime = probs[:, c]

            try:
                from sklearn.linear_model import Ridge
                from sklearn.preprocessing import StandardScaler
                scaler      = StandardScaler()
                masks_sc    = scaler.fit_transform(masks.astype(np.float32))
                ridge       = Ridge(alpha=1.0)
                ridge.fit(masks_sc, y_lime)
                coefs       = ridge.coef_
            except Exception as e:
                logger.warning("  LIME ridge fit failed: %s", e)
                continue

            cmap_pos = plt.cm.Reds
            cmap_neg = plt.cm.Blues
            vmax     = np.max(np.abs(coefs)) + 1e-9

            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(14, 7),
                gridspec_kw={"height_ratios": [2, 1]}, sharex=False
            )
            for s_i in range(n_segments):
                start   = s_i * seg_len
                end     = start + seg_len if s_i < n_segments - 1 else INPUT_LENGTH
                t_seg   = t[start:end]
                sig_seg = signal[start:end]
                coef    = coefs[s_i]
                color   = cmap_pos(0.3 + 0.7 * coef / vmax) if coef >= 0 \
                          else cmap_neg(0.3 + 0.7 * (-coef) / vmax)
                ax1.fill_between(t_seg, sig_seg.min() - 0.02, sig_seg.max() + 0.02,
                                 color=color, alpha=0.35, zorder=1)

            ax1.plot(t, signal, color="#1A237E", linewidth=0.9, zorder=3)
            ax1.set_ylabel("PPG Amplitude")
            ax1.set_title(
                f"{model_name} -- LIME Explanation\n"
                f"Class: {cls_name} | Sample {sample_idx} | "
                f"{n_segments} segments, {n_perturbations} perturbations",
                fontsize=11
            )
            ax1.grid(alpha=0.2)

            x_pos    = np.arange(n_segments)
            bar_cols = [
                cmap_pos(0.4 + 0.6 * max(0, cv) / vmax) if cv >= 0
                else cmap_neg(0.4 + 0.6 * max(0, -cv) / vmax)
                for cv in coefs
            ]
            ax2.bar(x_pos, coefs, color=bar_cols, edgecolor="gray", linewidth=0.4)
            ax2.axhline(0, color="black", linewidth=0.7)
            ax2.set_xlabel("Segment index")
            ax2.set_ylabel("LIME coefficient")
            ax2.set_xticks(x_pos)
            ax2.tick_params(axis="x", labelsize=6)
            ax2.grid(axis="y", alpha=0.3)

            fig.tight_layout()
            _save(fig, os.path.join(out_dir, f"lime_{cls_name}_sample{i}.png"))

    logger.info("-- %s LIME done --", model_name)


# =============================================================================
#  6. XGBoost LIME (tabular feature vectors)
# =============================================================================

def interpret_xgb_lime(model, X_test, y_test, feature_names, out_dir,
                       n_per_class=2, n_perturbations=2000):
    """
    LIME for XGBoost on tabular feature vectors.
    Perturbs features with Gaussian noise and fits a local Ridge regression.
    """
    logger.info("-- XGBoost LIME (tabular) --")
    import xgboost as xgb
    from sklearn.linear_model import Ridge

    present = [c for c in range(NUM_CLASSES) if np.any(y_test == c)]
    rng     = np.random.RandomState(42)

    for c in present:
        cls_name = CLASS_NAMES[c]
        cls_idx  = np.where(y_test == c)[0]
        chosen   = cls_idx[:n_per_class]

        for i, sample_idx in enumerate(chosen):
            x0      = X_test[sample_idx]
            std     = X_test.std(axis=0) + 1e-9
            noise   = rng.randn(n_perturbations, X_test.shape[1]) * std
            perturb = x0[np.newaxis] + noise
            perturb[0] = x0

            dm    = xgb.DMatrix(perturb, feature_names=feature_names)

            # model was trained with multi:softmax → returns class indices (int),
            # NOT probabilities.  Re-predict with multi:softprob to get P(class).
            raw_pred = model.predict(dm)   # shape: (n_perturbations,) class labels

            # Check if output is flat class indices (softmax) or probabilities (softprob)
            if raw_pred.ndim == 1 and raw_pred.dtype in (np.int32, np.float32) \
                    and raw_pred.max() <= NUM_CLASSES:
                # Likely class labels from multi:softmax — switch to softprob
                model.set_param("objective", "multi:softprob")
                model.set_param("num_class", str(NUM_CLASSES))
                probs = model.predict(dm).reshape(-1, NUM_CLASSES)  # (n, classes)
                # Restore original objective
                model.set_param("objective", "multi:softmax")
            else:
                probs = raw_pred.reshape(-1, NUM_CLASSES)

            y_lime = probs[:, c]   # (n_perturbations,) — probability of class c

            ridge = Ridge(alpha=1.0)
            ridge.fit(noise / std, y_lime)
            coefs = ridge.coef_

            top_k        = 15
            top_idx      = np.argsort(np.abs(coefs))[-top_k:][::-1]
            top_coefs    = coefs[top_idx]
            top_names_lc = [feature_names[j] for j in top_idx]

            fig, ax = plt.subplots(figsize=(9, 6))
            colors  = ["#E53935" if v >= 0 else "#1565C0" for v in top_coefs]
            ax.barh(range(top_k), top_coefs[::-1], color=colors[::-1])
            ax.set_yticks(range(top_k))
            ax.set_yticklabels(top_names_lc[::-1], fontsize=9)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("LIME coefficient (signed)")
            ax.set_title(
                f"XGBoost LIME -- Class: {cls_name}\n"
                f"Sample {sample_idx} | Top {top_k} features",
                fontsize=11
            )
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout()
            _save(fig, os.path.join(out_dir, f"lime_xgb_{cls_name}_sample{i}.png"))

    logger.info("-- XGBoost LIME done --")


# =============================================================================
#  7. INTERPRETATION REPORT
# =============================================================================

def write_interpretation_report(model_name, out_dir, start_time):
    """Write a human-readable summary report of all generated outputs."""
    outputs = sorted(Path(out_dir).glob("*.png")) + \
              sorted(Path(out_dir).glob("*.txt"))
    elapsed = time.time() - start_time

    lines = [
        f"Model Interpretation Report -- {model_name}\n",
        "=" * 60 + "\n",
        f"Generated at : {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"Elapsed time : {elapsed:.1f} s\n",
        f"Output dir   : {out_dir}\n\n",
        "Classes\n-------\n",
        *[f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)],
        "\nGenerated Files\n---------------\n",
        *[f"  {p.name}\n" for p in outputs],
        "\nInterpretation Methods\n----------------------\n",
        "  SHAP (TreeExplainer / DeepExplainer)\n",
        "    Global feature ranking, beeswarm plots, dependence plots.\n",
        "  Grad-CAM\n",
        "    Highlights PPG time regions driving each class prediction.\n",
        "  Integrated Gradients\n",
        "    Signed attribution map; baseline = flat-zero PPG signal.\n",
        "  LIME\n",
        "    25-segment perturbation; local Ridge proxy model.\n",
    ]

    report_path = os.path.join(out_dir, "interpretation_report.txt")
    with open(report_path, "w") as f:
        f.writelines(lines)
    logger.info("Report written -> %s", report_path)


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Model Interpretation for PPG Arrhythmia Detection"
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=["xgb", "cnn", "bilstm", "tcn"],
        default=["xgb", "cnn", "bilstm", "tcn"],
        help="Which models to interpret (default: all)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=10,
        help="Approx. number of test samples to explain per method (default: 10)",
    )
    parser.add_argument(
        "--classes", nargs="+", choices=CLASS_NAMES, default=None,
        help="Restrict to specific class names",
    )
    parser.add_argument("--no-deep-shap", action="store_true",
                        help="Skip DeepSHAP (slow / high memory)")
    parser.add_argument("--no-gradcam",   action="store_true",
                        help="Skip Grad-CAM")
    parser.add_argument("--no-intgrad",   action="store_true",
                        help="Skip Integrated Gradients")
    parser.add_argument("--no-lime",      action="store_true",
                        help="Skip LIME")
    parser.add_argument("--ig-steps",     type=int, default=50,
                        help="Interpolation steps for Integrated Gradients")
    parser.add_argument("--no-gpu",       action="store_true",
                        help="Force CPU-only")
    args = parser.parse_args()

    # -- GPU / TF setup -------------------------------------------------------
    dl_args = ["cnn", "bilstm", "tcn"]
    if any(m in args.models for m in dl_args):
        if args.no_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            logger.info("GPU disabled -- running on CPU.")
        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            logger.info("TensorFlow %s | GPUs: %s",
                        tf.__version__,
                        tf.config.list_physical_devices("GPU"))
        except ImportError:
            logger.error("TensorFlow not installed.")
            sys.exit(1)

    t_start     = time.time()
    n_per_class = max(1, args.n_samples // NUM_CLASSES)

    class_filter = None
    if args.classes:
        class_filter = [CLASS_NAMES.index(c) for c in args.classes
                        if c in CLASS_NAMES]
        logger.info("Restricting to classes: %s", args.classes)

    # =========================================================================
    #  XGBoost
    # =========================================================================
    if "xgb" in args.models:
        logger.info("\n%s\nInterpreting: XGBoost\n%s", "=" * 60, "=" * 60)
        model_xgb = load_xgb_model()
        feat_ok   = True
        if model_xgb is not None:
            try:
                X_feat, y_feat, feat_names = load_feature_test()
            except FileNotFoundError as e:
                logger.error("Cannot load XGBoost features: %s", e)
                feat_ok = False

        if model_xgb is not None and feat_ok:
            out_xgb = os.path.join(OUTPUT_DIR, "xgb", "interpretation")
            os.makedirs(out_xgb, exist_ok=True)

            interpret_xgb_shap(model_xgb, X_feat, y_feat, feat_names, out_xgb,
                               n_explain=min(200, len(X_feat)))

            if not args.no_lime:
                try:
                    interpret_xgb_lime(model_xgb, X_feat, y_feat, feat_names,
                                       out_xgb, n_per_class=n_per_class)
                except Exception as e:
                    logger.error("XGBoost LIME failed: %s", e)

            write_interpretation_report("XGBoost", out_xgb, t_start)

    # =========================================================================
    #  Deep Learning models
    # =========================================================================
    dl_map = {"cnn": "edge_cnn", "bilstm": "bilstm", "tcn": "tcn"}
    dl_requested = [m for m in args.models if m in dl_map]

    X_raw, y_raw = None, None
    if dl_requested:
        try:
            X_raw, y_raw = load_raw_test()
        except FileNotFoundError as e:
            logger.error("Cannot load raw dataset: %s", e)

    for arg_name in dl_requested:
        model_name = dl_map[arg_name]
        logger.info("\n%s\nInterpreting: %s\n%s", "=" * 60, model_name, "=" * 60)

        if X_raw is None:
            logger.error("Skipping %s -- no raw dataset.", model_name)
            continue

        import tensorflow as tf
        model = load_keras_model(model_name)
        if model is None:
            logger.warning("No saved model for %s -- skipping.", model_name)
            continue

        out_dl = os.path.join(OUTPUT_DIR, model_name, "interpretation")
        os.makedirs(out_dl, exist_ok=True)

        # Optionally restrict to requested classes
        if class_filter is not None:
            mask  = np.isin(y_raw, class_filter)
            X_sub = X_raw[mask]
            y_sub = y_raw[mask]
        else:
            X_sub, y_sub = X_raw, y_raw

        # Limit total samples
        max_samples = n_per_class * NUM_CLASSES * 4
        if len(X_sub) > max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_sub), max_samples, replace=False)
            X_sub = X_sub[idx]
            y_sub = y_sub[idx]

        # DeepSHAP
        if not args.no_deep_shap:
            try:
                interpret_dl_shap(model, X_sub, y_sub, model_name, out_dl,
                                  n_background=min(100, len(X_sub)),
                                  n_explain=args.n_samples)
            except Exception as e:
                logger.error("DeepSHAP failed for %s: %s", model_name, e)
                logger.info("  Tip: use --no-deep-shap to skip.")

        # Grad-CAM
        if not args.no_gradcam:
            try:
                interpret_gradcam(model, X_sub, y_sub, model_name, out_dl,
                                  n_per_class=n_per_class)
            except Exception as e:
                logger.error("Grad-CAM failed for %s: %s", model_name, e)

        # Integrated Gradients
        if not args.no_intgrad:
            try:
                interpret_integrated_gradients(
                    model, X_sub, y_sub, model_name, out_dl,
                    n_per_class=n_per_class, n_steps=args.ig_steps
                )
            except Exception as e:
                logger.error("Integrated Gradients failed for %s: %s", model_name, e)

        # LIME
        if not args.no_lime:
            try:
                interpret_lime(model, X_sub, y_sub, model_name, out_dl,
                               n_per_class=n_per_class)
            except Exception as e:
                logger.error("LIME failed for %s: %s", model_name, e)

        write_interpretation_report(model_name, out_dl, t_start)

    total = time.time() - t_start
    logger.info("\nAll interpretation complete in %.1f s.", total)
    logger.info("Results saved under: %s/<model>/interpretation/", OUTPUT_DIR)


if __name__ == "__main__":
    main()
