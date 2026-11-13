"""
benchmark.py
============
Unified evaluation and benchmarking script for all trained PPG arrhythmia
detection models.

Produces a side-by-side comparison table across all models covering:
  - Clinical metrics  : Precision, Recall, Macro F1, per-class Confusion Matrix
  - Edge feasibility  : Parameter count, model size (KB), MACs/FLOPs estimate

Usage
-----
  python benchmark.py                    # benchmark all 4 models
  python benchmark.py --models xgb cnn  # subset
  python benchmark.py --no-gpu          # force CPU inference
  python benchmark.py --output-dir outputs/benchmark_results

Outputs
-------
  outputs/benchmark_results/
    benchmark_table.txt       -- side-by-side ASCII/CSV table
    benchmark_table.csv       -- machine-readable CSV
    confusion_matrix_<name>.png  -- per-model confusion matrices
    per_class_f1_<name>.png      -- per-class F1 bar charts
    benchmark_radar.png       -- radar chart comparing all models
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Paths & constants ----------------------------------------------------------
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "outputs")
DATASET_PATH = os.path.join(OUTPUT_DIR, "dataset.npz")
FEATURES_TEST = os.path.join(OUTPUT_DIR, "features_test.parquet")

NUM_CLASSES = 5
INPUT_LENGTH = 1250  # 10 s @ 125 Hz

CLASS_NAMES = ["Normal", "AFib", "Brady", "Tachy", "Other"]

# Import the storage helper (supports .parquet with pyarrow OR .csv fallback)
try:
    from feature_extraction import load_features as _load_feat_file
except ImportError:
    def _load_feat_file(path: str) -> pd.DataFrame:
        csv = path.replace(".parquet", ".csv")
        if os.path.isfile(path):
            return pd.read_parquet(path)
        if os.path.isfile(csv):
            return pd.read_csv(csv)
        raise FileNotFoundError(f"Feature file not found: {path}")

# Model paths ------------------------------------------------------------------
MODEL_PATHS = {
    "xgb":      os.path.join(OUTPUT_DIR, "xgb",      "xgb_model.ubj"),
    "edge_cnn": os.path.join(OUTPUT_DIR, "edge_cnn",  "best_edge_cnn.h5"),
    "bilstm":   os.path.join(OUTPUT_DIR, "bilstm",    "best_bilstm.h5"),
    "tcn":      os.path.join(OUTPUT_DIR, "tcn",       "best_tcn.h5"),
}

# Friendly display names
DISPLAY_NAMES = {
    "xgb":      "XGBoost",
    "edge_cnn": "Edge CNN (DS-Conv)",
    "bilstm":   "Hybrid Conv+BiLSTM",
    "tcn":      "TCN (Dilated Conv)",
}


# ==============================================================================
#  LOADER HELPERS
# ==============================================================================

def _load_test_raw() -> tuple[np.ndarray, np.ndarray]:
    """Return (X_test, y_test) raw PPG windows from dataset.npz."""
    data   = np.load(DATASET_PATH, allow_pickle=True)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int32)
    logger.info("Test set (raw): %s", X_test.shape)
    return X_test, y_test


def _load_test_features() -> tuple[np.ndarray, np.ndarray]:
    """Return (X_test, y_test) feature vectors, supporting .parquet and .csv."""
    df     = _load_feat_file(FEATURES_TEST)
    y_test = df["label"].values.astype(np.int32)
    X_test = df.drop(columns=["label"]).values.astype(np.float32)
    logger.info("Test set (features): %s", X_test.shape)
    return X_test, y_test


# ==============================================================================
#  MAC / FLOP ESTIMATION
# ==============================================================================

def estimate_macs_keras(model: Any) -> int:
    """
    Estimate total multiply-accumulate operations (MACs) for a Keras model.

    Uses a simple layer-by-layer formula:
      Conv1D     : MACs = out_length * filters * kernel_size * in_channels
      DepthwiseC : MACs = out_length * in_channels * kernel_size
      Dense      : MACs = in_features * out_features
      LSTM       : MACs = 4 * T * (H * H + H * I)   (4 gates, H hidden, I input)
    This is an approximation; exact count requires profiling tools like
    tf.lite.experimental.Analyzer or tf-profile.
    """
    total_macs = 0
    for layer in model.layers:
        cfg = layer.get_config()
        name_lower = layer.__class__.__name__.lower()

        try:
            out_shape = layer.output_shape
        except AttributeError:
            continue

        if "conv1d" in name_lower and "depthwise" not in name_lower \
                and "separable" not in name_lower:
            # Standard Conv1D
            if isinstance(out_shape, list):
                out_shape = out_shape[0]
            L_out    = out_shape[1] if len(out_shape) >= 2 else 1
            filters  = cfg.get("filters", 1)
            ks       = cfg.get("kernel_size", [1])[0] \
                       if isinstance(cfg.get("kernel_size"), (list, tuple)) \
                       else cfg.get("kernel_size", 1)
            in_ch    = getattr(layer.input_spec, "shape", [None, None, 1])
            if hasattr(layer, "kernel") and layer.kernel is not None:
                in_ch = layer.kernel.shape[1]
            else:
                in_ch = 1
            total_macs += int(L_out) * int(filters) * int(ks) * int(in_ch)

        elif "depthwiseconv1d" in name_lower:
            if isinstance(out_shape, list):
                out_shape = out_shape[0]
            L_out = out_shape[1] if len(out_shape) >= 2 else 1
            ks    = cfg.get("kernel_size", [1])[0] \
                    if isinstance(cfg.get("kernel_size"), (list, tuple)) \
                    else cfg.get("kernel_size", 1)
            if hasattr(layer, "depthwise_kernel") and layer.depthwise_kernel is not None:
                in_ch = layer.depthwise_kernel.shape[1]
            else:
                in_ch = 1
            total_macs += int(L_out) * int(in_ch) * int(ks)

        elif "separableconv1d" in name_lower:
            # Depthwise + pointwise passes
            if isinstance(out_shape, list):
                out_shape = out_shape[0]
            L_out   = out_shape[1] if len(out_shape) >= 2 else 1
            filters = cfg.get("filters", 1)
            ks      = cfg.get("kernel_size", [1])[0] \
                      if isinstance(cfg.get("kernel_size"), (list, tuple)) \
                      else cfg.get("kernel_size", 1)
            if hasattr(layer, "depthwise_kernel") and layer.depthwise_kernel is not None:
                in_ch = layer.depthwise_kernel.shape[1]
            else:
                in_ch = 1
            # Depthwise pass
            total_macs += int(L_out) * int(in_ch) * int(ks)
            # Pointwise pass
            total_macs += int(L_out) * int(filters) * int(in_ch)

        elif "dense" in name_lower:
            if isinstance(out_shape, list):
                out_shape = out_shape[0]
            out_units = out_shape[-1] if len(out_shape) >= 1 else 1
            if hasattr(layer, "kernel") and layer.kernel is not None:
                in_units = layer.kernel.shape[0]
            else:
                in_units = 1
            total_macs += int(in_units) * int(out_units)

        elif "lstm" in name_lower:
            # 4 gates, each has (H*H + H*I) MACs
            # LSTM.units = H,  input_dim = I
            H = cfg.get("units", 32)
            if hasattr(layer, "kernel") and layer.kernel is not None:
                I = layer.kernel.shape[0]
            else:
                I = 1
            # T = sequence length at input to this LSTM
            try:
                in_shape = layer.input_shape
                if isinstance(in_shape, list):
                    in_shape = in_shape[0]
                T = in_shape[1] if len(in_shape) >= 2 else 1
            except Exception:
                T = INPUT_LENGTH
            total_macs += 4 * int(T) * (int(H) * int(H) + int(H) * int(I))

        # Bidirectional wrapper adds 2x
        elif "bidirectional" in name_lower:
            H = cfg.get("merge_mode", "concat")
            # The wrapped LSTM MACs are counted above inside the sub-layers

    return total_macs


# ==============================================================================
#  PER-MODEL INFERENCE + METRICS
# ==============================================================================

def benchmark_xgb(
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_dir: str,
) -> dict:
    """Benchmark XGBoost: load, infer, compute metrics."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("xgboost not installed -- skipping XGBoost benchmark")
        return {}

    model_path = MODEL_PATHS["xgb"]
    if not os.path.isfile(model_path):
        logger.warning("XGBoost model not found at %s -- skipping", model_path)
        return {}

    model = xgb.Booster()
    model.load_model(model_path)
    model_size_kb = os.path.getsize(model_path) / 1024

    dtest = xgb.DMatrix(X_test)

    t0 = time.perf_counter()
    y_pred = model.predict(dtest).astype(np.int32)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Approximate: # trees * # leaves * 12 bytes per leaf (rough)
    # XGBoost num_trees accessible via model.num_boosted_rounds()
    try:
        num_trees = model.num_boosted_rounds()
    except Exception:
        num_trees = -1

    metrics = _compute_metrics(y_test, y_pred)
    metrics.update({
        "name":            DISPLAY_NAMES["xgb"],
        "params":          "N/A (trees)",
        "model_size_kb":   f"{model_size_kb:.1f}",
        "macs":            "N/A",
        "latency_ms_total": f"{elapsed_ms:.1f}",
        "latency_ms_sample": f"{elapsed_ms / len(X_test) * 1000:.2f}",  # ms / sample
        "num_trees":       num_trees,
    })
    _plot_confusion_matrix(y_test, y_pred, "XGBoost", out_dir)
    return metrics


def benchmark_keras_model(
    key: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_dir: str,
) -> dict:
    """Benchmark a Keras model (CNN / BiLSTM / TCN): load, infer, compute metrics."""
    import tensorflow as tf

    model_path = MODEL_PATHS[key]
    if not os.path.isfile(model_path):
        logger.warning("Model not found at %s -- skipping", model_path)
        return {}

    logger.info("Loading %s from %s ...", key, model_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    model_size_kb = os.path.getsize(model_path) / 1024

    total_params = model.count_params()

    # Estimate MACs
    macs = estimate_macs_keras(model)

    # Batch inference timing
    t0 = time.perf_counter()
    y_prob = model.predict(X_test, batch_size=256, verbose=0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    y_pred = np.argmax(y_prob, axis=1).astype(np.int32)

    metrics = _compute_metrics(y_test, y_pred)
    metrics.update({
        "name":            DISPLAY_NAMES[key],
        "params":          f"{total_params:,}",
        "model_size_kb":   f"{model_size_kb:.1f}",
        "macs":            f"{macs:,}",
        "latency_ms_total":  f"{elapsed_ms:.1f}",
        "latency_ms_sample": f"{elapsed_ms / len(X_test) * 1000:.2f}",
        "num_trees":       "N/A",
    })
    _plot_confusion_matrix(y_test, y_pred, DISPLAY_NAMES[key], out_dir)
    return metrics


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute classification metrics."""
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, accuracy_score,
        classification_report, confusion_matrix,
    )
    return {
        "accuracy":    f"{accuracy_score(y_true, y_pred):.4f}",
        "precision":   f"{precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}",
        "recall":      f"{recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}",
        "macro_f1":    f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}",
        "f1_per_class": f1_score(y_true, y_pred, average=None,
                                 labels=list(range(NUM_CLASSES)), zero_division=0),
        "y_pred":       y_pred,
    }


# ==============================================================================
#  PLOTS
# ==============================================================================

def _plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, name: str, out_dir: str
) -> None:
    """Save a normalised confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)),
                          normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, linewidths=0.4, cbar_kws={"label": "Recall"},
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (normalised)\n{name}", fontsize=11)
    plt.tight_layout()
    safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_{safe}.png"), dpi=150)
    plt.close()


def _plot_per_class_f1(results: list[dict], out_dir: str) -> None:
    """Bar chart comparing per-class F1 across all models."""
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(4 * NUM_CLASSES, 4),
                             sharey=True)
    cmap = plt.get_cmap("tab10")
    for cls_i, (cls_name, ax) in enumerate(zip(CLASS_NAMES, axes)):
        for m_i, res in enumerate(results):
            f1_arr = res.get("f1_per_class", [])
            if len(f1_arr) > cls_i:
                ax.bar(res["name"], f1_arr[cls_i],
                       color=cmap(m_i), alpha=0.85, width=0.4)
        ax.set_title(cls_name, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.set_ylabel("F1-Score" if cls_i == 0 else "")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-Class F1-Score by Model", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_class_f1.png"), dpi=150)
    plt.close()


def _plot_radar_chart(results: list[dict], out_dir: str) -> None:
    """
    Radar (spider) chart with 5 axes:
      Macro F1, Precision, Recall, Accuracy, Edge-Score
    Edge-Score = 1 - normalised model size (larger = more edge-friendly).
    """
    metrics_keys = ["macro_f1", "precision", "recall", "accuracy"]
    labels       = ["Macro F1", "Precision", "Recall", "Accuracy", "Edge Score"]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Compute edge score from model sizes
    sizes = []
    for r in results:
        try:
            sz = float(r.get("model_size_kb", "9999").replace(",", ""))
        except (ValueError, AttributeError):
            sz = 9999.0
        sizes.append(sz)
    max_sz = max(sizes) + 1e-3
    edge_scores = [1.0 - (s / max_sz) for s in sizes]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    cmap = plt.get_cmap("tab10")

    for i, (res, edge_sc) in enumerate(zip(results, edge_scores)):
        vals = [float(res.get(k, 0)) for k in metrics_keys] + [edge_sc]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, color=cmap(i), label=res["name"])
        ax.fill(angles, vals, alpha=0.08, color=cmap(i))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=7)
    ax.set_title("Model Comparison Radar Chart", y=1.12, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "benchmark_radar.png"), dpi=150)
    plt.close()
    logger.info("Radar chart saved -> %s", os.path.join(out_dir, "benchmark_radar.png"))


# ==============================================================================
#  TABLE FORMATTING
# ==============================================================================

def _print_benchmark_table(results: list[dict], out_dir: str) -> None:
    """Print and save a formatted ASCII comparison table + CSV."""
    # ── Column headers ─────────────────────────────────────────────────────────
    col_order = [
        "name", "params", "model_size_kb",
        "macs", "accuracy", "precision", "recall", "macro_f1",
        "latency_ms_sample",
    ]
    col_labels = [
        "Model", "Parameters", "Size (KB)",
        "MACs", "Accuracy", "Precision", "Recall", "Macro F1",
        "Latency (ms/sample)",
    ]

    rows = []
    for r in results:
        row = {k: str(r.get(k, "N/A")) for k in col_order}
        rows.append(row)

    # ── CSV ───────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows, columns=col_order)
    df.columns = col_labels
    csv_path = os.path.join(out_dir, "benchmark_table.csv")
    df.to_csv(csv_path, index=False)
    logger.info("CSV saved -> %s", csv_path)

    # ── ASCII table ───────────────────────────────────────────────────────────
    col_widths = {
        "Model":              20,
        "Parameters":         14,
        "Size (KB)":          10,
        "MACs":               16,
        "Accuracy":           10,
        "Precision":          10,
        "Recall":             10,
        "Macro F1":           10,
        "Latency (ms/sample)": 22,
    }

    def _fmt_row(vals: list[str]) -> str:
        return "| " + " | ".join(
            f"{v:<{col_widths[c]}}" for c, v in zip(col_labels, vals)
        ) + " |"

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths.values()) + "+"
    header = _fmt_row(col_labels)

    lines = [
        "",
        "=" * 120,
        "  PPG Arrhythmia Detection -- Model Benchmark Summary",
        "=" * 120,
        sep, header, sep,
    ]
    for row in rows:
        lines.append(_fmt_row([row[k] for k in col_order]))
    lines += [sep, ""]

    table_str = "\n".join(lines)
    print(table_str)

    txt_path = os.path.join(out_dir, "benchmark_table.txt")
    Path(txt_path).write_text(table_str)
    logger.info("ASCII table saved -> %s", txt_path)

    # ── Per-class F1 detail ---------------------------------------------------
    print("\n  Per-Class F1-Score Detail\n  " + "-" * 60)
    header2 = f"  {'Model':<22}" + "".join(f"  {n:<12}" for n in CLASS_NAMES)
    print(header2)
    for res in results:
        f1s = res.get("f1_per_class", [0.0] * NUM_CLASSES)
        row_str = f"  {res['name']:<22}" + "".join(f"  {v:.3f}       " for v in f1s)
        print(row_str)
    print()


# ==============================================================================
#  MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark all PPG arrhythmia detection models"
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=["xgb", "edge_cnn", "bilstm", "tcn"],
        default=["xgb", "edge_cnn", "bilstm", "tcn"],
        help="Models to benchmark (default: all)",
    )
    parser.add_argument("--no-gpu",     action="store_true")
    parser.add_argument(
        "--output-dir", default=os.path.join(OUTPUT_DIR, "benchmark_results"),
        help="Directory to save benchmark outputs",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── GPU setup (for Keras inference) ---------------------------------------
    dl_requested = [m for m in args.models if m != "xgb"]
    if dl_requested:
        from gpu_config import setup_gpu
        setup_gpu(mixed_precision=False, no_gpu=args.no_gpu)

    # ── Load test data ─────────────────────────────────────────────────────────
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Run: python preprocessing/fuse_datasets.py first."
        )
    X_test_raw, y_test = _load_test_raw()

    X_test_feats = None
    if "xgb" in args.models:
        if os.path.isfile(FEATURES_TEST):
            X_test_feats, _ = _load_test_features()
        else:
            logger.warning(
                "Feature test file not found (%s). "
                "Run feature_extraction.py first to benchmark XGBoost.",
                FEATURES_TEST,
            )

    # ── Run benchmarks ─────────────────────────────────────────────────────────
    results: list[dict] = []

    if "xgb" in args.models and X_test_feats is not None:
        logger.info("Benchmarking XGBoost ...")
        r = benchmark_xgb(X_test_feats, y_test, args.output_dir)
        if r:
            results.append(r)

    for key in ["edge_cnn", "bilstm", "tcn"]:
        if key in args.models:
            logger.info("Benchmarking %s ...", DISPLAY_NAMES[key])
            r = benchmark_keras_model(key, X_test_raw, y_test, args.output_dir)
            if r:
                results.append(r)

    if not results:
        logger.error("No models were successfully benchmarked. "
                     "Run train_models.py first to generate saved models.")
        return

    # ── Tables & plots ─────────────────────────────────────────────────────────
    _print_benchmark_table(results, args.output_dir)
    _plot_per_class_f1(results, args.output_dir)
    if len(results) >= 2:
        _plot_radar_chart(results, args.output_dir)

    logger.info("\nBenchmark complete. Outputs -> %s", args.output_dir)


if __name__ == "__main__":
    main()
