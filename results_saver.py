"""
results_saver.py
================
Comprehensive artifact saver for every model training run.

Saves ALL of the following per model into  outputs/<model_name>/results/ :

  Training history
  ----------------
  history.csv            -- epoch-level loss / accuracy / val_macro_f1
  history.json           -- same, machine-readable
  plot_history.png       -- loss + accuracy + val_macro_f1 curves

  Classification metrics
  ----------------------
  classification_report.txt   -- sklearn full-text report
  classification_report.csv   -- per-class precision/recall/f1 as CSV
  classification_report.json  -- same, machine-readable
  macro_f1_summary.json       -- single-number summary for quick comparison

  Predictions
  -----------
  y_test.npy             -- ground-truth labels (test set)
  y_pred.npy             -- predicted class indices
  y_prob.npy             -- softmax probabilities (N x num_classes)

  Confusion matrices
  ------------------
  confusion_matrix_counts.png      -- raw counts heatmap
  confusion_matrix_normalised.png  -- recall-normalised heatmap
  confusion_matrix_counts.csv      -- raw counts as CSV
  confusion_matrix_normalised.csv  -- normalised as CSV

  Per-class metrics charts
  ------------------------
  per_class_precision.png
  per_class_recall.png
  per_class_f1.png
  per_class_metrics.csv   -- precision / recall / f1 / support per class

  ROC curves (One-vs-Rest)
  ------------------------
  roc_curves.png          -- all classes overlaid
  roc_auc_scores.json     -- per-class AUC + macro AUC

  Precision-Recall curves (One-vs-Rest)
  --------------------------------------
  pr_curves.png           -- all classes overlaid
  pr_ap_scores.json       -- per-class AP + macro AP

  Model information
  -----------------
  model_summary.txt       -- Keras model.summary() text
  model_info.json         -- params / size / architecture name

  Benchmark snapshot
  ------------------
  benchmark_snapshot.json -- params, KB, inference ms/sample (quick reference)

Usage
-----
  from results_saver import save_all_results

  save_all_results(
      model_name  = "edge_cnn",
      out_dir     = "outputs/edge_cnn",
      y_test      = y_test,
      y_pred      = y_pred,
      y_prob      = y_prob,
      history     = history.history,          # dict from model.fit()
      model       = keras_model,              # optional Keras model object
      class_names = ["Normal","AFib",...],
      inference_ms= 5.2,                      # optional, ms per sample
  )
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)

# ── Default class names (overrideable per call) ───────────────────────────────
DEFAULT_CLASS_NAMES = ["Normal", "AFib", "Bradycardia", "Tachycardia", "Other"]

# ── Plot aesthetics ───────────────────────────────────────────────────────────
PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


# ==============================================================================
#  MAIN ENTRY POINT
# ==============================================================================

def save_all_results(
    model_name:   str,
    out_dir:      str,
    y_test:       np.ndarray,
    y_pred:       np.ndarray,
    y_prob:       np.ndarray | None = None,
    history:      dict | None       = None,
    model:        Any               = None,
    class_names:  list[str] | None  = None,
    inference_ms: float | None      = None,
) -> str:
    """
    Save every possible result artifact for one model run.

    Parameters
    ----------
    model_name   : short identifier (e.g. "edge_cnn")
    out_dir      : model output directory (e.g. "outputs/edge_cnn")
    y_test       : ground-truth integer labels, shape (N,)
    y_pred       : predicted class indices,    shape (N,)
    y_prob       : softmax probabilities,      shape (N, C)  [optional]
    history      : dict from Keras history.history            [optional]
    model        : Keras model object                         [optional]
    class_names  : list of class label strings               [optional]
    inference_ms : total inference time in ms                 [optional]

    Returns
    -------
    str  -- path to the results sub-directory
    """
    results_dir = os.path.join(out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    names = class_names or DEFAULT_CLASS_NAMES
    present = sorted(np.unique(y_test).tolist())
    present_names = [names[i] for i in present if i < len(names)]

    logger.info("Saving all results for '%s' -> %s", model_name, results_dir)

    # ── 1. Raw predictions ────────────────────────────────────────────────────
    _save_predictions(results_dir, y_test, y_pred, y_prob)

    # ── 2. Classification metrics ─────────────────────────────────────────────
    _save_classification_metrics(
        results_dir, model_name, y_test, y_pred, names, present
    )

    # ── 3. Confusion matrices ─────────────────────────────────────────────────
    _save_confusion_matrices(results_dir, y_test, y_pred, present_names)

    # ── 4. Per-class bar charts ───────────────────────────────────────────────
    _save_per_class_charts(results_dir, y_test, y_pred, present_names)

    # ── 5. ROC + PR curves (need y_prob) ─────────────────────────────────────
    if y_prob is not None and len(present) > 1:
        _save_roc_curves(results_dir, y_test, y_prob, present, present_names)
        _save_pr_curves( results_dir, y_test, y_prob, present, present_names)

    # ── 6. Training history ───────────────────────────────────────────────────
    if history:
        _save_history(results_dir, model_name, history)

    # ── 7. Model summary + benchmark snapshot ────────────────────────────────
    _save_model_info(results_dir, model_name, model, inference_ms, y_test)

    logger.info("All results saved -> %s", results_dir)
    return results_dir


# ==============================================================================
#  1. RAW PREDICTIONS
# ==============================================================================

def _save_predictions(
    d: str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
) -> None:
    np.save(os.path.join(d, "y_test.npy"), y_test)
    np.save(os.path.join(d, "y_pred.npy"), y_pred)
    if y_prob is not None:
        np.save(os.path.join(d, "y_prob.npy"), y_prob)

    # Also save a simple CSV of (true, pred) pairs for easy inspection
    df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    if y_prob is not None:
        for c in range(y_prob.shape[1]):
            df[f"prob_class{c}"] = y_prob[:, c]
    df.to_csv(os.path.join(d, "predictions.csv"), index=False)
    logger.info("  [1/7] Predictions saved (%d samples)", len(y_test))


# ==============================================================================
#  2. CLASSIFICATION METRICS
# ==============================================================================

def _save_classification_metrics(
    d:          str,
    model_name: str,
    y_test:     np.ndarray,
    y_pred:     np.ndarray,
    names:      list[str],
    present:    list[int],
) -> None:
    present_names = [names[i] for i in present if i < len(names)]
    labels_arr    = list(present)

    # ── Full sklearn report ───────────────────────────────────────────────────
    report_str = classification_report(
        y_test, y_pred,
        labels=labels_arr,
        target_names=present_names,
        zero_division=0,
    )
    header = (
        f"\n{'=' * 60}\n"
        f"  {model_name}  --  Test Set Results\n"
        f"{'=' * 60}\n"
    )
    full_report = header + report_str
    Path(os.path.join(d, "classification_report.txt")).write_text(full_report)

    # ── Per-class CSV ─────────────────────────────────────────────────────────
    report_dict = classification_report(
        y_test, y_pred,
        labels=labels_arr,
        target_names=present_names,
        zero_division=0,
        output_dict=True,
    )
    rows = []
    for cls_name in present_names:
        if cls_name in report_dict:
            m = report_dict[cls_name]
            rows.append({
                "class":     cls_name,
                "precision": round(m["precision"], 4),
                "recall":    round(m["recall"],    4),
                "f1_score":  round(m["f1-score"],  4),
                "support":   int(m["support"]),
            })
    # Append aggregate rows
    for agg in ("macro avg", "weighted avg", "accuracy"):
        if agg in report_dict:
            v = report_dict[agg]
            if isinstance(v, dict):
                rows.append({
                    "class":     agg,
                    "precision": round(v.get("precision", 0), 4),
                    "recall":    round(v.get("recall",    0), 4),
                    "f1_score":  round(v.get("f1-score",  0), 4),
                    "support":   int(v.get("support",     0)),
                })
            else:  # accuracy is a scalar
                rows.append({
                    "class": "accuracy", "precision": round(v, 4),
                    "recall": "", "f1_score": "", "support": len(y_test),
                })

    df_report = pd.DataFrame(rows)
    df_report.to_csv(os.path.join(d, "classification_report.csv"), index=False)

    # ── JSON ──────────────────────────────────────────────────────────────────
    with open(os.path.join(d, "classification_report.json"), "w") as f:
        json.dump(report_dict, f, indent=2)

    # ── Single-number summary ─────────────────────────────────────────────────
    macro_f1 = f1_score(y_test, y_pred, average="macro",
                        labels=labels_arr, zero_division=0)
    macro_p  = precision_score(y_test, y_pred, average="macro",
                               labels=labels_arr, zero_division=0)
    macro_r  = recall_score(y_test, y_pred, average="macro",
                            labels=labels_arr, zero_division=0)
    accuracy = float((y_test == y_pred).mean())

    summary = {
        "model":           model_name,
        "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_test_samples": int(len(y_test)),
        "num_classes_present": len(present),
        "accuracy":        round(accuracy, 4),
        "macro_f1":        round(macro_f1, 4),
        "macro_precision": round(macro_p,  4),
        "macro_recall":    round(macro_r,  4),
    }
    with open(os.path.join(d, "macro_f1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "  [2/7] Metrics  Acc=%.4f  MacroF1=%.4f  MacroP=%.4f  MacroR=%.4f",
        accuracy, macro_f1, macro_p, macro_r,
    )


# ==============================================================================
#  3. CONFUSION MATRICES
# ==============================================================================

def _save_confusion_matrices(
    d:       str,
    y_test:  np.ndarray,
    y_pred:  np.ndarray,
    names:   list[str],
) -> None:
    labels = list(range(len(names)))
    cm_raw  = confusion_matrix(y_test, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_test, y_pred, labels=labels, normalize="true")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    pd.DataFrame(cm_raw,  index=names, columns=names).to_csv(
        os.path.join(d, "confusion_matrix_counts.csv"))
    pd.DataFrame(cm_norm.round(4), index=names, columns=names).to_csv(
        os.path.join(d, "confusion_matrix_normalised.csv"))

    # ── Plot raw counts ───────────────────────────────────────────────────────
    _plot_cm(cm_raw,  names, "Confusion Matrix (Raw Counts)",
             os.path.join(d, "confusion_matrix_counts.png"), fmt="d", cmap="Blues")

    # ── Plot normalised ───────────────────────────────────────────────────────
    _plot_cm(cm_norm, names, "Confusion Matrix (Normalised Recall)",
             os.path.join(d, "confusion_matrix_normalised.png"), fmt=".2f", cmap="YlOrRd")

    logger.info("  [3/7] Confusion matrices saved")


def _plot_cm(cm, names, title, path, fmt, cmap):
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n)))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap=cmap,
        xticklabels=names, yticklabels=names,
        ax=ax, linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title(title,        fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ==============================================================================
#  4. PER-CLASS CHARTS
# ==============================================================================

def _save_per_class_charts(
    d:     str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    names:  list[str],
) -> None:
    labels = list(range(len(names)))

    p_arr = precision_score(y_test, y_pred, labels=labels,
                            average=None, zero_division=0)
    r_arr = recall_score(   y_test, y_pred, labels=labels,
                            average=None, zero_division=0)
    f_arr = f1_score(       y_test, y_pred, labels=labels,
                            average=None, zero_division=0)
    support = np.array(
        [int((y_test == i).sum()) for i in labels]
    )

    # ── CSV ───────────────────────────────────────────────────────────────────
    df = pd.DataFrame({
        "class":     names,
        "precision": p_arr.round(4),
        "recall":    r_arr.round(4),
        "f1_score":  f_arr.round(4),
        "support":   support,
    })
    df.to_csv(os.path.join(d, "per_class_metrics.csv"), index=False)

    # ── Combined bar chart (P / R / F1 side-by-side per class) ───────────────
    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.8), 5))
    ax.bar(x - w, p_arr, w, label="Precision", color=PALETTE[0], alpha=0.85)
    ax.bar(x,     r_arr, w, label="Recall",    color=PALETTE[1], alpha=0.85)
    ax.bar(x + w, f_arr, w, label="F1-Score",  color=PALETTE[2], alpha=0.85)

    for xi, (p, r, f) in enumerate(zip(p_arr, r_arr, f_arr)):
        ax.text(xi - w, p + 0.01, f"{p:.2f}", ha="center", fontsize=7)
        ax.text(xi,     r + 0.01, f"{r:.2f}", ha="center", fontsize=7)
        ax.text(xi + w, f + 0.01, f"{f:.2f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(d, "per_class_prf_chart.png"), dpi=150)
    plt.close()

    # ── Individual metric charts ───────────────────────────────────────────────
    for metric_name, values in [
        ("Precision", p_arr), ("Recall", r_arr), ("F1_Score", f_arr)
    ]:
        fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.4), 4))
        bars = ax.bar(names, values,
                      color=[PALETTE[i % len(PALETTE)] for i in range(len(names))],
                      alpha=0.85)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(metric_name)
        ax.set_title(f"Per-Class {metric_name}")
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(d, f"per_class_{metric_name.lower()}.png"), dpi=150
        )
        plt.close()

    logger.info("  [4/7] Per-class charts saved")


# ==============================================================================
#  5. ROC CURVES
# ==============================================================================

def _save_roc_curves(
    d:       str,
    y_test:  np.ndarray,
    y_prob:  np.ndarray,
    present: list[int],
    names:   list[str],
) -> None:
    y_bin = label_binarize(y_test, classes=present)

    fig, ax = plt.subplots(figsize=(7, 6))
    auc_scores: dict[str, float] = {}

    for i, (cls_idx, cls_name) in enumerate(zip(present, names)):
        col = y_bin[:, i] if y_bin.ndim > 1 else y_bin
        prob_col = y_prob[:, cls_idx] if y_prob.shape[1] > cls_idx else y_prob[:, -1]
        if col.sum() == 0:
            continue
        try:
            fpr, tpr, _ = roc_curve(col, prob_col)
            auc_val = roc_auc_score(col, prob_col)
            auc_scores[cls_name] = round(float(auc_val), 4)
            ax.plot(fpr, tpr,
                    label=f"{cls_name}  (AUC={auc_val:.3f})",
                    color=PALETTE[i % len(PALETTE)], linewidth=2)
        except Exception:
            pass

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("ROC Curves (One-vs-Rest)",  fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(d, "roc_curves.png"), dpi=150)
    plt.close()

    if auc_scores:
        macro_auc = round(float(np.mean(list(auc_scores.values()))), 4)
        auc_scores["macro_avg"] = macro_auc
        with open(os.path.join(d, "roc_auc_scores.json"), "w") as f:
            json.dump(auc_scores, f, indent=2)
        logger.info("  [5/7] ROC curves saved  macro_AUC=%.4f", macro_auc)


# ==============================================================================
#  6. PRECISION-RECALL CURVES
# ==============================================================================

def _save_pr_curves(
    d:       str,
    y_test:  np.ndarray,
    y_prob:  np.ndarray,
    present: list[int],
    names:   list[str],
) -> None:
    y_bin = label_binarize(y_test, classes=present)

    fig, ax = plt.subplots(figsize=(7, 6))
    ap_scores: dict[str, float] = {}

    for i, (cls_idx, cls_name) in enumerate(zip(present, names)):
        col      = y_bin[:, i] if y_bin.ndim > 1 else y_bin
        prob_col = y_prob[:, cls_idx] if y_prob.shape[1] > cls_idx else y_prob[:, -1]
        if col.sum() == 0:
            continue
        try:
            prec, rec, _ = precision_recall_curve(col, prob_col)
            ap = average_precision_score(col, prob_col)
            ap_scores[cls_name] = round(float(ap), 4)
            ax.plot(rec, prec,
                    label=f"{cls_name}  (AP={ap:.3f})",
                    color=PALETTE[i % len(PALETTE)], linewidth=2)
        except Exception:
            pass

    ax.set_xlabel("Recall",    fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves (One-vs-Rest)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(d, "pr_curves.png"), dpi=150)
    plt.close()

    if ap_scores:
        macro_ap = round(float(np.mean(list(ap_scores.values()))), 4)
        ap_scores["macro_avg"] = macro_ap
        with open(os.path.join(d, "pr_ap_scores.json"), "w") as f:
            json.dump(ap_scores, f, indent=2)
        logger.info("  [6/7] PR curves saved  macro_AP=%.4f", macro_ap)


# ==============================================================================
#  7. TRAINING HISTORY
# ==============================================================================

def _save_history(d: str, model_name: str, history: dict) -> None:
    # ── CSV & JSON ────────────────────────────────────────────────────────────
    df_hist = pd.DataFrame(history)
    df_hist.index.name = "epoch"
    df_hist.to_csv(os.path.join(d, "history.csv"))
    with open(os.path.join(d, "history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.items()}, f, indent=2)

    # ── Determine available metrics ───────────────────────────────────────────
    has_f1   = "val_macro_f1" in history
    has_loss = "loss"         in history
    has_acc  = "accuracy"     in history

    num_panels = sum([has_loss, has_acc, has_f1])
    if num_panels == 0:
        return

    fig, axes = plt.subplots(1, num_panels,
                             figsize=(6 * num_panels, 4.5), squeeze=False)
    axes = axes[0]
    panel = 0

    if has_loss:
        ax = axes[panel]; panel += 1
        ax.plot(history["loss"],     label="Train Loss", color=PALETTE[0], linewidth=2)
        if "val_loss" in history:
            ax.plot(history["val_loss"], label="Val Loss",   color=PALETTE[1],
                    linewidth=2, linestyle="--")
        ax.set_title("Loss");  ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(alpha=0.3)
        # Annotate best val loss
        if "val_loss" in history:
            best_ep = int(np.argmin(history["val_loss"]))
            ax.axvline(best_ep, color="gray", linestyle=":", alpha=0.7)
            ax.annotate(f"best={history['val_loss'][best_ep]:.4f}",
                        xy=(best_ep, history["val_loss"][best_ep]),
                        xytext=(best_ep + 1, history["val_loss"][best_ep] + 0.02),
                        fontsize=8, color="gray")

    if has_acc:
        ax = axes[panel]; panel += 1
        ax.plot(history["accuracy"],     label="Train Acc", color=PALETTE[2], linewidth=2)
        if "val_accuracy" in history:
            ax.plot(history["val_accuracy"], label="Val Acc",   color=PALETTE[3],
                    linewidth=2, linestyle="--")
        ax.set_ylim(0, 1.05)
        ax.set_title("Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.legend(); ax.grid(alpha=0.3)

    if has_f1:
        ax = axes[panel]; panel += 1
        ax.plot(history["val_macro_f1"], label="Val Macro F1",
                color=PALETTE[4], linewidth=2)
        ax.set_ylim(0, 1.05)
        ax.set_title("Val Macro F1"); ax.set_xlabel("Epoch"); ax.set_ylabel("F1")
        best_ep = int(np.argmax(history["val_macro_f1"]))
        ax.axvline(best_ep, color="gray", linestyle=":", alpha=0.7)
        ax.annotate(f"best={history['val_macro_f1'][best_ep]:.4f}",
                    xy=(best_ep, history["val_macro_f1"][best_ep]),
                    xytext=(best_ep + 1, history["val_macro_f1"][best_ep] - 0.05),
                    fontsize=8, color="gray")
        ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(f"{model_name} — Training History", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(d, "plot_history.png"), dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("  [7a] Training history saved (%d epochs)", len(df_hist))


# ==============================================================================
#  8. MODEL INFO + BENCHMARK SNAPSHOT
# ==============================================================================

def _save_model_info(
    d:            str,
    model_name:   str,
    model:        Any,
    inference_ms: float | None,
    y_test:       np.ndarray,
) -> None:
    info: dict[str, Any] = {
        "model_name":    model_name,
        "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_test":      int(len(y_test)),
    }

    if model is not None:
        # ── Save model summary as text ────────────────────────────────────────
        buf = io.StringIO()
        model.summary(print_fn=lambda s: buf.write(s + "\n"))
        summary_txt = buf.getvalue()
        Path(os.path.join(d, "model_summary.txt")).write_text(summary_txt)

        total_params = model.count_params()
        info["total_params"]  = total_params
        info["model_size_kb"] = round(total_params * 4 / 1024, 1)  # float32

        # Find H5 file in parent dir for actual on-disk size
        parent = os.path.dirname(d)
        for fname in os.listdir(parent):
            if fname.endswith(".h5") and model_name in fname:
                kb = os.path.getsize(os.path.join(parent, fname)) / 1024
                info["h5_size_kb"] = round(kb, 1)
                break

    if inference_ms is not None:
        info["inference_ms_total"]  = round(inference_ms, 2)
        info["inference_ms_sample"] = round(inference_ms / max(len(y_test), 1) * 1000, 4)

    with open(os.path.join(d, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # ── Benchmark snapshot (also written at top level for easy comparison) ────
    snap_path = os.path.join(os.path.dirname(d), "benchmark_snapshot.json")
    existing: dict = {}
    if os.path.isfile(snap_path):
        with open(snap_path) as f:
            try:
                existing = json.load(f)
            except Exception:
                pass
    existing[model_name] = info
    with open(snap_path, "w") as f:
        json.dump(existing, f, indent=2)

    logger.info("  [7b] Model info saved")


# ==============================================================================
#  XGBoost results (separate entry point)
# ==============================================================================

def save_xgb_results(
    out_dir:      str,
    y_test:       np.ndarray,
    y_pred:       np.ndarray,
    y_prob:       np.ndarray | None = None,
    model:        Any               = None,
    class_names:  list[str] | None  = None,
    inference_ms: float | None      = None,
) -> str:
    """
    Save all XGBoost results.  Includes feature importance plot if
    the booster has feature scores.
    """
    results_dir = save_all_results(
        model_name   = "xgb",
        out_dir      = out_dir,
        y_test       = y_test,
        y_pred       = y_pred,
        y_prob       = y_prob,
        class_names  = class_names,
        inference_ms = inference_ms,
    )

    # ── Feature importance (if booster available) ─────────────────────────────
    if model is not None:
        try:
            import xgboost as xgb
            scores = model.get_score(importance_type="gain")
            if scores:
                top_n = 30
                items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
                feat_names, feat_scores = zip(*items)

                fig, ax = plt.subplots(figsize=(8, max(5, len(feat_names) * 0.35)))
                y_pos = np.arange(len(feat_names))
                ax.barh(y_pos, feat_scores,
                        color=[PALETTE[i % len(PALETTE)] for i in range(len(feat_names))],
                        alpha=0.85)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(feat_names, fontsize=9)
                ax.invert_yaxis()
                ax.set_xlabel("Gain (Feature Importance)")
                ax.set_title(f"XGBoost — Top {len(feat_names)} Features by Gain")
                ax.grid(axis="x", alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, "feature_importance.png"), dpi=150)
                plt.close()

                fi_df = pd.DataFrame(list(scores.items()),
                                     columns=["feature", "gain"])
                fi_df.sort_values("gain", ascending=False, inplace=True)
                fi_df.to_csv(os.path.join(results_dir, "feature_importance.csv"),
                             index=False)
                logger.info("  [+] XGBoost feature importance saved")
        except Exception as e:
            logger.warning("Could not save XGBoost feature importance: %s", e)

    return results_dir
