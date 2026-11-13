"""
evaluate.py
===========
Load the best-saved model and compute full evaluation metrics on the test set.

Outputs saved to outputs/:
    confusion_matrix.png    – colour-coded confusion matrix
    classification_report.txt

Usage:
    python model/evaluate.py
    python model/evaluate.py --model outputs/best_model.h5
"""

import os
import sys
import argparse
import logging
import numpy as np

# ── Force CPU-only mode (must happen BEFORE TensorFlow loads) ─────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DIRECTML_DISABLE"]  = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preprocessing.label_mapping import CLASS_NAMES

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATASET_PATH = os.path.join(OUTPUT_DIR, "dataset.npz")
DEFAULT_MODEL = os.path.join(OUTPUT_DIR, "best_model.h5")
CM_PATH      = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
REPORT_PATH  = os.path.join(OUTPUT_DIR, "classification_report.txt")


def evaluate(model_path: str) -> None:
    import tensorflow as tf
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        accuracy_score, f1_score, precision_score, recall_score,
    )

    # ── Load data ──────────────────────────────────────────────────────────────
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    data = np.load(DATASET_PATH, allow_pickle=True)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int32)
    logger.info("Test set: %s", X_test.shape)

    # ── Load model ─────────────────────────────────────────────────────────────
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    logger.info("Loading model: %s", model_path)
    model = tf.keras.models.load_model(model_path)

    # ── Predict ────────────────────────────────────────────────────────────────
    logger.info("Running inference on %d test windows …", len(X_test))
    y_prob = model.predict(X_test, batch_size=128, verbose=0)
    y_pred = np.argmax(y_prob, axis=1).astype(np.int32)

    # ── Metrics ────────────────────────────────────────────────────────────────
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_test, y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    summary = (
        f"\n{'='*55}\n"
        f"  PPG Arrhythmia Detection – Evaluation Results\n"
        f"{'='*55}\n"
        f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)\n"
        f"  Precision : {prec:.4f}  (macro)\n"
        f"  Recall    : {rec:.4f}  (macro)\n"
        f"  F1-Score  : {f1:.4f}  (macro)\n"
        f"{'='*55}\n\n"
        f"{report}\n"
    )
    print(summary)

    # ── Save report ────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as fh:
        fh.write(summary)
    logger.info("Classification report saved → %s", REPORT_PATH)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASS_NAMES))))

    # Get only classes present in test set
    present = sorted(set(y_test.tolist()))
    present_names = [CLASS_NAMES[i] for i in present]
    cm_present = cm[np.ix_(present, present)]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_present,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=present_names,
        yticklabels=present_names,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title("Confusion Matrix – PPG Arrhythmia Detection", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved → %s", CM_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPG arrhythmia model")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Path to .h5 model file")
    args = parser.parse_args()
    evaluate(args.model)
