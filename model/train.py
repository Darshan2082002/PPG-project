"""
train.py
========
Training script for the PPG arrhythmia detection model.

CPU-only mode
-------------
This project runs exclusively on CPU (no CUDA / DirectML GPU required).
All GPU devices are suppressed before TensorFlow is loaded.

Usage (from project root, with venv activated):
    python model/train.py
    python model/train.py --epochs 50 --batch-size 8
    python model/train.py --epochs 3             # quick smoke test

Outputs:
    outputs/best_model.h5          – best checkpoint (lowest val loss)
    outputs/final_model.h5         – weights at end of training
    outputs/training_history.png   – loss + accuracy curves
"""

import os
import sys
import argparse
import logging

# ── Force CPU-only mode (must happen BEFORE TensorFlow loads) ─────────────────
# Suppress any GPU / DirectML devices unconditionally so tensorflow-cpu
# never tries to initialise GPU kernels.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DIRECTML_DISABLE"]  = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # quieter TF C++ logs

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for servers
import matplotlib.pyplot as plt

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.architecture import build_model, NUM_CLASSES, INPUT_LENGTH

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATASET_PATH  = os.path.join(OUTPUT_DIR, "dataset.npz")
BEST_MODEL    = os.path.join(OUTPUT_DIR, "best_model.h5")
FINAL_MODEL   = os.path.join(OUTPUT_DIR, "final_model.h5")
HISTORY_PLOT  = os.path.join(OUTPUT_DIR, "training_history.png")


# ══════════════════════════════════════════════════════════════════════════════
#  DEVICE CONFIGURATION (CPU-only)
# ══════════════════════════════════════════════════════════════════════════════

def configure_device():
    """
    Return a CPU OneDeviceStrategy.

    Returns
    -------
    strategy : tf.distribute.OneDeviceStrategy on /cpu:0
    num_replicas : int  (always 1)
    """
    import tensorflow as tf
    logger.info("Running on CPU (tensorflow-cpu build, no GPU required).")
    strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    return strategy, 1


# ══════════════════════════════════════════════════════════════════════════════
#  DATA / WEIGHTS / CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(path: str):
    """Load pre-processed numpy arrays from .npz file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Run:  python preprocessing/fuse_datasets.py  first."
        )
    logger.info("Loading dataset from %s …", path)
    data = np.load(path, allow_pickle=True)
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.int32)
    X_val   = data["X_val"].astype(np.float32)
    y_val   = data["y_val"].astype(np.int32)
    X_test  = data["X_test"].astype(np.float32)
    y_test  = data["y_test"].astype(np.int32)
    logger.info("  Train: %s  |  Val: %s  |  Test: %s",
                X_train.shape, X_val.shape, X_test.shape)
    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_class_weights(y: np.ndarray, num_classes: int) -> dict:
    """Inverse-frequency class weights to handle imbalanced classes.
       Robust to missing classes in small dataset subsets.
    """
    from sklearn.utils.class_weight import compute_class_weight
    present_classes = np.unique(y)
    cw_values = compute_class_weight("balanced", classes=present_classes, y=y)

    cw_dict = {i: 1.0 for i in range(num_classes)}
    for cls, weight in zip(present_classes, cw_values):
        cw_dict[cls] = float(weight)

    return cw_dict


def build_callbacks(best_model_path: str):
    """Standard training callbacks."""
    import tensorflow as tf
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_history(history, save_path: str) -> None:
    """Save training / validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Loss ──────────────────────────────────────────────────────────────────
    axes[0].plot(history.history["loss"],     label="Train Loss", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss",   linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    axes[1].plot(history.history["accuracy"],     label="Train Acc", linewidth=2)
    axes[1].plot(history.history["val_accuracy"], label="Val Acc",   linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    logger.info("Training history saved → %s", save_path)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def train(
    epochs: int = 100,
    batch_size: int = 8,
    use_class_weights: bool = True,
    seed: int = 42,
):
    import tensorflow as tf
    tf.random.set_seed(seed)
    np.random.seed(seed)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Device / strategy setup ──────────────────────────────────────────────────
    strategy, num_replicas = configure_device()

    # Scale batch size when using multiple GPUs so each replica sees
    # `batch_size` samples (global_batch = batch_size × num_replicas)
    global_batch_size = batch_size * num_replicas
    if num_replicas > 1:
        logger.info(
            "Scaling batch size: %d (per-replica) × %d (replicas) = %d (global)",
            batch_size, num_replicas, global_batch_size,
        )

    # ── Data ──────────────────────────────────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(DATASET_PATH)

    # Verify input shape matches model expectation
    if X_train.shape[1] != INPUT_LENGTH:
        raise ValueError(
            f"Dataset window length {X_train.shape[1]} ≠ model INPUT_LENGTH {INPUT_LENGTH}."
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Building model …")
    model = build_model(num_classes=NUM_CLASSES)
    model.summary(print_fn=logger.info)

    # ── Class weights ─────────────────────────────────────────────────────────
    cw = None
    if use_class_weights:
        cw = compute_class_weights(y_train, NUM_CLASSES)
        logger.info("Class weights: %s", cw)

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info(
        "Training for up to %d epochs (global_batch=%d, replicas=%d) …",
        epochs, global_batch_size, num_replicas,
    )
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=global_batch_size,
        validation_data=(X_val, y_val),
        class_weight=cw,
        callbacks=build_callbacks(BEST_MODEL),
        verbose=1,
    )

    # ── Save final weights ────────────────────────────────────────────────────
    model.save(FINAL_MODEL)
    logger.info("Final model saved → %s", FINAL_MODEL)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_history(history, HISTORY_PLOT)

    # ── Quick test-set accuracy ───────────────────────────────────────────────
    logger.info("Evaluating on test set …")
    best_model = tf.keras.models.load_model(BEST_MODEL)
    test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
    logger.info("Test loss: %.4f  |  Test accuracy: %.4f", test_loss, test_acc)
    logger.info("Training complete. Best model: %s", BEST_MODEL)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPG arrhythmia model (CPU-only)")
    parser.add_argument("--epochs",           type=int,   default=100,
                        help="Maximum training epochs (default: 100)")
    parser.add_argument("--batch-size",       type=int,   default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Disable inverse-frequency class weights")
    parser.add_argument("--seed",             type=int,   default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_class_weights=not args.no_class_weights,
        seed=args.seed,
    )
