"""
train_models.py
===============
Multi-architecture training script for PPG-based arrhythmia detection.

Implements and trains four candidate architectures:

  1. XGBoost Classifier          -- on extracted feature vectors (tabular)
  2. 1D Edge-Optimised CNN       -- MobileNet-style depthwise-separable blocks
  3. Bidirectional LSTM / Conv-LSTM -- Hybrid Conv1D + BiLSTM
  4. Temporal Convolutional Network (TCN) -- dilated causal convolutions

Architecture trade-offs for ESP32-S3 edge deployment
-----------------------------------------------------
  XGBoost   : ~20-200 KB C code (emlearn), zero runtime library, <1 ms latency.
               Requires hand-crafted features; loses raw-waveform detail.

  Edge CNN  : ~60-120 KB INT8 TFLite; SIMD-accelerated on S3 Xtensa LX7.
               Best param/accuracy ratio for morphological pattern capture.

  BiLSTM    : ~300-600 KB INT8 TFLite; LSTM cells are harder to quantize cleanly.
               Best at long-range rhythm dynamics; highest RAM footprint.

  TCN       : ~200-400 KB INT8 TFLite; dilation gives large receptive field
               without recurrence; parallelisable unlike LSTM.

Usage
-----
  python train_models.py                  # train all 4 models
  python train_models.py --models xgb cnn # train only XGBoost and Edge CNN
  python train_models.py --epochs 50 --batch-size 64
  python train_models.py --no-gpu         # force CPU

Outputs (per model, written to outputs/<model_name>/)
------
  best_<name>.h5 / best_<name>.json+bin   -- saved model
  history_<name>.png                      -- training curves
  class_report_<name>.txt                 -- classification report
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import warnings
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Import the storage helpers so we read both .parquet and .csv fallback
try:
    from feature_extraction import load_features as _load_feat_file
except ImportError:
    def _load_feat_file(path: str) -> "pd.DataFrame":
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)

# Import comprehensive results saver
try:
    from results_saver import save_all_results, save_xgb_results
except ImportError:
    save_all_results = None
    save_xgb_results = None
    logging.getLogger(__name__).warning(
        "results_saver.py not found -- extended result saving disabled"
    )

# ── Project constants ----------------------------------------------------------
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "outputs")
DATASET_PATH = os.path.join(OUTPUT_DIR, "dataset.npz")
FEATURES_TRAIN = os.path.join(OUTPUT_DIR, "features_train.parquet")
FEATURES_VAL   = os.path.join(OUTPUT_DIR, "features_val.parquet")
FEATURES_TEST  = os.path.join(OUTPUT_DIR, "features_test.parquet")

NUM_CLASSES  = 5
INPUT_LENGTH = 1250    # 10 s at 125 Hz
FS           = 125

CLASS_NAMES = ["Normal", "AFib", "Bradycardia", "Tachycardia", "Other"]


# ==============================================================================
#  DATA LOADERS
# ==============================================================================

def load_raw_dataset() -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """Load raw windowed PPG signals from dataset.npz."""
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Run:  python preprocessing/fuse_datasets.py  first."
        )
    data = np.load(DATASET_PATH, allow_pickle=True)
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.int32)
    X_val   = data["X_val"].astype(np.float32)
    y_val   = data["y_val"].astype(np.int32)
    X_test  = data["X_test"].astype(np.float32)
    y_test  = data["y_test"].astype(np.int32)
    logger.info("Raw signals   Train=%s  Val=%s  Test=%s",
                X_train.shape, X_val.shape, X_test.shape)
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_feature_dataset() -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """Load extracted feature vectors, supporting both .parquet and .csv."""
    # Accept either .parquet or .csv (csv is the fallback when pyarrow is absent)
    def _load(base_path: str) -> tuple[np.ndarray, np.ndarray]:
        df = _load_feat_file(base_path)       # handles extension swap automatically
        y  = df["label"].values.astype(np.int32)
        X  = df.drop(columns=["label"]).values.astype(np.float32)
        return X, y

    missing = []
    for p in [FEATURES_TRAIN, FEATURES_VAL, FEATURES_TEST]:
        parquet_ok = os.path.isfile(p)
        csv_ok     = os.path.isfile(p.replace(".parquet", ".csv"))
        if not parquet_ok and not csv_ok:
            missing.append(p)
    if missing:
        raise FileNotFoundError(
            "Feature files not found:\n  " + "\n  ".join(missing) + "\n"
            "Run:  python feature_extraction.py  first."
        )

    X_train, y_train = _load(FEATURES_TRAIN)
    X_val,   y_val   = _load(FEATURES_VAL)
    X_test,  y_test  = _load(FEATURES_TEST)

    logger.info("Feature vecs  Train=%s  Val=%s  Test=%s",
                X_train.shape, X_val.shape, X_test.shape)
    return X_train, y_train, X_val, y_val, X_test, y_test


# ==============================================================================
#  HELPERS: class weights, evaluation print
# ==============================================================================

def compute_class_weights_dict(y: np.ndarray) -> dict[int, float]:
    """Inverse-frequency class weights -- robust to missing classes."""
    from sklearn.utils.class_weight import compute_class_weight
    present = np.unique(y)
    values  = compute_class_weight("balanced", classes=present, y=y)
    cw: dict[int, float] = {i: 1.0 for i in range(NUM_CLASSES)}
    for cls, w in zip(present, values):
        cw[int(cls)] = float(w)
    return cw


def print_eval(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> str:
    """Print and return a classification report string."""
    from sklearn.metrics import classification_report, f1_score
    report = classification_report(
        y_true, y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    summary = (
        f"\n{'='*60}\n"
        f"  {model_name}  --  Test Set Results\n"
        f"{'='*60}\n"
        f"  Macro F1-Score : {macro_f1:.4f}\n"
        f"{'='*60}\n\n"
        f"{report}"
    )
    print(summary)
    return summary


# ==============================================================================
#  MODEL 1: XGBoost Classifier
# ==============================================================================

def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    out_dir: str,
    seed:    int = 42,
) -> None:
    """
    Train a hyperparameter-tuned XGBoost classifier on the extracted feature set.

    Class imbalance handling
    ------------------------
    * ``scale_pos_weight`` is NOT used directly (multiclass).
    * Instead, sample_weights are passed to xgb.train via DMatrix, giving each
      training sample a weight inversely proportional to its class frequency.
    * This is the most effective strategy for extreme imbalance with XGBoost.

    Hyperparameter rationale
    ------------------------
    max_depth=6      : deep enough to capture feature interactions
    n_estimators=400 : enough trees; early stopping prevents overfitting
    eta=0.05         : small learning rate + more trees = lower generalisation error
    subsample=0.8    : row subsampling (prevents overfitting, speeds up training)
    colsample=0.8    : column subsampling (prevents feature co-adaptation)
    min_child_weight=5: larger leaf minimum prevents memorising rare classes
    """
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("XGBoost not installed. Run: pip install xgboost")
        return

    logger.info("=" * 60)
    logger.info("Training Model 1: XGBoost Classifier")
    logger.info("=" * 60)
    os.makedirs(out_dir, exist_ok=True)

    # -- Sample weights (class-balanced) ---------------------------------------
    cw    = compute_class_weights_dict(y_train)
    sw    = np.array([cw[int(lbl)] for lbl in y_train], dtype=np.float32)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sw)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    dtest  = xgb.DMatrix(X_test,  label=y_test)

    params = {
        "objective":         "multi:softmax",
        "num_class":         NUM_CLASSES,
        "eval_metric":       "mlogloss",
        "max_depth":         6,
        "eta":               0.05,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "min_child_weight":  5,
        "gamma":             0.1,       # min split gain (regularisation)
        "reg_alpha":         0.1,       # L1 regularisation
        "reg_lambda":        1.0,       # L2 regularisation
        "seed":              seed,
        "nthread":           -1,        # use all CPU cores
        "verbosity":         0,
    }

    evals_result: dict = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=600,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30,
        evals_result=evals_result,
        verbose_eval=50,
    )

    # -- Save model ------------------------------------------------------------
    model_path = os.path.join(out_dir, "xgb_model.ubj")
    model.save_model(model_path)
    logger.info("XGBoost model saved -> %s", model_path)

    # -- Evaluate --------------------------------------------------------------
    t0     = time.perf_counter()
    y_pred = model.predict(dtest).astype(np.int32)
    infer_ms = (time.perf_counter() - t0) * 1000.0

    # Softmax probabilities (needed for ROC / PR curves)
    params_prob = dict(params)
    params_prob["objective"] = "multi:softprob"
    model_prob = xgb.train(
        params_prob, dtrain,
        num_boost_round=model.best_iteration + 1,
        verbose_eval=False,
    )
    y_prob_flat = model_prob.predict(dtest)
    y_prob = y_prob_flat.reshape(-1, NUM_CLASSES)

    report = print_eval(y_test, y_pred, "XGBoost")
    report_path = os.path.join(out_dir, "class_report_xgb.txt")
    Path(report_path).write_text(report)

    # -- Comprehensive results save --------------------------------------------
    if save_xgb_results is not None:
        save_xgb_results(
            out_dir      = out_dir,
            y_test       = y_test,
            y_pred       = y_pred,
            y_prob       = y_prob,
            model        = model,
            class_names  = CLASS_NAMES,
            inference_ms = infer_ms,
        )

    # -- Plot training curves --------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        train_loss = evals_result["train"]["mlogloss"]
        val_loss   = evals_result["val"]["mlogloss"]
        plt.figure(figsize=(10, 4))
        plt.plot(train_loss, label="Train mlogloss")
        plt.plot(val_loss,   label="Val mlogloss")
        plt.xlabel("Boosting round")
        plt.ylabel("mlogloss")
        plt.title("XGBoost Training Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "history_xgb.png"), dpi=120)
        plt.close()
    except Exception as e:
        logger.warning("Could not save XGBoost plot: %s", e)

    logger.info("XGBoost training complete.")


# ==============================================================================
#  MODEL 2: 1D Edge-Optimised CNN (MobileNet-style DS-Conv blocks)
# ==============================================================================

def build_edge_cnn(
    num_classes: int = NUM_CLASSES,
    input_length: int = INPUT_LENGTH,
    learning_rate: float = 1e-3,
) -> "tf.keras.Model":
    """
    Build an ultra-lightweight 1D Depthwise-Separable CNN.

    Architecture rationale
    ----------------------
    - Pure depthwise-separable (DS) conv blocks throughout -- each DS block
      replaces a standard Conv with a depthwise conv (K parameters) followed
      by a 1x1 pointwise conv, reducing FLOPs by 8-9x vs. standard Conv.
    - No recurrent layer -- avoids the memory-bandwidth bottleneck of LSTM
      on ESP32-S3; relies instead on large receptive field via multiple blocks.
    - Aggressive pooling (x4, x4, x4) compresses 1250 -> 19 time steps in
      3 stages while increasing depth, following MobileNet intuition.
    - Global Average Pooling instead of Flatten reduces parameters ~50x vs.
      a Dense head applied directly to the conv output.

    Network Profile (approximate)
    ----------------------------
    Parameters : ~45,000
    FP32 size  : ~180 KB
    INT8 size  : ~45 KB   <- fits in ESP32-S3 internal SRAM
    """
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    l2 = 1e-4

    inp = tf.keras.Input(shape=(input_length, 1), name="ppg_input")

    # ── Stem: single standard conv (not DS) to lift 1 channel to 32 ----------
    x = layers.Conv1D(32, kernel_size=7, padding="same", use_bias=False,
                      kernel_regularizer=regularizers.l2(l2), name="stem_conv")(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu6", name="stem_relu6")(x)

    # ── DS-Conv blocks --------------------------------------------------------
    # Block config: (filters, kernel_size, pool_size)
    ds_configs = [
        (64,  7, 4),    # 1250 -> 312
        (96,  5, 4),    # 312  -> 78
        (128, 5, 4),    # 78   -> 19
        (128, 3, 1),    # 19   -> 19  (no pool, increases depth)
    ]
    for i, (filters, ks, pool) in enumerate(ds_configs):
        prefix = f"ds{i+1}"
        # Depthwise conv: each channel filtered independently
        x = layers.DepthwiseConv1D(
            kernel_size=ks, padding="same", use_bias=False,
            depthwise_regularizer=regularizers.l2(l2),
            name=f"{prefix}_dw",
        )(x)
        x = layers.BatchNormalization(name=f"{prefix}_dw_bn")(x)
        x = layers.Activation("relu6", name=f"{prefix}_dw_relu")(x)
        # Pointwise conv: mix channels with 1x1 conv
        x = layers.Conv1D(
            filters, kernel_size=1, padding="same", use_bias=False,
            kernel_regularizer=regularizers.l2(l2),
            name=f"{prefix}_pw",
        )(x)
        x = layers.BatchNormalization(name=f"{prefix}_pw_bn")(x)
        x = layers.Activation("relu6", name=f"{prefix}_pw_relu")(x)
        x = layers.SpatialDropout1D(0.1, name=f"{prefix}_sdrop")(x)
        if pool > 1:
            x = layers.MaxPooling1D(pool_size=pool, name=f"{prefix}_pool")(x)

    # ── Global Average Pooling + Classification head --------------------------
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dropout(0.4, name="head_drop")(x)
    # dtype="float32" mandatory for mixed-precision numerical stability
    out = layers.Dense(num_classes, activation="softmax",
                       dtype="float32", name="output")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="edge_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_bilstm(
    num_classes: int = NUM_CLASSES,
    input_length: int = INPUT_LENGTH,
    learning_rate: float = 3e-4,
) -> "tf.keras.Model":
    """
    Build a Hybrid Conv1D + Bidirectional LSTM model.

    Architecture rationale
    ----------------------
    - Conv1D front-end extracts local morphological patterns (peak shapes,
      notches) before the LSTM sees the sequence.
    - Bidirectional LSTM: forward + backward passes double the effective
      context.  Particularly powerful for detecting rhythm irregularities
      where the pattern is symmetric around an event.
    - Two BiLSTM layers: first returns full sequence for the second to
      consume, giving richer temporal hierarchy.
    - Trade-off: BiLSTM is ~3x heavier than CNN-only for the same accuracy;
      suitable for Tier-2 constrained devices (e.g. with PSRAM).

    Network Profile (approximate)
    ----------------------------
    Parameters : ~180,000
    FP32 size  : ~720 KB
    INT8 size  : ~180 KB
    """
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    l2 = 1e-4

    inp = tf.keras.Input(shape=(input_length, 1), name="ppg_input")

    # ── Conv1D front-end (3 blocks) -------------------------------------------
    x = layers.Conv1D(32, 7, padding="same", use_bias=False,
                      kernel_regularizer=regularizers.l2(l2), name="c1_conv")(inp)
    x = layers.BatchNormalization(name="c1_bn")(x)
    x = layers.Activation("relu", name="c1_relu")(x)
    x = layers.MaxPooling1D(4, name="c1_pool")(x)      # 1250 -> 312

    x = layers.SeparableConv1D(64, 5, padding="same", use_bias=False,
                                name="c2_sep")(x)
    x = layers.BatchNormalization(name="c2_bn")(x)
    x = layers.Activation("relu", name="c2_relu")(x)
    x = layers.MaxPooling1D(4, name="c2_pool")(x)      # 312 -> 78

    x = layers.SeparableConv1D(128, 3, padding="same", use_bias=False,
                                name="c3_sep")(x)
    x = layers.BatchNormalization(name="c3_bn")(x)
    x = layers.Activation("relu", name="c3_relu")(x)
    # No pool here -- keep sequence length=78 for the BiLSTM

    # ── Bidirectional LSTM stack ----------------------------------------------
    # return_sequences=True on first BiLSTM to feed the second
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, name="lstm1"),
        name="bilstm1",
    )(x)
    x = layers.Dropout(0.3, name="d1")(x)

    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=False, name="lstm2"),
        name="bilstm2",
    )(x)
    x = layers.Dropout(0.4, name="d2")(x)

    # ── Classification head ---------------------------------------------------
    x   = layers.Dense(64, activation="relu",
                        kernel_regularizer=regularizers.l2(l2), name="dense1")(x)
    x   = layers.Dropout(0.3, name="d3")(x)
    out = layers.Dense(num_classes, activation="softmax",
                       dtype="float32", name="output")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="bilstm_ppg")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_tcn(
    num_classes: int = NUM_CLASSES,
    input_length: int = INPUT_LENGTH,
    learning_rate: float = 1e-4,   # lowered from 3e-4 -- dilated nets need smaller LR
) -> "tf.keras.Model":
    """
    Build a Temporal Convolutional Network (TCN) with dilated causal convolutions.

    Architecture rationale
    ----------------------
    - Dilated causal convolutions: each layer has an exponentially growing
      dilation factor (1, 2, 4, 8, ...), giving a receptive field that grows
      as 2^N without increasing parameters or compute.
    - Residual connections: skip connections (with 1x1 projection if needed)
      stabilise training of deep dilated networks (ResNet-style).
    - No recurrence: every time step can be computed in parallel (unlike LSTM),
      making TCN significantly faster at inference on CPUs/MCUs.
    - Receptive field with 2 stacks of [1,2,4,8] = total dilation 30 per stack.
      With kernel_size=3: RF = 2 * (3-1) * (1+2+4+8) + 1 = 61 samples per stack.
      For 2 stacks: RF ~= 121 samples @ 125 Hz ~= ~1 second.

    Network Profile (approximate)
    ----------------------------
    Parameters : ~120,000
    FP32 size  : ~480 KB
    INT8 size  : ~120 KB
    """
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    l2 = 1e-4
    num_filters  = 64
    kernel_size  = 3
    dilations    = [1, 2, 4, 8]    # one stack; repeat for two stacks
    dropout_rate = 0.2

    def _tcn_residual_block(
        x: "tf.Tensor",
        filters: int,
        kernel_size: int,
        dilation: int,
        block_name: str,
    ) -> "tf.Tensor":
        """
        TCN residual block:
          - Two dilated causal Conv1D layers with layer norm + ReLU + dropout
          - 1x1 Conv projection on the residual if channel dims differ
        """
        residual = x

        # Causal padding: pad (kernel_size - 1) * dilation on the left only
        pad_left = (kernel_size - 1) * dilation

        # Layer 1
        x = layers.ZeroPadding1D(padding=(pad_left, 0), name=f"{block_name}_pad1")(x)
        x = layers.Conv1D(
            filters, kernel_size, dilation_rate=dilation, padding="valid",
            use_bias=False, kernel_regularizer=regularizers.l2(l2),
            name=f"{block_name}_conv1",
        )(x)
        x = layers.LayerNormalization(name=f"{block_name}_ln1")(x)
        x = layers.Activation("relu", name=f"{block_name}_relu1")(x)
        x = layers.SpatialDropout1D(dropout_rate, name=f"{block_name}_drop1")(x)

        # Layer 2
        x = layers.ZeroPadding1D(padding=(pad_left, 0), name=f"{block_name}_pad2")(x)
        x = layers.Conv1D(
            filters, kernel_size, dilation_rate=dilation, padding="valid",
            use_bias=False, kernel_regularizer=regularizers.l2(l2),
            name=f"{block_name}_conv2",
        )(x)
        x = layers.LayerNormalization(name=f"{block_name}_ln2")(x)
        x = layers.Activation("relu", name=f"{block_name}_relu2")(x)
        x = layers.SpatialDropout1D(dropout_rate, name=f"{block_name}_drop2")(x)

        # Residual projection if channel count changed
        if residual.shape[-1] != filters:
            residual = layers.Conv1D(
                filters, 1, use_bias=False, name=f"{block_name}_res_proj"
            )(residual)

        return layers.Add(name=f"{block_name}_add")([x, residual])

    inp = tf.keras.Input(shape=(input_length, 1), name="ppg_input")

    # ── Stem conv (lift 1 -> num_filters) ------------------------------------
    x = layers.Conv1D(num_filters, 1, padding="same", name="stem_proj",
                      use_bias=False)(inp)

    # ── Two stacks of dilated residual blocks ---------------------------------
    for stack in range(2):
        for d in dilations:
            x = _tcn_residual_block(
                x, num_filters, kernel_size, d,
                block_name=f"tcn_s{stack}_d{d}",
            )

    # ── Classification head ---------------------------------------------------
    # Slice the last time step (causal network: last step has full context)
    x   = layers.Lambda(lambda t: t[:, -1, :], name="last_step")(x)
    x   = layers.Dropout(0.4, name="head_drop")(x)
    x   = layers.Dense(64, activation="relu",
                        kernel_regularizer=regularizers.l2(l2), name="dense1")(x)
    x   = layers.Dropout(0.3, name="d2")(x)
    out = layers.Dense(num_classes, activation="softmax",
                       dtype="float32", name="output")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="tcn_ppg")
    model.compile(
        # Gradient clipping (clipnorm=1.0) is critical for TCN stability:
        # dilated residual networks can produce large gradients early in
        # training, causing the network to collapse to predicting the
        # majority class. Clipping limits gradient norm to 1.0 per step.
        optimizer=tf.keras.optimizers.Adam(
            learning_rate,
            clipnorm=1.0,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ==============================================================================
#  GENERIC KERAS TRAINING LOOP
# ==============================================================================

def train_keras_model(
    model: "tf.keras.Model",
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    out_dir:    str,
    model_name: str,
    epochs:     int = 100,
    batch_size: int = 64,
    seed:       int = 42,
) -> None:
    """
    Generic training loop for Keras models (CNN, BiLSTM, TCN).

    Trains with:
    - Class-weighted loss (handles severe class imbalance)
    - ReduceLROnPlateau + EarlyStopping callbacks
    - ModelCheckpoint on val_loss
    - TensorBoard logging per model
    """
    import tensorflow as tf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model.summary(print_fn=logger.info)
    total_params = model.count_params()
    logger.info("  Total parameters : %d  (~%.0f KB float32)",
                total_params, total_params * 4 / 1024)

    cw    = compute_class_weights_dict(y_train)
    logger.info("  Class weights: %s", cw)

    # ── Remove zero-support classes from class_weight dict ────────────────────
    # Classes present in y_train weights but absent from y_test cause
    # misleading 0.0 F1 in the macro average. Skip them in class_weight.
    present_classes = set(np.unique(y_train))
    cw_filtered = {k: v for k, v in cw.items() if k in present_classes}
    logger.info("  Present classes in train: %s", sorted(present_classes))

    ckpt_path  = os.path.join(out_dir, f"best_{model_name}.h5")
    tb_log_dir = os.path.join(out_dir, "logs", model_name)

    # ── MacroF1 callback ──────────────────────────────────────────────────────
    # val_loss is dominated by the majority class (Normal ~69%); EarlyStopping
    # on val_loss will stop training as soon as Normal is well-predicted,
    # ignoring minority classes. Monitoring val_macro_f1 keeps training
    # going until ALL classes improve.
    # tf is already imported via `import tensorflow as tf` above in this function
    class MacroF1Callback(tf.keras.callbacks.Callback):
        """Compute macro F1 on validation set each epoch and log as val_macro_f1."""
        def __init__(self, X_val, y_val, batch_size):
            super().__init__()
            self.X_val = X_val
            self.y_val = y_val
            self.bs    = batch_size

        def on_epoch_end(self, epoch, logs=None):
            import gc
            from sklearn.metrics import f1_score
            # Clear GPU memory fragmentation that builds up over many epochs.
            # The "Dst tensor is not initialized" error is caused by GPU OOM
            # when trying to copy validation data for predict().
            tf.keras.backend.clear_session() if False else None  # keep weights
            gc.collect()
            try:
                # Use a smaller batch size for prediction to reduce peak memory.
                pred_bs = max(16, self.bs // 4)
                y_prob = self.model.predict(self.X_val, batch_size=pred_bs, verbose=0)
            except Exception as e:
                # Fallback: run predict on CPU to avoid GPU OOM entirely.
                logger.warning("GPU predict failed (%s), falling back to CPU.", e)
                with tf.device("/CPU:0"):
                    y_prob = self.model.predict(self.X_val, batch_size=16, verbose=0)
            y_pred = np.argmax(y_prob, axis=1)
            f1 = f1_score(self.y_val, y_pred, average="macro", zero_division=0)
            logs["val_macro_f1"] = float(f1)
            logger.info("  epoch %d  val_macro_f1=%.4f", epoch + 1, f1)

    callbacks = [
        MacroF1Callback(X_val, y_val, batch_size),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=20,                 # raised: minority classes need more epochs
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_macro_f1",
            mode="max",
            factor=0.5,
            patience=7,                  # raised: avoid premature LR decay
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=tb_log_dir, update_freq="epoch", profile_batch=0,
        ),
    ]

    logger.info("Training %s for up to %d epochs ...", model_name, epochs)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        class_weight=cw_filtered,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate on test set --------------------------------------------------
    import time as _time
    best_model = tf.keras.models.load_model(ckpt_path)
    t0     = _time.perf_counter()
    y_prob = best_model.predict(X_test, batch_size=batch_size * 2, verbose=0)
    infer_ms = (_time.perf_counter() - t0) * 1000.0
    y_pred = np.argmax(y_prob, axis=1).astype(np.int32)

    # Only report classes that actually appear in the test set
    test_classes = sorted(np.unique(y_test).tolist())
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(y_test, y_pred, average="macro",
                        labels=test_classes, zero_division=0)
    logger.info("  Test Macro F1 (present classes only): %.4f", macro_f1)

    report = print_eval(y_test, y_pred, model_name)
    Path(os.path.join(out_dir, f"class_report_{model_name}.txt")).write_text(report)

    # ── Comprehensive results save --------------------------------------------
    if save_all_results is not None:
        save_all_results(
            model_name   = model_name,
            out_dir      = out_dir,
            y_test       = y_test,
            y_pred       = y_pred,
            y_prob       = y_prob,
            history      = history.history,
            model        = best_model,
            class_names  = CLASS_NAMES,
            inference_ms = infer_ms,
        )

    # ── Plot training curves (legacy simple plot kept for quick viewing) -------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{model_name} -- Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["accuracy"],     label="Train Acc")
    axes[1].plot(history.history["val_accuracy"], label="Val Acc")
    axes[1].set_title(f"{model_name} -- Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"history_{model_name}.png"), dpi=120)
    plt.close()

    logger.info("%s training complete. Saved -> %s", model_name, ckpt_path)


# ==============================================================================
#  MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train multiple PPG arrhythmia model architectures"
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=["xgb", "cnn", "bilstm", "tcn"],
        default=["xgb", "cnn", "bilstm", "tcn"],
        help="Which models to train (default: all)",
    )
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--no-gpu",     action="store_true",
                        help="Force CPU-only training")
    parser.add_argument("--no-mixed-precision", action="store_true",
                        help="Disable float16 mixed precision")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── GPU setup (only needed for Keras models) ------------------------------
    if any(m in args.models for m in ["cnn", "bilstm", "tcn"]):
        from gpu_config import setup_gpu
        setup_gpu(
            mixed_precision=not args.no_mixed_precision,
            no_gpu=args.no_gpu,
        )

    # ── XGBoost on extracted features ----------------------------------------
    if "xgb" in args.models:
        logger.info("\n%s\nMODEL 1: XGBoost\n%s", "=" * 60, "=" * 60)
        X_tr, y_tr, X_v, y_v, X_te, y_te = load_feature_dataset()
        train_xgboost(
            X_tr, y_tr, X_v, y_v, X_te, y_te,
            out_dir=os.path.join(OUTPUT_DIR, "xgb"),
            seed=args.seed,
        )

    # Load raw signals for all DL models (once, shared)
    dl_models_requested = [m for m in args.models if m in ("cnn", "bilstm", "tcn")]
    if dl_models_requested:
        X_train, y_train, X_val, y_val, X_test, y_test = load_raw_dataset()

    # ── Edge CNN -------------------------------------------------------------
    if "cnn" in args.models:
        logger.info("\n%s\nMODEL 2: Edge CNN (DS-Conv)\n%s", "=" * 60, "=" * 60)
        import tensorflow as tf
        with tf.distribute.MirroredStrategy().scope():
            model = build_edge_cnn()
        train_keras_model(
            model, X_train, y_train, X_val, y_val, X_test, y_test,
            out_dir=os.path.join(OUTPUT_DIR, "edge_cnn"),
            model_name="edge_cnn",
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )

    # ── Bidirectional LSTM ---------------------------------------------------
    if "bilstm" in args.models:
        logger.info("\n%s\nMODEL 3: BiLSTM\n%s", "=" * 60, "=" * 60)
        import tensorflow as tf
        with tf.distribute.MirroredStrategy().scope():
            model = build_bilstm()
        train_keras_model(
            model, X_train, y_train, X_val, y_val, X_test, y_test,
            out_dir=os.path.join(OUTPUT_DIR, "bilstm"),
            model_name="bilstm",
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )

    # ── TCN ------------------------------------------------------------------
    if "tcn" in args.models:
        logger.info("\n%s\nMODEL 4: TCN\n%s", "=" * 60, "=" * 60)
        import tensorflow as tf
        with tf.distribute.MirroredStrategy().scope():
            model = build_tcn()
        train_keras_model(
            model, X_train, y_train, X_val, y_val, X_test, y_test,
            out_dir=os.path.join(OUTPUT_DIR, "tcn"),
            model_name="tcn",
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )

    logger.info("\nAll requested models trained. Results in: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
