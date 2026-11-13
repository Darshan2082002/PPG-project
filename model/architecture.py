"""
architecture.py
===============
Lightweight Depthwise-Separable 1D-CNN + LSTM model for PPG arrhythmia detection.

GPU / Mixed-Precision Notes
---------------------------
* The global mixed-precision policy (``mixed_float16``) is applied by
  ``gpu_config.setup_gpu()`` *before* this module is imported.  All layer
  compute therefore runs in float16; Keras stores weights in float32.
* The final Dense output is **explicitly cast to float32** via a ``tf.cast``
  Lambda layer.  This is mandatory: softmax cross-entropy numerically
  requires float32 logits/probabilities.
* ``recurrent_dropout`` has been replaced with a standard Dropout layer
  placed *after* the LSTM.  LSTM ``recurrent_dropout`` is incompatible with
  XLA / ``jit_compile=True``, which is used in the training loop.

Network Profile (approximate, float32 weights):
  Parameters : ~83 K
  Model size  : ~325 KB (float32) → ~82 KB (INT8 TFLite)
  Input shape : (1250, 1)  –  10 s @ 125 Hz
  Output      : Softmax over 5 arrhythmia classes (float32)
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

NUM_CLASSES    = 5
INPUT_LENGTH   = 1250   # samples  (10 s × 125 Hz)
INPUT_CHANNELS = 1


def _dw_sep_conv1d_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    pool_size: int = 4,
    name_prefix: str = "",
    l2: float = 1e-4,
    spatial_dropout: float = 0.1,
) -> tf.Tensor:
    """
    Depthwise-Separable 1-D convolution block.

    Flow: SeparableConv1D → BatchNorm → ReLU → SpatialDropout1D → MaxPool1D

    L2 regularization + SpatialDropout work together to reduce the severe
    overfitting observed in the training history (train loss 0.05 vs val 0.33).
    """
    x = layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        use_bias=False,
        depthwise_regularizer=regularizers.l2(l2),
        pointwise_regularizer=regularizers.l2(l2),
        name=f"{name_prefix}_sepconv",
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu")(x)
    # SpatialDropout1D drops entire feature-map channels — more effective
    # than standard Dropout for conv outputs
    x = layers.SpatialDropout1D(spatial_dropout, name=f"{name_prefix}_sdrop")(x)
    x = layers.MaxPooling1D(pool_size=pool_size, name=f"{name_prefix}_pool")(x)
    return x


def build_model(
    num_classes: int = NUM_CLASSES,
    input_length: int = INPUT_LENGTH,
    lstm_units: int = 64,
    dense_units: int = 64,
    dropout_rate: float = 0.5,       # increased 0.3→0.5 to counter overfitting
    l2_reg: float = 1e-4,            # L2 weight decay on conv + dense
    learning_rate: float = 3e-4,     # lowered 1e-3→3e-4 for stable convergence
) -> keras.Model:
    """
    Build and compile the PPG arrhythmia detection model.

    Architecture
    ------------
    Input(1250, 1)
    → Conv1D(32, k=7)  + BN + ReLU + SpatialDropout(0.1)
    → DS-Conv1D(64,  k=7) + BN + ReLU + SDrop + MaxPool(4)   1250→312
    → DS-Conv1D(128, k=5) + BN + ReLU + SDrop + MaxPool(4)    312→78
    → DS-Conv1D(128, k=5) + BN + ReLU + SDrop + MaxPool(4)     78→19
    → LSTM(64)
    → Dropout(0.5)          ← replaces recurrent_dropout for XLA compatibility
    → Dense(64, ReLU, L2)
    → Dropout(0.5)
    → Dense(5, Softmax, dtype=float32)   ← explicit fp32 cast for mixed precision

    Parameters
    ----------
    num_classes   : number of output classes
    input_length  : number of time-steps in input window
    lstm_units    : LSTM hidden units
    dense_units   : dense layer width
    dropout_rate  : dropout probability (applied after LSTM and Dense)
    l2_reg        : L2 regularization coefficient
    learning_rate : initial Adam learning rate
    """
    inputs = keras.Input(
        shape=(input_length, INPUT_CHANNELS), name="ppg_input"
    )

    # ── Initial standard conv ─────────────────────────────────────────────────
    x = layers.Conv1D(
        32,
        kernel_size=7,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(l2_reg),
        name="conv0",
    )(inputs)
    x = layers.BatchNormalization(name="conv0_bn")(x)
    x = layers.Activation("relu", name="conv0_relu")(x)
    x = layers.SpatialDropout1D(0.1, name="conv0_sdrop")(x)

    # ── Depthwise-separable blocks ────────────────────────────────────────────
    x = _dw_sep_conv1d_block(x, filters=64,  kernel_size=7, pool_size=4,
                              name_prefix="dsconv1", l2=l2_reg)
    x = _dw_sep_conv1d_block(x, filters=128, kernel_size=5, pool_size=4,
                              name_prefix="dsconv2", l2=l2_reg)
    x = _dw_sep_conv1d_block(x, filters=128, kernel_size=5, pool_size=4,
                              name_prefix="dsconv3", l2=l2_reg)

    # ── Temporal context (LSTM) ───────────────────────────────────────────────
    # NOTE: recurrent_dropout removed — it breaks XLA jit_compile.
    # A standard Dropout after the LSTM achieves equivalent regularization.
    x = layers.LSTM(lstm_units, name="lstm")(x)
    x = layers.Dropout(dropout_rate, name="lstm_dropout")(x)  # XLA-compatible

    # ── Classification head ───────────────────────────────────────────────────
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense1",
    )(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)

    # CRITICAL for mixed precision: the output MUST be float32.
    # Keras mixed_float16 policy makes Dense compute in float16 by default,
    # which causes numerical instability in softmax + cross-entropy.
    # dtype="float32" on the last Dense overrides the global policy.
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",   # ← mandatory for mixed-precision stability
        name="output",
    )(x)

    model = keras.Model(
        inputs=inputs, outputs=outputs, name="ppg_arrhythmia_cnn_lstm"
    )

    # Use a LossScaleOptimizer wrapper automatically applied by Keras when
    # mixed_float16 policy is active — no manual wrapping required in TF 2.x
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def count_params(model: keras.Model) -> None:
    """Print a concise parameter and memory summary."""
    total     = model.count_params()
    trainable = sum(tf.size(v).numpy() for v in model.trainable_variables)
    print(f"\n{'='*45}")
    print(f"  Model        : {model.name}")
    print(f"  Total params : {total:,}")
    print(f"  Trainable    : {trainable:,}")
    print(f"  Float32 size : ~{total * 4 / 1024:.0f} KB")
    print(f"  INT8 size    : ~{total / 1024:.0f} KB")
    print(f"{'='*45}\n")


if __name__ == "__main__":
    model = build_model()
    model.summary()
    count_params(model)
