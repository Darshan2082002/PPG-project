"""
architecture.py
===============
Lightweight Depthwise-Separable 1D-CNN + LSTM model for PPG arrhythmia detection.

Network Profile (approximate):
  Parameters : ~180K
  Model size  : ~720 KB (float32) → ~180 KB (INT8)
  Input shape : (1250, 1)  –  10 s @ 125 Hz
  Output      : Softmax over 5 arrhythmia classes
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_CLASSES   = 5
INPUT_LENGTH  = 1250   # samples  (10 s × 125 Hz)
INPUT_CHANNELS = 1


def _dw_sep_conv1d_block(
    x,
    filters: int,
    kernel_size: int,
    pool_size: int = 4,
    name_prefix: str = "",
) -> tf.Tensor:
    """
    Depthwise-separable 1-D convolution block:
      DepthwiseConv → BatchNorm → ReLU
      PointwiseConv → BatchNorm → ReLU
      MaxPool
    """
    # Pointwise projection first to avoid channel mismatch in depthwise
    x = layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        use_bias=False,
        name=f"{name_prefix}_sepconv",
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu")(x)
    x = layers.MaxPooling1D(pool_size=pool_size, name=f"{name_prefix}_pool")(x)
    return x


def build_model(
    num_classes: int = NUM_CLASSES,
    input_length: int = INPUT_LENGTH,
    lstm_units: int = 64,
    dense_units: int = 64,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """
    Build and return the compiled model.

    Architecture:
      Input(1250, 1)
      → Conv1D(32, k=7) + BN + ReLU              ← initial feature extraction
      → DS-Conv1D(64,  k=7) + BN + ReLU + Pool4  ← 1250 → 312
      → DS-Conv1D(128, k=5) + BN + ReLU + Pool4  ← 312  → 78
      → DS-Conv1D(128, k=5) + BN + ReLU + Pool4  ← 78   → 19
      → LSTM(64)                                  ← temporal context
      → Dense(64, ReLU) + Dropout(0.3)
      → Dense(num_classes, Softmax)

    Parameters
    ----------
    num_classes   : number of output classes
    input_length  : number of time-steps in input window
    lstm_units    : LSTM hidden units
    dense_units   : dense layer width
    dropout_rate  : dropout probability

    Returns
    -------
    model : compiled keras.Model
    """
    inputs = keras.Input(shape=(input_length, INPUT_CHANNELS), name="ppg_input")

    # ── Initial standard conv ─────────────────────────────────────────────────
    x = layers.Conv1D(32, kernel_size=7, padding="same", use_bias=False,
                      name="conv0")(inputs)
    x = layers.BatchNormalization(name="conv0_bn")(x)
    x = layers.Activation("relu", name="conv0_relu")(x)

    # ── Depthwise-separable blocks ────────────────────────────────────────────
    x = _dw_sep_conv1d_block(x, filters=64,  kernel_size=7, pool_size=4, name_prefix="dsconv1")
    x = _dw_sep_conv1d_block(x, filters=128, kernel_size=5, pool_size=4, name_prefix="dsconv2")
    x = _dw_sep_conv1d_block(x, filters=128, kernel_size=5, pool_size=4, name_prefix="dsconv3")

    # ── Temporal context (LSTM) ───────────────────────────────────────────────
    x = layers.LSTM(lstm_units, name="lstm")(x)

    # ── Classification head ───────────────────────────────────────────────────
    x = layers.Dense(dense_units, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ppg_arrhythmia_cnn_lstm")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def count_params(model: keras.Model) -> None:
    """Print a concise parameter summary."""
    total    = model.count_params()
    trainable = sum(tf.size(v).numpy() for v in model.trainable_variables)
    print(f"\n{'='*40}")
    print(f"  Model: {model.name}")
    print(f"  Total parameters    : {total:,}")
    print(f"  Trainable           : {trainable:,}")
    print(f"  Float32 size        : ~{total * 4 / 1024:.0f} KB")
    print(f"  INT8 estimate       : ~{total / 1024:.0f} KB")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    model = build_model()
    model.summary()
    count_params(model)
