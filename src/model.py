"""
Gesture Recognition Models.

Defines three model architectures:
  - LSTM  (default, best for temporal sequences)
  - TCN   (1D-CNN with dilated convolutions, fast & accurate)
  - MLP   (legacy flat model, kept for comparison)

All models are compiled with label smoothing and Adam.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import config


# ─────────────────────────────────────────────────────────────────────────────
# Architecture builders
# ─────────────────────────────────────────────────────────────────────────────

def build_lstm_model(num_features: int, num_classes: int,
                     seq_length: int = None) -> keras.Model:
    """
    Bidirectional LSTM for temporal gesture sequences.

    Architecture:
        Input(seq_length, num_features)
        → Masking(0.0)
        → BiLSTM(128) → Dropout(0.3)
        → LSTM(64)    → Dropout(0.2)
        → Dense(64, relu) → Dropout(0.2)
        → Dense(num_classes, softmax)
    """
    if seq_length is None:
        seq_length = config.SEQUENCE_LENGTH

    inputs = keras.Input(shape=(seq_length, num_features), name="landmarks")

    # Mask zero-padded frames (missing hands)
    x = layers.Masking(mask_value=0.0)(inputs)

    # Bidirectional LSTM captures both forward and backward temporal patterns
    x = layers.Bidirectional(
        layers.LSTM(config.LSTM_UNITS_1, return_sequences=True,
                    dropout=0.1, recurrent_dropout=0.1)
    )(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)

    x = layers.LSTM(config.LSTM_UNITS_2, return_sequences=False,
                    dropout=0.1, recurrent_dropout=0.1)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(config.DENSE_UNITS, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="ISL_BiLSTM")
    return model


def build_tcn_model(num_features: int, num_classes: int,
                    seq_length: int = None) -> keras.Model:
    """
    Temporal Convolutional Network (1D-CNN with dilated convolutions).

    Faster to train than LSTM, comparable accuracy on fixed-length sequences.

    Architecture:
        Input → [Conv1D(dilation=1) → BN → ReLU → Dropout] × 4 stacked
              → GlobalAvgPool1D → Dense(64) → Dense(num_classes)
    """
    if seq_length is None:
        seq_length = config.SEQUENCE_LENGTH

    filters = config.TCN_FILTERS
    kernel = config.TCN_KERNEL_SIZE

    inputs = keras.Input(shape=(seq_length, num_features), name="landmarks")
    x = inputs

    # Stack dilated causal convolutions (receptive field grows exponentially)
    for dilation in [1, 2, 4, 8]:
        residual = x
        x = layers.Conv1D(filters, kernel, padding="causal",
                          dilation_rate=dilation, activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.2)(x)

        # 1×1 conv for residual when shape changes
        if residual.shape[-1] != filters:
            residual = layers.Conv1D(filters, 1, padding="same")(residual)
        x = layers.Add()([x, residual])

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(config.DENSE_UNITS, activation="relu")(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="ISL_TCN")
    return model


def build_mlp_model(num_features: int, num_classes: int,
                    seq_length: int = None) -> keras.Model:
    """
    Legacy flat MLP model (kept for comparison / ablation).

    Input → Flatten → Dense(128) → Dropout → Dense(64) → Dense(num_classes)
    """
    if seq_length is None:
        seq_length = config.SEQUENCE_LENGTH

    model = keras.Sequential([
        layers.Input(shape=(seq_length, num_features)),
        layers.Flatten(),
        layers.Dense(config.MLP_UNITS_1, activation="relu"),
        layers.Dropout(config.DROPOUT_RATE),
        layers.Dense(config.MLP_UNITS_2, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="ISL_MLP")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_features: int, num_classes: int,
                seq_length: int = None,
                model_type: str = None) -> keras.Model:
    """
    Build and compile the gesture recognition model.

    Args:
        num_features:  Feature dimension per frame.
        num_classes:   Number of gesture classes.
        seq_length:    Sequence length (defaults to config.SEQUENCE_LENGTH).
        model_type:    'lstm' | 'tcn' | 'mlp' (defaults to config.MODEL_TYPE).

    Returns:
        Compiled Keras model.
    """
    if seq_length is None:
        seq_length = config.SEQUENCE_LENGTH
    if model_type is None:
        model_type = config.MODEL_TYPE

    if model_type == "lstm":
        model = build_lstm_model(num_features, num_classes, seq_length)
    elif model_type == "tcn":
        model = build_tcn_model(num_features, num_classes, seq_length)
    elif model_type == "mlp":
        model = build_mlp_model(num_features, num_classes, seq_length)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose: lstm, tcn, mlp")

    # Label smoothing reduces overconfidence on small datasets
    loss = keras.losses.CategoricalCrossentropy(
        label_smoothing=config.LABEL_SMOOTHING
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=loss,
        metrics=["accuracy"],
    )

    model.summary()
    print(f"\n[INFO] Model type: {model_type.upper()}")
    print(f"[INFO] Parameters: {model.count_params():,}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X: np.ndarray, y: np.ndarray, num_classes: int,
                model: keras.Model = None,
                validation_split: float = 0.2,
                X_val_override: np.ndarray = None,
                y_val_override: np.ndarray = None):
    """
    Train the gesture recognition model.

    If X_val_override / y_val_override are provided (pre-split unaugmented
    validation data), they are used as-is. Otherwise an internal stratified
    split of X, y is performed.

    Args:
        X:               Training sequences (N, seq_length, num_features).
        y:               Integer labels (N,).
        num_classes:     Number of classes.
        model:           Pre-built model, or None to build a new one.
        validation_split: Fraction held out (only used when no override).
        X_val_override:  External validation features (unaugmented).
        y_val_override:  External validation labels.

    Returns:
        Tuple of (trained_model, training_history).
    """
    from sklearn.utils import class_weight
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split as tts

    num_features = X.shape[2]

    if model is None:
        model = build_model(num_features, num_classes)

    # Use pre-split validation data when provided (preferred: unaugmented hold-out)
    if X_val_override is not None and y_val_override is not None:
        X_train, y_train = X, y
        X_val, y_val = X_val_override, y_val_override
    else:
        # !! CRITICAL: split on ORIGINAL data BEFORE any augmentation !!
        X_train, X_val, y_train, y_val = tts(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )

    # Class weights for residual imbalance
    present_classes = np.unique(y_train)
    class_weights = class_weight.compute_class_weight(
        "balanced",
        classes=present_classes,
        y=y_train,
    )
    # Build full dict for ALL num_classes (1.0 for any class not in training)
    class_weights_dict = {i: 1.0 for i in range(num_classes)}
    for cls_idx, w in zip(present_classes, class_weights):
        class_weights_dict[int(cls_idx)] = float(w)

    print(f"\n[INFO] Hold-out validation set: {X_val.shape[0]} ORIGINAL samples (no augmentation)")
    print(f"[INFO] Training set (before aug): {X_train.shape[0]} samples")
    print(f"[INFO] Classes:                   {num_classes}")
    print(f"[INFO] Features per frame:        {num_features}")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=30,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            config.MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
    print(f"\n[RESULT] Best validation accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")
    print(f"[RESULT] Best validation loss:     {val_loss:.4f}")

    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model: keras.Model, path: str = None):
    if path is None:
        path = config.MODEL_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"[INFO] Model saved to {path}")


def load_model(path: str = None) -> keras.Model:
    if path is None:
        path = config.MODEL_PATH
    if not os.path.exists(path):
        print(f"[ERROR] Model not found at {path}")
        return None
    model = keras.models.load_model(path)
    print(f"[INFO] Model loaded from {path}")
    return model


def plot_training_history(history):
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(history.history["accuracy"], label="Train", linewidth=2)
        ax1.plot(history.history["val_accuracy"], label="Val", linewidth=2)
        ax1.axhline(0.9, color="red", linestyle="--", alpha=0.6, label="90% target")
        ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(history.history["loss"], label="Train", linewidth=2)
        ax2.plot(history.history["val_loss"], label="Val", linewidth=2)
        ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(config.BASE_DIR, "models", "training_history.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Training plot saved to {plot_path}")
        plt.show()

    except ImportError:
        print("[WARNING] matplotlib not available, skipping plot.")
