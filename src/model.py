"""
LSTM Gesture Recognition Model.

Defines, trains, and manages the deep learning model for
ISL word-level gesture classification.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import config


def build_model(num_features: int, num_classes: int, 
                seq_length: int = None) -> keras.Model:
    """
    Build the LSTM gesture recognition model.
    
    Architecture:
        Input → LSTM(128, return_sequences) → Dropout(0.3)
              → LSTM(64) → Dropout(0.3)
              → Dense(64, relu) → Dropout(0.3)
              → Dense(num_classes, softmax)
    
    Args:
        num_features: Number of input features per frame.
        num_classes: Number of gesture classes (words).
        seq_length: Sequence length. Defaults to config.SEQUENCE_LENGTH.
    
    Returns:
        Compiled Keras model.
    """
    if seq_length is None:
        seq_length = config.SEQUENCE_LENGTH
    
    model = keras.Sequential([
        layers.Input(shape=(seq_length, num_features)),
        
        # First LSTM layer — returns sequences for stacking
        layers.LSTM(config.LSTM_UNITS_1, return_sequences=True,
                    activation='tanh', recurrent_activation='sigmoid'),
        layers.BatchNormalization(),
        layers.Dropout(config.DROPOUT_RATE),
        
        # Second LSTM layer — returns final hidden state
        layers.LSTM(config.LSTM_UNITS_2, return_sequences=False,
                    activation='tanh', recurrent_activation='sigmoid'),
        layers.BatchNormalization(),
        layers.Dropout(config.DROPOUT_RATE),
        
        # Dense classification layers
        layers.Dense(config.DENSE_UNITS, activation='relu'),
        layers.Dropout(config.DROPOUT_RATE),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model


def train_model(X: np.ndarray, y: np.ndarray, num_classes: int,
                model: keras.Model = None, validation_split: float = 0.2):
    """
    Train the gesture recognition model.
    
    Uses early stopping and learning rate reduction on plateau.
    
    Args:
        X: Training sequences of shape (num_samples, seq_length, num_features).
        y: Integer labels of shape (num_samples,).
        num_classes: Number of classes.
        model: Pre-built model, or None to build a new one.
        validation_split: Fraction of data for validation.
    
    Returns:
        Tuple of (trained_model, training_history).
    """
    num_features = X.shape[2]
    
    if model is None:
        model = build_model(num_features, num_classes)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=42, stratify=y
    )
    
    print(f"\n[INFO] Training set: {X_train.shape[0]} samples")
    print(f"[INFO] Validation set: {X_val.shape[0]} samples")
    print(f"[INFO] Number of classes: {num_classes}")
    print(f"[INFO] Feature dimensions: {num_features}")
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            config.MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n[RESULT] Best validation accuracy: {val_acc:.4f}")
    print(f"[RESULT] Best validation loss: {val_loss:.4f}")
    
    return model, history


def save_model(model: keras.Model, path: str = None):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained Keras model.
        path: Save path. Defaults to config.MODEL_PATH.
    """
    if path is None:
        path = config.MODEL_PATH
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"[INFO] Model saved to {path}")


def load_model(path: str = None) -> keras.Model:
    """
    Load a trained model from disk.
    
    Args:
        path: Model path. Defaults to config.MODEL_PATH.
    
    Returns:
        Loaded Keras model.
    """
    if path is None:
        path = config.MODEL_PATH
    
    if not os.path.exists(path):
        print(f"[ERROR] Model not found at {path}")
        return None
    
    model = keras.models.load_model(path)
    print(f"[INFO] Model loaded from {path}")
    return model


def plot_training_history(history):
    """
    Plot training/validation accuracy and loss curves.
    
    Args:
        history: Keras training history object.
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        plot_path = os.path.join(config.BASE_DIR, "models", "training_history.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Training plot saved to {plot_path}")
        
        plt.show()
    
    except ImportError:
        print("[WARNING] matplotlib not available, skipping plot.")
