"""
Training Entry Point for ISL Gesture Recognition.

Loads collected gesture data, prepares features, trains the LSTM model,
and saves the model + vocabulary mapping.

Usage:
    python train.py
    python train.py --augment          # With data augmentation
    python train.py --velocity         # Include velocity features
    python train.py --epochs 150       # Custom epoch count
"""

import argparse
import os
import json
import numpy as np
from sklearn.utils import class_weight
import config
from src.dataset import load_dataset, save_dataset_npz, load_dataset_npz
from src.feature_engineer import build_feature_vector, normalize_sequence
from src.model import build_model, train_model, save_model, plot_training_history
from src.utils import load_vocabulary, save_vocabulary
from src.augment_landmarks import augment_sequence


def augment_sequences(X: np.ndarray, y: np.ndarray, 
                      augment_factor: int = 5) -> tuple:
    """
    Augment training data using landmark-level augmentations.
    
    Uses noise, scale jitter, 2D rotation, time warping, landmark 
    dropout, and hand swapping to simulate signer variation.
    
    Args:
        X: Original sequences (N, seq_len, features).
        y: Original labels (N,).
        augment_factor: Augmented copies per original sample.
    
    Returns:
        Tuple of (augmented_X, augmented_y).
    """
    X_aug = [X]
    y_aug = [y]
    
    for i in range(augment_factor):
        # Increase intensity gradually for more diversity
        intensity = 0.5 + (i / augment_factor) * 1.0
        
        batch = []
        for seq in X:
            aug = augment_sequence(seq, intensity=intensity)
            batch.append(aug)
        
        X_aug.append(np.array(batch, dtype=np.float32))
        y_aug.append(y)
    
    return np.concatenate(X_aug), np.concatenate(y_aug)


def main():
    parser = argparse.ArgumentParser(description="Train ISL Gesture Recognition Model")
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--velocity', action='store_true', help='Include velocity features')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--reload', action='store_true', help='Reload from raw data (ignore cached .npz)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  ISL Gesture Recognition — Model Training")
    print("=" * 60)
    
    # ── Load dataset ─────────────────────────────────────────────────────────
    X, y, label_names = None, None, None
    
    if not args.reload and os.path.exists(config.DATASET_PATH):
        print("\n[STEP 1] Loading cached dataset...")
        X, y, label_names = load_dataset_npz()
    
    if X is None:
        print("\n[STEP 1] Loading data from raw samples...")
        X, y, label_names = load_dataset()
        
        if X is None:
            print("\n[ERROR] No training data found!")
            print("  Run data collection first:")
            print("  python main.py --mode collect --word HELLO")
            print("  python main.py --mode collect --word WATER")
            print("  (collect at least 3 different words)")
            return
        
        # Cache for faster future loading
        save_dataset_npz(X, y, label_names)
    
    num_classes = len(label_names)
    
    if num_classes < 2:
        print("\n[ERROR] Need at least 2 different words to train!")
        print("  Collect data for more words using:")
        print("  python main.py --mode collect --word <WORD>")
        return
    
    # ── Data augmentation ────────────────────────────────────────────────────
    if args.augment:
        print("\n[STEP 2] Augmenting data...")
        original_count = len(X)
        X, y = augment_sequences(X, y)
        print(f"  Augmented: {original_count} → {len(X)} samples")
    
    # ── Add velocity features ────────────────────────────────────────────────
    num_features = X.shape[2]
    
    if args.velocity:
        print("\n[STEP 3] Adding velocity features...")
        X_enriched = np.array([build_feature_vector(seq) for seq in X], dtype=np.float32)
        num_features = X_enriched.shape[2]
        print(f"  Feature dimensions: {config.NUM_FEATURES} → {num_features}")
        X = X_enriched
    
    # ── Update config for epochs/batch ───────────────────────────────────────
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    
    # ── Build and train model ────────────────────────────────────────────────
    print(f"\n[STEP 4] Building LSTM model...")
    model = build_model(num_features, num_classes)
    
    print(f"\n[STEP 5] Training for {args.epochs} epochs...")
    model, history = train_model(X, y, num_classes, model)
    
    # ── Save model ───────────────────────────────────────────────────────────
    print("\n[STEP 6] Saving model and vocabulary...")
    save_model(model)
    
    # Update vocabulary with actual training labels
    vocab = {
        "words": label_names,
        "word_to_index": {w: i for i, w in enumerate(label_names)},
        "index_to_word": {str(i): w for i, w in enumerate(label_names)},
        "use_velocity": args.velocity
    }
    save_vocabulary(vocab)
    print(f"  Vocabulary: {label_names}")
    
    # ── Plot training history ────────────────────────────────────────────────
    print("\n[STEP 7] Generating training plots...")
    plot_training_history(history)
    
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  Model saved to: {config.MODEL_PATH}")
    print(f"  Vocabulary: {num_classes} words")
    print("  Run recognition: python main.py --mode recognize")
    print("=" * 60)


if __name__ == "__main__":
    main()
