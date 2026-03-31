"""
Training Entry Point for ISL Gesture Recognition.

Loads collected gesture data, balances the dataset, augments it,
trains an LSTM/TCN/MLP model, and saves the model + vocabulary.

Usage:
    python train.py                                   # LSTM, velocity, balance+augment
    python train.py --model-type tcn                  # TCN architecture
    python train.py --model-type mlp                  # Legacy MLP
    python train.py --no-balance                      # Skip dataset balancing
    python train.py --no-augment                      # Skip augmentation
    python train.py --no-velocity                     # No velocity features
    python train.py --epochs 150 --batch-size 16      # Custom hyperparams
    python train.py --kfold 5                         # K-Fold cross-validation
    python train.py --reload                          # Force reload from raw .npy
"""

import argparse
import os
import json
import numpy as np
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import config
from src.dataset import load_dataset, save_dataset_npz, load_dataset_npz
from src.dataset import normalize_labels, filter_labels
from src.feature_engineer import build_feature_vector, normalize_hands_sequence
from src.model import build_model, train_model, save_model, plot_training_history
from src.augment_landmarks import augment_sequence
from src.utils import load_vocabulary, save_vocabulary


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment_sequences(X: np.ndarray, y: np.ndarray,
                      augment_factor: int = 5,
                      intensity: float = 0.7) -> tuple:
    """
    Augment training data using the full augmentation suite.

    Each original sample gets `augment_factor` augmented copies, applying
    a random combination of: noise, scale jitter, 2D rotation, time warp,
    landmark dropout, and hand swap.

    Args:
        X:               Original sequences (N, seq_len, features).
        y:               Labels (N,).
        augment_factor:  Augmented copies per original sample.
        intensity:       Augmentation strength (0.0–2.0).

    Returns:
        Tuple of (augmented_X, augmented_y).
    """
    X_aug_list = [X]
    y_aug_list = [y]

    print(f"  Generating {augment_factor}x augmented copies (intensity={intensity:.1f})...")
    for pass_num in range(augment_factor):
        batch = np.array(
            [augment_sequence(seq, intensity=intensity) for seq in X],
            dtype=np.float32,
        )
        X_aug_list.append(batch)
        y_aug_list.append(y)
        if (pass_num + 1) % 5 == 0:
            print(f"    Pass {pass_num + 1}/{augment_factor} done")

    return np.concatenate(X_aug_list), np.concatenate(y_aug_list)


# ─────────────────────────────────────────────────────────────────────────────
# K-Fold cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def run_kfold(X: np.ndarray, y: np.ndarray, label_names: list, args):
    """Run K-Fold cross-validation and print per-fold metrics."""
    kfold = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=42)
    fold_accuracies = []
    all_true = []
    all_pred = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y), start=1):
        print(f"\n[CV] Fold {fold_idx}/{args.kfold}")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        if not args.no_augment:
            X_train, y_train = augment_sequences(
                X_train, y_train, augment_factor=5, intensity=1.0
            )

        seq_length = X_train.shape[1]
        num_features = X_train.shape[2]
        num_classes = len(label_names)

        model = build_model(num_features, num_classes,
                            seq_length=seq_length, model_type=args.model_type)

        cw = class_weight.compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )
        cw_dict = {i: float(w) for i, w in enumerate(cw)}

        y_train_cat = to_categorical(y_train, num_classes=num_classes)
        y_val_cat = to_categorical(y_val, num_classes=num_classes)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=20, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            ),
        ]

        model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            class_weight=cw_dict,
            verbose=1,
        )

        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        acc = accuracy_score(y_val, y_pred)
        fold_accuracies.append(acc)
        all_true.extend(y_val.tolist())
        all_pred.extend(y_pred.tolist())

        print(f"[CV] Fold {fold_idx} accuracy: {acc:.4f}")
        print(classification_report(y_val, y_pred, target_names=label_names, digits=3))

    print("\n[CV] Cross-validation summary:")
    print(f"  Mean accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"  Std accuracy:  {np.std(fold_accuracies):.4f}")
    print("\n[CV] Confusion matrix (all folds):")
    print(confusion_matrix(all_true, all_pred))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ISL Gesture Recognition Model")

    # Architecture
    parser.add_argument("--model-type", type=str, default=config.MODEL_TYPE,
                        choices=["lstm", "tcn", "mlp"],
                        help=f"Model architecture (default: {config.MODEL_TYPE})")

    # Data
    parser.add_argument("--data-dir", type=str, default=config.DATA_DIR,
                        help="Path to raw .npy samples directory")
    parser.add_argument("--reload", action="store_true",
                        help="Force reload from raw data (ignore cached .npz)")
    parser.add_argument("--all-classes", action="store_true",
                        help="Train on ALL available classes (default: 19 target words)")

    # Balancing
    parser.add_argument("--no-balance", action="store_true",
                        help="Skip dataset balancing step")
    parser.add_argument("--balance-target", type=int, default=config.BALANCE_TARGET,
                        help=f"Min samples per class after balancing (default: {config.BALANCE_TARGET})")

    # Augmentation
    parser.add_argument("--no-augment", action="store_true",
                        help="Skip data augmentation")
    parser.add_argument("--augment-factor", type=int, default=5,
                        help="Augmented copies per original sample (default: 5)")

    # Features
    parser.add_argument("--no-velocity", action="store_true",
                        help="Disable velocity feature concatenation")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)

    # Cross-validation
    parser.add_argument("--kfold", type=int, default=0,
                        help="Run K-Fold cross-validation (e.g. 5) before final training")

    # INCLUDE video processing (legacy flags kept for compatibility)
    parser.add_argument("--process-include", action="store_true",
                        help="Process INCLUDE videos before training")
    parser.add_argument("--include-dir", type=str, default=config.INCLUDE_DIR)
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--max-videos", type=int, default=None)

    args = parser.parse_args()

    print("=" * 60)
    print("  ISL Gesture Recognition — Model Training")
    print(f"  Architecture : {args.model_type.upper()}")
    print(f"  Velocity     : {'ON' if not args.no_velocity else 'OFF'}")
    print(f"  Balance      : {'OFF' if args.no_balance else f'target {args.balance_target}'}")
    print(f"  Augment      : {'OFF' if args.no_augment else f'{args.augment_factor}x'}")
    print("=" * 60)

    # ── Optionally process INCLUDE videos ───────────────────────────────────
    if args.process_include:
        from process_videos import scan_include_dataset, process_dataset
        if not os.path.isdir(args.include_dir):
            print(f"\n[ERROR] INCLUDE videos directory not found: {args.include_dir}")
            return
        print(f"\n[STEP 0] Processing INCLUDE videos from {args.include_dir}...")
        word_videos = scan_include_dataset(args.include_dir)
        process_dataset(word_videos, args.data_dir,
                        max_words=args.max_words, max_videos_per_word=args.max_videos)

    # ── Step 1: Balance dataset ──────────────────────────────────────────────
    if not args.no_balance:
        print(f"\n[STEP 1] Balancing dataset (target: {args.balance_target} samples/class)...")
        from src.balance_dataset import balance_dataset
        balance_dataset(data_dir=args.data_dir, target=args.balance_target)
    else:
        print("\n[STEP 1] Skipping dataset balancing (--no-balance)")

    # ── Step 2: Load dataset ─────────────────────────────────────────────────
    X, y, label_names = None, None, None
    dataset_npz_path = os.path.join(args.data_dir, "dataset.npz")

    if not args.reload and os.path.exists(dataset_npz_path):
        print(f"\n[STEP 2] Loading cached dataset from {dataset_npz_path}...")
        X, y, label_names = load_dataset_npz(dataset_npz_path)

    if X is None:
        print(f"\n[STEP 2] Loading data from raw samples in {args.data_dir}...")
        X, y, label_names = load_dataset(args.data_dir)

        if X is None:
            print("\n[ERROR] No training data found!")
            print("  Record data:   python main.py --mode collect --word HELLO")
            print("  Or process INCLUDE videos: python train.py --process-include")
            return

        # We always reload after balancing since new files were written
        save_dataset_npz(X, y, label_names, dataset_npz_path)
    elif not args.no_balance:
        # Re-load raw to pick up newly augmented samples
        print("  Re-loading after balancing...")
        X, y, label_names = load_dataset(args.data_dir)
        save_dataset_npz(X, y, label_names, dataset_npz_path)

    # Normalize labels (merge numbered dirs like 48._HELLO → HELLO)
    X, y, label_names = normalize_labels(X, y, label_names)

    if not args.all_classes:
        X, y, label_names = filter_labels(X, y, label_names, config.MAIN_WORDS)

    num_classes = len(label_names)
    if num_classes < 2:
        print("\n[ERROR] Need at least 2 different words to train!")
        return

    # Print per-class distribution
    print(f"\n[INFO] Dataset: {X.shape[0]} samples, {num_classes} classes")
    unique, counts = np.unique(y, return_counts=True)
    print("  Per-class samples:")
    for idx, cnt in zip(unique, counts):
        bar = "#" * (cnt // 10)
        print(f"    {label_names[idx]:20s}: {cnt:4d}  {bar}")
    print(f"  Min: {counts.min()}  Max: {counts.max()}  Ratio: {counts.max()/max(counts.min(),1):.2f}x")

    # ── Step 3: Normalize landmarks ──────────────────────────────────────────
    if config.NORMALIZE_LANDMARKS:
        print("\n[STEP 3] Normalizing hand landmarks...")
        X = np.array([normalize_hands_sequence(seq) for seq in X], dtype=np.float32)

    # ── Step 4: Split FIRST, augment train-only ──────────────────────────────
    # CRITICAL: split must happen on original data before augmentation.
    # Otherwise augmented copies of training samples leak into the val set,
    # causing artificially inflated val accuracy.
    from sklearn.model_selection import train_test_split as tts
    X_train_orig, X_val, y_train_orig, y_val = tts(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[INFO] Original split: {len(X_train_orig)} train | {len(X_val)} val (original only)")

    if not args.no_augment:
        print(f"\n[STEP 4] Augmenting TRAINING split ({args.augment_factor}x full augmentation)...")
        X_train, y_train = augment_sequences(
            X_train_orig, y_train_orig, augment_factor=args.augment_factor,
            intensity=0.7
        )
        print(f"  Augmented training: {len(X_train_orig)} → {len(X_train)} samples")
        print(f"  Validation stays at {len(X_val)} REAL samples")
        # Shuffle the augmented training set
        perm = np.random.permutation(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]
    else:
        X_train, y_train = X_train_orig, y_train_orig
        print("\n[STEP 4] Skipping augmentation (--no-augment)")

    # ── Step 5: Velocity features ────────────────────────────────────────────
    seq_length = X.shape[1]
    num_features = X.shape[2]

    if not args.no_velocity:
        print("\n[STEP 5] Adding velocity features (position + motion)...")
        X_train = np.array([build_feature_vector(seq) for seq in X_train], dtype=np.float32)
        X_val   = np.array([build_feature_vector(seq) for seq in X_val],   dtype=np.float32)
        num_features = X_train.shape[2]
        print(f"  Feature dims: {config.NUM_FEATURES} → {num_features} (position + velocity)")
    else:
        print("\n[STEP 5] Skipping velocity features (--no-velocity)")

    # ── Step 6: K-Fold cross-validation (optional) ───────────────────────────
    if args.kfold and args.kfold > 1:
        print(f"\n[STEP 6] Running {args.kfold}-fold cross-validation...")
        config.EPOCHS = args.epochs
        config.BATCH_SIZE = args.batch_size
        run_kfold(X, y, label_names, args)

    # ── Step 7: Build and train final model ──────────────────────────────────
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size

    print(f"\n[STEP 7] Building {args.model_type.upper()} model...")
    model = build_model(num_features, num_classes,
                        seq_length=seq_length, model_type=args.model_type)

    print(f"\n[STEP 8] Training for up to {args.epochs} epochs...")
    # Pass pre-split data: model.py will NOT re-split (it receives pre-split arrays)
    model, history = train_model(X_train, y_train, num_classes, model,
                                 X_val_override=X_val, y_val_override=y_val)

    # ── Step 9: Save model + vocabulary ─────────────────────────────────────
    print("\n[STEP 9] Saving model and vocabulary...")
    save_model(model)

    vocab = {
        "words": label_names,
        "word_to_index": {w: i for i, w in enumerate(label_names)},
        "index_to_word": {str(i): w for i, w in enumerate(label_names)},
        "use_velocity": not args.no_velocity,
        "sequence_length": seq_length,
        "num_features": num_features,
        "model_type": args.model_type,
    }
    save_vocabulary(vocab)
    print(f"  Vocabulary: {label_names}")

    # Per-class classification report on the val-only split
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    print("\n[RESULT] Final per-class report (on original, unaugmented validation set):")
    print(classification_report(y_val, y_pred, target_names=label_names, digits=3))

    # ── Step 10: Plot ────────────────────────────────────────────────────────
    print("\n[STEP 10] Generating training plots...")
    plot_training_history(history)

    best_val_acc = max(history.history["val_accuracy"])
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  Best val accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
    print(f"  Model:   {config.MODEL_PATH}")
    print(f"  Classes: {num_classes} words")
    print("  Run recognition: python main.py --mode recognize")
    print("=" * 60)


if __name__ == "__main__":
    main()
