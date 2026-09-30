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
import json
import os
import time
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import config
from src.augment_landmarks import augment_sequence
from src.dataset import normalize_label_name
from src.feature_engineer import build_feature_vector, normalize_hands_sequence, training_features
from src.manifest import load_split_arrays, read_manifest, write_manifest, write_split
from src.model_bundle import pipeline_spec, save_model_bundle
from src.splits import assert_no_leakage, assign_splits
from src.utils import save_vocabulary


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
    """Window-level k-fold is disabled because overlapping windows leak."""
    raise RuntimeError(
        "K-fold over windows is disabled. Overlapping windows from the same "
        "video would land in different folds. The evaluation protocol is one "
        "held-out test split grouped by video id. Run python train.py without --kfold."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ISL Gesture Recognition Model")

    # Architecture
    parser.add_argument("--model-type", type=str, default=config.MODEL_TYPE,
                        choices=["lstm", "gru", "tcn", "mlp", "transformer"],
                        help=f"Model architecture (default: {config.MODEL_TYPE})")
    parser.add_argument("--bundle-dir", type=str, default=config.BUNDLE_DIR,
                        help="Directory for model.keras and the JSON sidecar files")

    # Data
    parser.add_argument("--dataset", type=str, default="include",
                        help="Dataset name. Only 'include' is the supported isolated-sign set.")
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
    config.BUNDLE_DIR = args.bundle_dir

    import random
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(config.SEED)

    print("=" * 60)
    print("  ISL Gesture Recognition - Model Training")
    print(f"  Dataset      : {args.dataset}")
    print(f"  Architecture : {args.model_type.upper()}")
    print(f"  Bundle       : {config.BUNDLE_DIR}")
    print(f"  Seed         : {config.SEED}")
    print(f"  Velocity     : {'ON' if not args.no_velocity else 'OFF'}")
    print(f"  Augment      : {'OFF' if args.no_augment else f'{args.augment_factor}x on train only'}")
    print("=" * 60)

    if args.dataset != "include":
        print("\n[ERROR] Only the INCLUDE dataset is wired to this isolated-sign pipeline.")
        print("  iSign and ISLTranslate are continuous translation sets and are not used here.")
        return

    if args.kfold and args.kfold > 1:
        print("\n[ERROR] --kfold is disabled.")
        print("  Overlapping windows cannot be folded independently of their source video.")
        print("  Evaluation is a single held-out test split grouped by video id.")
        return

    if args.process_include:
        from process_videos import scan_include_dataset, process_dataset
        if not os.path.isdir(args.include_dir):
            print(f"\n[ERROR] INCLUDE videos directory not found: {args.include_dir}")
            print("  Download first: python download_dataset.py --dataset include")
            return
        print(f"\n[STEP 0] Processing INCLUDE videos from {args.include_dir}...")
        word_videos = scan_include_dataset(args.include_dir)
        process_dataset(
            word_videos,
            max_words=args.max_words,
            max_videos_per_word=args.max_videos,
        )

    records = read_manifest()
    if not records:
        print("\n[ERROR] No window manifest found.")
        print("  1. python download_dataset.py --dataset include")
        print("  2. python prepare_dataset.py --dataset include")
        print("  3. python train.py --dataset include")
        return

    if any(record.get("split") not in ("train", "val", "test") for record in records):
        records, summary = assign_splits(records, seed=config.SEED)
        write_manifest(records)
        write_split(summary)
    else:
        summary = {
            "strategy": "video",
            "seed": config.SEED,
            "videos": {
                name: sorted({r["video_id"] for r in records if r["split"] == name})
                for name in ("train", "val", "test")
            },
        }

    if not args.all_classes:
        allowed = {normalize_label_name(word) for word in config.MAIN_WORDS}
        records = [
            record for record in records
            if normalize_label_name(record["label"]) in allowed
        ]
        for record in records:
            record["label"] = normalize_label_name(record["label"])

    summary["videos"] = {
        name: sorted({record["video_id"] for record in records if record["split"] == name})
        for name in ("train", "val", "test")
    }
    assert_no_leakage(records)

    use_velocity = not args.no_velocity
    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        label_names,
        skipped,
    ) = load_split_arrays(records)

    print("\n[INFO] Video-level split")
    print(f"  Train videos:      {len(summary['videos']['train'])}")
    print(f"  Validation videos: {len(summary['videos']['val'])}")
    print(f"  Test videos:       {len(summary['videos']['test'])}")
    print(f"  Train windows:     {len(X_train)}")
    print(f"  Val windows:       {len(X_val)}")
    print(f"  Test windows:      {len(X_test)}")
    if skipped:
        print(f"  Skipped windows:   {skipped}")

    if len(label_names) < 2 or len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print("\n[ERROR] Need train, validation, and test windows from at least two classes.")
        print("  A class with only one video cannot fill every split.")
        print("  No test metric was written.")
        return

    print("\n[STEP 1] Applying the shared feature transform to validation and test.")
    X_val = np.stack([
        training_features(seq, use_velocity=use_velocity) for seq in X_val
    ])
    X_test = np.stack([
        training_features(seq, use_velocity=use_velocity) for seq in X_test
    ])

    print("\n[STEP 1b] Normalizing the training split with normalize_hands_sequence.")
    X_train = np.stack([normalize_hands_sequence(seq) for seq in X_train])

    if not args.no_augment:
        print(f"\n[STEP 2] Augmenting the training split only ({args.augment_factor}x).")
        X_train, y_train = augment_sequences(
            X_train, y_train, augment_factor=args.augment_factor, intensity=0.7
        )
        permutation = np.random.default_rng(config.SEED).permutation(len(X_train))
        X_train, y_train = X_train[permutation], y_train[permutation]
    else:
        print("\n[STEP 2] Training split left unaugmented.")

    if use_velocity:
        print("\n[STEP 3] Adding velocity to the training split.")
        X_train = np.stack([build_feature_vector(seq) for seq in X_train])
    else:
        print("\n[STEP 3] Velocity features off.")

    num_classes = len(label_names)
    num_features = X_train.shape[2]
    expected = config.NUM_FEATURES * (2 if use_velocity else 1)
    if num_features != expected:
        print(f"\n[ERROR] Feature width {num_features} does not match pipeline width {expected}.")
        return

    os.makedirs(config.BUNDLE_DIR, exist_ok=True)
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.MODEL_PATH = os.path.join(config.BUNDLE_DIR, "model.keras")

    from src.model import build_model, plot_training_history, train_model

    print(f"\n[STEP 4] Building {args.model_type.upper()} model.")
    model = build_model(
        num_features, num_classes,
        seq_length=config.SEQUENCE_LENGTH,
        model_type=args.model_type,
    )
    model, history = train_model(
        X_train, y_train, num_classes, model,
        X_val_override=X_val, y_val_override=y_val,
    )

    print("\n[STEP 5] Evaluating the held-out test split once.")
    probabilities = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    labels_present = list(range(num_classes))
    report = classification_report(
        y_test, y_pred, labels=labels_present, target_names=label_names,
        digits=3, zero_division=0, output_dict=True,
    )
    print(classification_report(
        y_test, y_pred, labels=labels_present, target_names=label_names,
        digits=3, zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=labels_present))

    latency = _latency_ms(model, X_test)
    per_class_precision = {
        name: float(report[name]["precision"]) for name in label_names if name in report
    }
    per_class_recall = {
        name: float(report[name]["recall"]) for name in label_names if name in report
    }
    per_class_f1 = {
        name: float(report[name]["f1-score"]) for name in label_names if name in report
    }
    matrix = confusion_matrix(y_test, y_pred, labels=labels_present)
    categories = _categories_from_video_ids(record["video_id"] for record in records)
    experiment = {
        "dataset": "include",
        "category": categories,
        "num_classes": num_classes,
        "train_videos": len(summary["videos"]["train"]),
        "validation_videos": len(summary["videos"]["val"]),
        "test_videos": len(summary["videos"]["test"]),
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "feature_width": num_features,
        "sequence_length": config.SEQUENCE_LENGTH,
        "model": args.model_type,
        "random_seed": config.SEED,
        "augment_factor": 0 if args.no_augment else args.augment_factor,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "split": "test",
    }
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": label_names,
        "num_classes": num_classes,
        "n_classes": num_classes,
        "test_samples": int(len(y_test)),
        "n_train_windows": int(len(y_train)),
        "n_val_windows": int(len(y_val)),
        "n_test_windows": int(len(y_test)),
        "n_train_videos": len(summary["videos"]["train"]),
        "n_val_videos": len(summary["videos"]["val"]),
        "n_test_videos": len(summary["videos"]["test"]),
        "parameters": int(model.count_params()),
        "latency_ms": latency,
        "p50_cpu_latency_ms": latency["p50"],
        "p95_cpu_latency_ms": latency["p95"],
        "split": "test",
        "dataset": "include",
        "seed": config.SEED,
        "sequence_length": config.SEQUENCE_LENGTH,
        "feature_width": num_features,
        "model_type": args.model_type,
        "category": categories,
        "experiment": experiment,
    }

    vocabulary = {
        "words": label_names,
        "word_to_index": {word: i for i, word in enumerate(label_names)},
        "index_to_word": {str(i): word for i, word in enumerate(label_names)},
        "use_velocity": use_velocity,
        "sequence_length": config.SEQUENCE_LENGTH,
        "num_features": num_features,
        "model_type": args.model_type,
    }
    bundle_config = pipeline_spec(model_type=args.model_type, use_velocity=use_velocity)
    save_model_bundle(
        config.BUNDLE_DIR, model, bundle_config, vocabulary, metrics, summary
    )
    model_path = os.path.join(config.BUNDLE_DIR, "model.keras")
    metrics["model_size_bytes"] = int(os.path.getsize(model_path))
    metrics["experiment"]["model_size_bytes"] = metrics["model_size_bytes"]
    metrics_path = os.path.join(config.BUNDLE_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with open(os.path.join(config.BUNDLE_DIR, "experiment.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics["experiment"], handle, indent=2)
    save_vocabulary(vocabulary)

    print("\n[STEP 6] Generating training plots...")
    plot_training_history(history)
    print("\n" + "=" * 60)
    print("  Training complete. Test metrics are in the model bundle.")
    print(f"  Test accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1:      {metrics['macro_f1']:.4f}")
    print(f"  Bundle:        {config.BUNDLE_DIR}")
    print("=" * 60)


def _categories_from_video_ids(video_ids) -> list:
    """Category folder sitting under include_videos in each video id."""
    categories = set()
    for video_id in video_ids:
        parts = str(video_id).replace("\\", "/").split("/")
        if "include_videos" in parts:
            index = parts.index("include_videos")
            if index + 1 < len(parts):
                categories.add(parts[index + 1])
    return sorted(categories)


def _latency_ms(model, X_test: np.ndarray, repeats: int = 30) -> dict:
    """Single-example CPU latency after a short warmup."""
    sample = X_test[:1]
    for _ in range(5):
        model.predict(sample, verbose=0)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(sample, verbose=0)
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    p50_index = len(samples) // 2
    p95_index = min(len(samples) - 1, int(round(0.95 * (len(samples) - 1))))
    return {
        "p50": float(samples[p50_index]),
        "p95": float(samples[p95_index]),
        "mean": float(np.mean(samples)),
        "repeats": repeats,
        "batch_size": 1,
    }


if __name__ == "__main__":
    main()
