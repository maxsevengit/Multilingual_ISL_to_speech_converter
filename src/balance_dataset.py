"""
Dataset Balancing Module.

Scans the raw data directory, identifies underrepresented classes,
and augments them up to a target minimum sample count using the
full augmentation suite (noise, scale, rotate, time-warp, dropout, hand-swap).

Usage (standalone):
    python -m src.balance_dataset
    python -m src.balance_dataset --target 200 --data-dir data/raw
"""

import os
import argparse
import numpy as np
import config
from src.augment_landmarks import augment_sequence


# ─── Target Vocabulary (19 words, no IDLE) ────────────────────────────────────
MAIN_WORDS = set(config.MAIN_WORDS)


def _list_samples(word_dir: str) -> list:
    """Return sorted list of .npy files in a word directory."""
    return sorted([f for f in os.listdir(word_dir) if f.endswith(".npy")])


def _next_sample_index(word_dir: str) -> int:
    """Return the next unused sample index for a word directory."""
    files = _list_samples(word_dir)
    if not files:
        return 0
    indices = []
    for f in files:
        try:
            indices.append(int(f.replace("sample_", "").replace(".npy", "")))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0


def balance_dataset(data_dir: str = None, target: int = 200,
                    dry_run: bool = False) -> dict:
    """
    Augment underrepresented classes up to `target` samples.

    Only operates on classes that are in config.MAIN_WORDS.
    Classes with >= target samples are left untouched.

    Args:
        data_dir:  Path to data/raw directory.
        target:    Minimum sample count per class after balancing.
        dry_run:   If True, print what would be done but don't save.

    Returns:
        Dict mapping word → (before, after) sample counts.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    print("=" * 60)
    print("  Dataset Balancer")
    print(f"  Target minimum: {target} samples per class")
    print(f"  Data directory: {data_dir}")
    if dry_run:
        print("  DRY RUN — no files will be written")
    print("=" * 60)

    # ── Scan existing data ─────────────────────────────────────────────────────
    word_dirs = [
        d for d in sorted(os.listdir(data_dir))
        if os.path.isdir(os.path.join(data_dir, d))
    ]

    # Normalize names so numbered folders map to canonical word names
    import re

    def normalize(name: str) -> str:
        label = str(name).strip().upper().replace(" ", "_")
        label = re.sub(r"^\d+\._", "", label)
        label = re.sub(r"^EX\._", "", label)
        label = re.sub(r"^\d+[_\.]+", "", label)
        return label

    # Group all directories by canonical name
    canonical_to_dirs: dict[str, list[str]] = {}
    for d in word_dirs:
        norm = normalize(d)
        if norm not in MAIN_WORDS:
            continue
        canonical_to_dirs.setdefault(norm, []).append(d)

    results = {}

    for word, dirs in sorted(canonical_to_dirs.items()):
        # Collect all existing samples across all variant folders for this word
        all_samples = []
        primary_dir = sorted(dirs)[0]  # Use the first (canonical) dir for saving
        primary_path = os.path.join(data_dir, primary_dir)

        for d in dirs:
            dpath = os.path.join(data_dir, d)
            for fname in _list_samples(dpath):
                fpath = os.path.join(dpath, fname)
                try:
                    seq = np.load(fpath)
                    if seq.shape != (config.SEQUENCE_LENGTH, config.NUM_FEATURES):
                        continue
                    # Skip all-zeros and mostly-zeros samples
                    if np.all(seq == 0):
                        continue
                    nonzero_frac = np.count_nonzero(seq) / seq.size
                    if nonzero_frac < 0.05:
                        continue
                    all_samples.append(seq)
                except Exception:
                    pass

        before = len(all_samples)
        results[word] = [before, before]

        if before >= target:
            print(f"  {word:20s}: {before:4d} samples — OK (>= {target})")
            continue

        needed = target - before
        print(f"  {word:20s}: {before:4d} samples — augmenting +{needed} → {target}")

        if dry_run:
            results[word][1] = target
            continue

        if not all_samples:
            print(f"    [WARN] No valid samples found for '{word}', skipping.")
            continue

        # Determine augmentation intensity: lower count = stronger augmentation
        ratio = target / max(before, 1)
        intensity = min(ratio * 0.5, 2.0)  # cap at 2.0

        next_idx = _next_sample_index(primary_path)
        generated = 0

        while generated < needed:
            # Pick a random base sample
            base = all_samples[np.random.randint(len(all_samples))]
            aug = augment_sequence(base, intensity=intensity)

            save_path = os.path.join(primary_path, f"sample_{next_idx:04d}.npy")
            np.save(save_path, aug.astype(np.float32))
            next_idx += 1
            generated += 1

        results[word][1] = before + generated
        print(f"    → Saved {generated} augmented samples to '{primary_dir}'")

    print("\n  Summary:")
    total_before = sum(v[0] for v in results.values())
    total_after = sum(v[1] for v in results.values())
    print(f"  Total samples: {total_before} → {total_after}")
    print(f"  Classes processed: {len(results)}")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Balance ISL dataset by augmenting underrepresented classes")
    parser.add_argument("--target", type=int, default=200,
                        help="Minimum samples per class (default: 200)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to data/raw directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without saving files")
    args = parser.parse_args()

    balance_dataset(
        data_dir=args.data_dir,
        target=args.target,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
