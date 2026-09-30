"""
Dataset balancing entry point.

Writing augmented copies into data/raw before the split leaks near-duplicates
into validation. Augmentation now happens in memory inside train.py, and only
on the training split.
"""

import argparse


def balance_dataset(data_dir: str = None, target: int = 200,
                    dry_run: bool = False) -> dict:
    """
    Refuse to materialize augmented samples on disk.

    Args:
        data_dir: Unused. Kept so older command lines still reach this error.
        target: Unused.
        dry_run: Unused.

    Raises:
        RuntimeError: Always. On-disk balancing is no longer part of training.
    """
    del data_dir, target, dry_run
    raise RuntimeError(
        "On-disk class balancing is disabled. Augmented samples must not be "
        "written into data/raw or data/processed. "
        "train.py augments only the training split in memory after the "
        "video-level split. Validation and test windows stay unchanged."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Disabled: augmentation is applied in memory by train.py"
    )
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    balance_dataset(data_dir=args.data_dir, target=args.target, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
