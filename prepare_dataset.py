"""
Prepare INCLUDE videos for training.

Does not download the dataset. Run download_dataset.py first.

Usage:
    python prepare_dataset.py --dataset include
    python prepare_dataset.py --dataset include --max-videos 2
"""

import argparse
import os
import sys

import config
from process_videos import process_dataset, scan_include_dataset
from src.logutil import get_logger

log = get_logger("prepare")


def main():
    parser = argparse.ArgumentParser(description="Prepare the INCLUDE dataset")
    parser.add_argument("--dataset", type=str, default="include")
    parser.add_argument("--input", type=str, default=config.INCLUDE_DIR)
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()

    if args.dataset != "include":
        log.info("[ERROR] Only --dataset include is supported.")
        log.info("  iSign and ISLTranslate are continuous translation resources.")
        sys.exit(2)

    if not os.path.isdir(args.input):
        log.info(f"[ERROR] Video directory not found: {args.input}")
        log.info("  Download it first. This script does not download data:")
        log.info("    python download_dataset.py --dataset include")
        sys.exit(2)

    word_videos = scan_include_dataset(args.input)
    if not word_videos:
        log.info(f"[ERROR] No INCLUDE videos found under {args.input}")
        log.info("  Expected: data/include_videos/<Category>/<Word>/*.mp4")
        sys.exit(2)

    total, words = process_dataset(
        word_videos,
        max_words=args.max_words,
        max_videos_per_word=args.max_videos,
        base_dir=config.BASE_DIR,
    )
    log.info(f"[INFO] Prepared {total} windows across {len(words)} words.")
    log.info(f"[INFO] Manifest: {config.MANIFEST_PATH}")


if __name__ == "__main__":
    main()
