"""
Video-to-Landmark Processing Script for ISL Gesture Recognition.

Processes a directory of ISL word-level videos, extracting MediaPipe
landmarks and saving them as .npy sequences ready for model training.

Supports:
  - INCLUDE dataset format (videos organized in category/word/video.mp4)
  - Generic format (videos organized in word/video.mp4)
  - Single video files

Usage:
    python process_videos.py --input data/include_videos
    python process_videos.py --input data/include_videos --max-words 30
    python process_videos.py --input my_videos/ --format generic
"""

import argparse
import os
import sys
import glob
import cv2
import numpy as np
from collections import defaultdict

import config
from src.dataset import normalize_label_name
from src.preprocessing import normalize_frame, convert_color
from src.landmark_extractor import LandmarkExtractor
from src.manifest import save_window, window_records_from_frames, write_manifest
from src.splits import assert_no_leakage, assign_splits
from src.manifest import write_split


def scan_include_dataset(input_dir: str) -> dict:
    """
    Scan the INCLUDE dataset directory structure.
    
    INCLUDE format:
        input_dir/
        ├── Category_Name/
        │   ├── Word_Name/
        │   │   ├── video1.mp4
        │   │   ├── video2.mp4
        │   │   └── ...
    
    Args:
        input_dir: Path to the INCLUDE video directory.
    
    Returns:
        Dict mapping word names to lists of video file paths.
    """
    word_videos = defaultdict(list)
    
    for category_dir in sorted(os.listdir(input_dir)):
        category_path = os.path.join(input_dir, category_dir)
        if not os.path.isdir(category_path):
            continue
        
        for word_dir in sorted(os.listdir(category_path)):
            word_path = os.path.join(category_path, word_dir)
            if not os.path.isdir(word_path):
                continue
            
            # INCLUDE folders look like "61. Summer" or "Ex. Monsoon".
            word_name = normalize_label_name(word_dir)
            if not word_name:
                continue
            
            seen = set()
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4', '*.AVI', '*.MOV', '*.MKV']:
                for video in glob.glob(os.path.join(word_path, ext)):
                    key = os.path.normcase(os.path.abspath(video))
                    if key in seen:
                        continue
                    seen.add(key)
                    word_videos[word_name].append(video)
    
    return dict(word_videos)


def scan_generic_dataset(input_dir: str) -> dict:
    """
    Scan a generic dataset directory structure.
    
    Generic format:
        input_dir/
        ├── Word_Name/
        │   ├── video1.mp4
        │   └── ...
    
    Args:
        input_dir: Path to video directory.
    
    Returns:
        Dict mapping word names to lists of video file paths.
    """
    word_videos = defaultdict(list)
    
    for word_dir in sorted(os.listdir(input_dir)):
        word_path = os.path.join(input_dir, word_dir)
        if not os.path.isdir(word_path):
            continue
        
        word_name = word_dir.strip().upper().replace(" ", "_")
        
        seen = set()
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4', '*.AVI', '*.MOV', '*.MKV']:
            for video in glob.glob(os.path.join(word_path, ext)):
                key = os.path.normcase(os.path.abspath(video))
                if key in seen:
                    continue
                seen.add(key)
                word_videos[word_name].append(video)
    
    return dict(word_videos)


def process_video(video_path: str, extractor: LandmarkExtractor) -> list:
    """
    Process a single video file and extract landmark sequences.
    
    Args:
        video_path: Path to the video file.
        extractor: LandmarkExtractor instance.
    
    Returns:
        List of landmark arrays (one per frame).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"      [WARN] Cannot open: {os.path.basename(video_path)}")
        return []
    
    landmarks_list = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame
        frame = normalize_frame(frame)
        frame_rgb = convert_color(frame, 'RGB')
        
        # Extract landmarks
        features = extractor.extract_landmarks(frame_rgb)
        landmarks_list.append(features)
        frame_count += 1
    
    cap.release()
    return landmarks_list


def create_samples_from_video(landmarks_list: list, 
                               min_frames: int = 10,
                               video_id: str = "video",
                               label: str = "UNKNOWN") -> list:
    """
    Convert one video into raw windows that keep the source video id.

    Returns a list of (metadata, array) pairs.
    """
    return window_records_from_frames(
        landmarks_list,
        video_id=video_id,
        label=label,
        min_frames=min_frames,
    )


def process_dataset(word_videos: dict, output_dir: str = None, 
                    max_words: int = None, max_videos_per_word: int = None,
                    base_dir: str = None):
    """
    Process videos into raw landmark windows and a manifest.

    Augmented samples are not written. Split labels are assigned by video id
    after every window has been saved.
    """
    if base_dir is None:
        base_dir = config.BASE_DIR
    extractor = LandmarkExtractor()
    
    words = sorted(word_videos.keys())
    if max_words:
        words = words[:max_words]
    
    records = []
    processed_words = []
    
    print(f"\n  Processing {len(words)} words into raw landmark windows...")
    print(f"  Manifest: {config.MANIFEST_PATH}\n")
    
    for word_idx, word in enumerate(words):
        videos = word_videos[word]
        if max_videos_per_word:
            videos = videos[:max_videos_per_word]
        
        print(f"  [{word_idx+1}/{len(words)}] {word}: processing {len(videos)} videos...")
        word_windows = 0
        
        for vid_idx, video_path in enumerate(videos):
            video_id = os.path.relpath(video_path, base_dir).replace("\\", "/")
            landmarks_list = process_video(video_path, extractor)
            if not landmarks_list:
                continue
            
            samples = create_samples_from_video(
                landmarks_list, video_id=video_id, label=word
            )
            for record, array in samples:
                save_window(base_dir, record, array)
                records.append(record)
                word_windows += 1
            
            if (vid_idx + 1) % 10 == 0:
                print(f"    Processed {vid_idx+1}/{len(videos)} videos ({word_windows} windows)")
        
        print(f"    -> {word_windows} windows saved for '{word}'")
        if word_windows:
            processed_words.append(word)
    
    extractor.release()

    records, summary = assign_splits(records, seed=config.SEED)
    assert_no_leakage(records)
    write_manifest(records, config.MANIFEST_PATH)
    write_split(summary, config.SPLIT_PATH)
    
    print(f"\n  Split strategy: {summary['strategy']}")
    print(f"  Videos: {summary['counts']['videos']}")
    print(f"  Windows: {summary['counts']['windows']}")
    if summary.get("limitation"):
        print(f"  Note: {summary['limitation']}")
    
    return len(records), processed_words


def main():
    parser = argparse.ArgumentParser(
        description="Process ISL videos into landmark training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process INCLUDE dataset (downloaded via download_dataset.py)
  python process_videos.py --input data/include_videos

  # Process with limit on number of words  
  python process_videos.py --input data/include_videos --max-words 20

  # Process a generic video dataset (videos/WORD_NAME/video.mp4)
  python process_videos.py --input my_videos/ --format generic
  
  # Process with custom output directory
  python process_videos.py --input data/include_videos --output data/raw
        """
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing video files')
    parser.add_argument('--output', type=str, default=None,
                        help='Unused. Windows are written under data/processed with a manifest.')
    parser.add_argument('--format', type=str, choices=['include', 'generic'],
                        default='include',
                        help='Dataset directory format (default: include)')
    parser.add_argument('--max-words', type=int, default=None,
                        help='Maximum number of words to process')
    parser.add_argument('--max-videos', type=int, default=None,
                        help='Maximum videos per word')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.isdir(args.input):
        print(f"[ERROR] Input directory not found: {args.input}")
        sys.exit(1)
    
    # Set output directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output or os.path.join(base_dir, "data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("  ISL Video to Landmark Processor")
    print("=" * 60)
    
    # ── Scan dataset ─────────────────────────────────────────────────────────
    print(f"\n  Scanning {args.input}...")
    
    if args.format == 'include':
        word_videos = scan_include_dataset(args.input)
    else:
        word_videos = scan_generic_dataset(args.input)
    
    if not word_videos:
        print("[ERROR] No videos found! Check your directory structure.")
        print("  Expected format:")
        if args.format == 'include':
            print("    input_dir/Category/WordName/video.mp4")
        else:
            print("    input_dir/WordName/video.mp4")
        sys.exit(1)
    
    # ── Print summary ────────────────────────────────────────────────────────
    total_videos = sum(len(v) for v in word_videos.values())
    print(f"  Found {len(word_videos)} words, {total_videos} total videos")
    print(f"\n  Words: {', '.join(sorted(word_videos.keys())[:15])}")
    if len(word_videos) > 15:
        print(f"    ... and {len(word_videos) - 15} more")
    
    # ── Process ──────────────────────────────────────────────────────────────
    total_samples, words = process_dataset(
        word_videos, output_dir,
        max_words=args.max_words,
        max_videos_per_word=args.max_videos
    )
    
    print(f"\n{'='*60}")
    print(f"  Processing Complete!")
    print(f"  Total: {total_samples} samples across {len(words)} words")
    print(f"  Output: data/processed and data/manifests/samples.jsonl")
    print(f"\n  Next step: Train the model:")
    print(f"  python train.py --dataset include")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
