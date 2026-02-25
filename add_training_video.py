"""
YouTube Video Training Integration Script.

Process a YouTube video (or any video) and add its landmarks
as training data for a specific ISL word. This adds signer
diversity beyond the INCLUDE dataset.

Usage:
    python add_training_video.py --video "ISL_Hello.mp4" --word HELLO
    python add_training_video.py --video "ISL_Greetings.mp4" --word GOOD_MORNING --start 5 --end 15
"""

import argparse
import os
import sys
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.preprocessing import normalize_frame, convert_color
from src.landmark_extractor import LandmarkExtractor
from src.feature_engineer import sliding_windows


def process_video_segment(video_path: str, start_sec: float = 0, 
                          end_sec: float = None) -> list:
    """Extract landmarks from a video segment."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps) if end_sec else total_frames
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    extractor = LandmarkExtractor()
    landmarks_list = []
    frame_num = start_frame
    
    print(f"  Extracting landmarks from frame {start_frame} to {end_frame}...")
    
    while frame_num < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = normalize_frame(frame)
        rgb = convert_color(frame)
        features = extractor.extract_landmarks(rgb)
        landmarks_list.append(features)
        frame_num += 1
        
        if frame_num % 30 == 0:
            sys.stdout.write(f"\r  Frame {frame_num - start_frame}/{end_frame - start_frame}")
            sys.stdout.flush()
    
    print()
    extractor.release()
    cap.release()
    
    return landmarks_list


def main():
    parser = argparse.ArgumentParser(
        description="Add a YouTube/video file as training data for a specific ISL word",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python add_training_video.py --video "ISL_Hello.mp4" --word HELLO
  python add_training_video.py --video "ISL_Greetings.mp4" --word GOOD_MORNING --start 5 --end 15
  python add_training_video.py --video "video.mp4" --word HELLO --start 0 --end 3 --repeat 3
        """
    )
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--word', required=True, help='ISL word label (e.g., HELLO, GOOD_MORNING)')
    parser.add_argument('--start', type=float, default=0, help='Start time in seconds (default: 0)')
    parser.add_argument('--end', type=float, default=None, help='End time in seconds (default: end of video)')
    parser.add_argument('--repeat', type=int, default=1, 
                        help='If video contains multiple repetitions, how many sign instances (default: 1)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        return
    
    word = args.word.upper().replace(' ', '_')
    output_dir = os.path.join(config.DATA_DIR, word)
    os.makedirs(output_dir, exist_ok=True)
    
    existing = len([f for f in os.listdir(output_dir) if f.endswith('.npy')])
    
    print(f"\n{'='*55}")
    print(f"  Adding training data for: {word}")
    print(f"  Video: {args.video}")
    print(f"  Time: {args.start}s → {args.end or 'end'}s")
    print(f"  Existing samples: {existing}")
    print(f"{'='*55}\n")
    
    # Extract landmarks
    landmarks_list = process_video_segment(args.video, args.start, args.end)
    
    if len(landmarks_list) < config.SEQUENCE_LENGTH:
        print(f"[WARNING] Only {len(landmarks_list)} frames extracted. "
              f"Need at least {config.SEQUENCE_LENGTH} for a sequence.")
        if len(landmarks_list) == 0:
            return
    
    # Create sliding window sequences
    sequences = sliding_windows(landmarks_list)
    
    print(f"  Generated {len(sequences)} training sequences")
    
    # Save sequences
    saved = 0
    for seq in sequences:
        idx = existing + saved
        filepath = os.path.join(output_dir, f"youtube_{idx:04d}.npy")
        np.save(filepath, seq.astype(np.float32))
        saved += 1
    
    total = existing + saved
    print(f"  Saved {saved} new samples → {output_dir}")
    print(f"  Total samples for '{word}': {total}")
    
    # Update vocabulary if word is new
    vocab_path = os.path.join(os.path.dirname(config.DATA_DIR), 'vocab', 'words.json')
    if os.path.exists(vocab_path):
        import json
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        
        if word not in vocab['words']:
            vocab['words'].append(word)
            vocab['words'].sort()
            # Rebuild indices
            vocab['word_to_index'] = {w: i for i, w in enumerate(vocab['words'])}
            vocab['index_to_word'] = {str(i): w for i, w in enumerate(vocab['words'])}
            
            with open(vocab_path, 'w') as f:
                json.dump(vocab, f, indent=4)
            print(f"  [NEW WORD] Added '{word}' to vocabulary")
    
    print(f"\n  Next: Retrain the model with:")
    print(f"    python train.py --augment")


if __name__ == "__main__":
    main()
