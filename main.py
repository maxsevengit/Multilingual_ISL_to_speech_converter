"""
Main Application Entry Point for ISL Gesture Recognition.

Supports three modes:
  1. COLLECT — Record training data for a specific ISL word
  2. RECOGNIZE — Real-time continuous gesture recognition from webcam
  3. RECOGNIZE --video — Recognize gestures from a video file

Usage:
    python main.py --mode collect --word HELLO
    python main.py --mode recognize
    python main.py --mode recognize --video path/to/video.mp4
"""

import argparse
import sys
import cv2
import numpy as np
import config
from src.preprocessing import normalize_frame, convert_color
from src.landmark_extractor import LandmarkExtractor
from src.recognizer import GestureRecognizer
from src.dataset import collect_training_data
from src.model import load_model
from src.utils import load_vocabulary, FPSCounter, draw_info_panel


def _open_video_source(video_path: str = None):
    """
    Open a video source — webcam or video file.
    Uses the EXACT same VideoCapture interface for both,
    ensuring identical frame pipeline downstream.
    
    Args:
        video_path: Path to video file, or None for webcam.
    
    Returns:
        Tuple of (cv2.VideoCapture, is_webcam: bool, source_name: str).
    """
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video file: {video_path}")
            return None, False, ""
        source_name = video_path
        return cap, False, source_name
    else:
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam.")
            return None, True, ""
        source_name = "Webcam"
        return cap, True, source_name


def run_recognition(use_velocity: bool = False, video_path: str = None):
    """
    Run the ISL gesture recognition pipeline.
    
    Works IDENTICALLY for webcam and video file input —
    the only difference is the frame source. The entire
    preprocessing → landmark → recognition pipeline is
    shared between both modes.
    
    Controls:
        'q' — Quit
        'c' — Clear sentence
        'r' — Reset recognizer
        SPACE — Pause/resume (video file mode only)
    
    Args:
        use_velocity: Whether to use velocity-enhanced features.
        video_path: Path to video file, or None for webcam.
    """
    # ── Load model and vocabulary ────────────────────────────────────────────
    print("[INFO] Loading model...")
    model = load_model()
    if model is None:
        print("\n[ERROR] No trained model found!")
        print("  Please train a model first:")
        print("  1. Download data:  python download_dataset.py")
        print("  2. Process data:   python process_videos.py --input data/include_videos")
        print("  3. Train model:    python train.py --augment")
        return
    
    print("[INFO] Loading vocabulary...")
    vocab = load_vocabulary()
    label_names = vocab['words']
    
    if 'use_velocity' in vocab:
        use_velocity = vocab['use_velocity']
    
    print(f"[INFO] Vocabulary: {label_names}")
    print(f"[INFO] Velocity features: {'ON' if use_velocity else 'OFF'}")
    
    # ── Open video source ────────────────────────────────────────────────────
    cap, is_webcam, source_name = _open_video_source(video_path)
    if cap is None:
        return
    
    mode_label = "WEBCAM" if is_webcam else "VIDEO"
    print(f"[INFO] Source: {source_name}")
    
    # ── Initialize components (SAME for both modes) ──────────────────────────
    extractor = LandmarkExtractor()
    recognizer = GestureRecognizer(model, label_names, use_velocity)
    fps_counter = FPSCounter()
    paused = False
    
    # For video files: get total frames and FPS for proper playback
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0
    video_fps = cap.get(cv2.CAP_PROP_FPS) if not is_webcam else 30
    if video_fps <= 0:
        video_fps = 30
    current_frame_num = 0
    
    # Detect frame size for window resizing (portrait videos need fitting)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_display_h = 720
    resize_needed = (not is_webcam) and (frame_h > max_display_h or frame_w < 400)
    if resize_needed:
        scale = min(max_display_h / frame_h, 640 / frame_w)
        display_w = int(frame_w * scale)
        display_h = int(frame_h * scale)
        print(f"[INFO] Resizing display: {frame_w}x{frame_h} → {display_w}x{display_h}")
    
    print(f"\n{'='*50}")
    print(f"  ISL Real-Time Recognition — {mode_label}")
    if not is_webcam:
        print(f"  Video: {source_name}")
        print(f"  Press SPACE to pause/resume")
    print(f"  Press 'q' to quit, 'c' to clear sentence")
    print(f"{'='*50}\n")
    
    try:
        while True:
            if paused:
                key = cv2.waitKey(50) & 0xFF
                if key == ord(' '):
                    paused = False
                elif key == ord('q'):
                    break
                continue
            
            ret, frame = cap.read()
            if not ret:
                if not is_webcam:
                    print("\n[INFO] Video ended.")
                else:
                    print("[ERROR] Failed to read frame.")
                break
            
            current_frame_num += 1
            
            # ── Mirror only for webcam (natural interaction) ─────────────────
            if is_webcam:
                frame = cv2.flip(frame, 1)
            
            # ═════════════════════════════════════════════════════════════════
            #  IDENTICAL PIPELINE — same for webcam AND video file
            # ═════════════════════════════════════════════════════════════════
            
            # 1. Preprocess frame
            processed = normalize_frame(frame)
            frame_rgb = convert_color(processed, 'RGB')
            
            # 2. Extract landmarks
            landmarks, results = extractor.extract_landmarks_with_results(frame_rgb)
            
            # 3. Draw landmarks on display frame
            display = extractor.draw_landmarks(processed, results)
            
            # 4. Run recognition (only if hands detected)
            word = None
            confidence = 0.0
            
            if extractor.has_hands(results):
                word, confidence = recognizer.update(landmarks)
                
                if word is not None:
                    print(f"  ✓ Recognized: {word} ({confidence:.0%})")
            
            # ═════════════════════════════════════════════════════════════════
            
            # ── Get display info ─────────────────────────────────────────────
            current_pred, current_conf = recognizer.get_current_prediction()
            sentence = recognizer.get_sentence()
            fps = fps_counter.tick()
            
            # ── Draw info panel ──────────────────────────────────────────────
            display = draw_info_panel(
                display,
                prediction=current_pred,
                confidence=current_conf,
                sentence=sentence,
                fps=fps,
                mode=mode_label
            )
            
            # ── Show hand detection status ───────────────────────────────────
            if not extractor.has_hands(results):
                dh, dw = display.shape[:2]
                cv2.putText(display, "No hands detected",
                            (dw // 2 - int(dw * 0.15), dh // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, max(0.5, dw / 900),
                            (0, 0, 255), max(1, int(dw / 400)))
            
            # ── Video progress bar (for video files only) ────────────────────
            if not is_webcam and total_frames > 0:
                progress = current_frame_num / total_frames
                bar_width = display.shape[1] - 40
                bar_y = display.shape[0] - 15
                cv2.rectangle(display, (20, bar_y), (20 + bar_width, bar_y + 8),
                              (50, 50, 50), -1)
                cv2.rectangle(display, (20, bar_y),
                              (20 + int(bar_width * progress), bar_y + 8),
                              (0, 200, 100), -1)
            
            # ── Resize for portrait/large videos ─────────────────────────────
            if resize_needed:
                display = cv2.resize(display, (display_w, display_h))
            
            cv2.imshow("ISL Gesture Recognition", display)
            
            # ── Handle key input ─────────────────────────────────────────────
            # Webcam: waitKey(1) for real-time
            # Video: match actual video FPS for natural playback
            wait_ms = 1 if is_webcam else max(1, int(1000 / video_fps))
            key = cv2.waitKey(wait_ms) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                recognizer.clear_sentence()
                print("  [CLEARED] Sentence reset.")
            elif key == ord('r'):
                recognizer.reset()
                print("  [RESET] Recognizer reset.")
            elif key == ord(' ') and not is_webcam:
                paused = True
                print("  [PAUSED] Press SPACE to resume.")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.release()
    
    # ── Print final output ───────────────────────────────────────────────────
    final_sentence = recognizer.get_sentence()
    if final_sentence:
        print(f"\n{'='*50}")
        print(f"  Final recognized words: {final_sentence}")
        print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="ISL Gesture Recognition — Real-Time Indian Sign Language Translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Collect training data:
    python main.py --mode collect --word HELLO
    python main.py --mode collect --word WATER --samples 40
  
  Real-time recognition (webcam):
    python main.py --mode recognize
  
  Recognize from video file:
    python main.py --mode recognize --video path/to/isl_video.mp4
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['collect', 'recognize'],
                        help='Operation mode: collect training data or recognize gestures')
    parser.add_argument('--word', type=str, default=None,
                        help='Word to collect data for (required in collect mode)')
    parser.add_argument('--samples', type=int, default=config.SAMPLES_PER_WORD,
                        help=f'Number of samples to collect (default: {config.SAMPLES_PER_WORD})')
    parser.add_argument('--velocity', action='store_true',
                        help='Use velocity-enhanced features')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video file for recognition (omit to use webcam)')
    
    args = parser.parse_args()
    
    if args.mode == 'collect':
        if args.word is None:
            print("[ERROR] --word is required in collect mode!")
            print("  Example: python main.py --mode collect --word HELLO")
            sys.exit(1)
        
        collect_training_data(args.word, args.samples)
    
    elif args.mode == 'recognize':
        run_recognition(args.velocity, args.video)


if __name__ == "__main__":
    main()
