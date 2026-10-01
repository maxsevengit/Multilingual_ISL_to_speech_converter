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
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import config
from src.logutil import get_logger
from src.preprocessing import normalize_frame, convert_color
from src.landmark_extractor import LandmarkExtractor
from src.recognizer import GestureRecognizer
from src.dataset import collect_training_data
from src.model_bundle import BundleMismatchError, load_model_bundle
from src.translator import ISLTranslator
from src.utils import FPSCounter, draw_info_panel

log = get_logger("main")


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
            log.info(f"[ERROR] Cannot open video file: {video_path}")
            return None, False, ""
        source_name = video_path
        return cap, False, source_name
    else:
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        if not cap.isOpened():
            log.info("[ERROR] Cannot open webcam.")
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
    log.info("[INFO] Loading model bundle...")
    try:
        bundle = load_model_bundle()
    except BundleMismatchError as exc:
        log.info(f"\n[ERROR] {exc}")
        return
    except Exception as exc:
        log.info(f"\n[ERROR] Could not load model bundle: {exc}")
        log.info("  Train first: python train.py --dataset include")
        return

    model = bundle.model
    vocab = bundle.vocabulary
    label_names = vocab["words"]
    use_velocity = bool(bundle.config["use_velocity"])
    log.info(f"[INFO] Bundle: {bundle.directory}")
    log.info(f"[INFO] Architecture: {bundle.config['model_type']}")
    log.info(f"[INFO] Vocabulary: {label_names}")
    log.info(f"[INFO] Velocity features: {'ON' if use_velocity else 'OFF'}")
    
    # ── Open video source ────────────────────────────────────────────────────
    cap, is_webcam, source_name = _open_video_source(video_path)
    if cap is None:
        return
    
    mode_label = "WEBCAM" if is_webcam else "VIDEO"
    log.info(f"[INFO] Source: {source_name}")
    
    # ── Initialize components (SAME for both modes) ──────────────────────────
    extractor = LandmarkExtractor()
    recognizer = GestureRecognizer(
        model, label_names, use_velocity_features=use_velocity
    )
    translator = ISLTranslator()
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
        log.info("[INFO] Resizing display: %sx%s to %sx%s", frame_w, frame_h, display_w, display_h)
    
    log.info(f"\n{'='*50}")
    log.info("  ISL recognition - %s", mode_label)
    if not is_webcam:
        log.info(f"  Video: {source_name}")
        log.info(f"  Press SPACE to pause/resume")
    log.info(f"  Press 'q' to quit, 'c' to clear")
    log.info(f"  Press 't' to translate sentence")
    log.info(f"  Press 'l' to change language")
    log.info(f"{'='*50}\n")
    
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
                    log.info("\n[INFO] Video ended.")
                else:
                    log.info("[ERROR] Failed to read frame.")
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
            
            # 2. Extract landmarks (Segmentation is handled internally by extractor)
            landmarks, results = extractor.extract_landmarks_with_results(frame_rgb, mirrored=is_webcam)
            
            # 3. Draw landmarks on display frame
            display = extractor.draw_landmarks(processed, results)
            
            # 4. Run recognition (only if hands detected)
            word = None
            confidence = 0.0
            
            if extractor.has_hands(results):
                raw_word, raw_confidence = recognizer.update(landmarks)
                
                # Treat 'IDLE' as a background class — do not add to sentence
                if raw_word is not None and raw_word != "IDLE":
                    word = raw_word
                    confidence = raw_confidence
                    log.info("  Recognized: %s (%.0f%%)", word, confidence * 100)
            
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
                mode=mode_label,
                translation=translator.last_translation,
                target_language=translator.get_current_language(),
                model_name=str(bundle.config.get("model_type", "")).upper(),
                camera_status="Camera: open" if cap.isOpened() else "Camera: no signal",
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
                log.info("  [CLEARED] Sentence reset.")
            elif key == ord('r'):
                recognizer.reset()
                log.info("  [RESET] Recognizer reset.")
            elif key == ord('t'):
                if sentence:
                    log.info(f"  [TRANSLATE] Generating {translator.get_current_language()} speech...")
                    translator.translate_and_speak_async(sentence)
                else:
                    log.info("  [WARNING] Empty sentence, nothing to translate.")
            elif key == ord('l'):
                new_lang = translator.next_language()
                log.info(f"  [LANGUAGE] Switched to {new_lang}")
            elif key == ord(' ') and not is_webcam:
                paused = True
                log.info("  [PAUSED] Press SPACE to resume.")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.release()
    
    # ── Print final output ───────────────────────────────────────────────────
    final_sentence = recognizer.get_sentence()
    if final_sentence:
        log.info(f"\n{'='*50}")
        log.info(f"  Final recognized words: {final_sentence}")
        log.info(f"{'='*50}")


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
            log.info("[ERROR] --word is required in collect mode!")
            log.info("  Example: python main.py --mode collect --word HELLO")
            sys.exit(1)
        
        collect_training_data(args.word, args.samples)
    
    elif args.mode == 'recognize':
        run_recognition(args.velocity, args.video)


if __name__ == "__main__":
    main()
