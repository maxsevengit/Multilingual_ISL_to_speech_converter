"""
Main Application Entry Point for ISL Gesture Recognition.

Supports two modes:
  1. COLLECT — Record training data for a specific ISL word
  2. RECOGNIZE — Real-time continuous gesture recognition from webcam

Usage:
    python main.py --mode collect --word HELLO
    python main.py --mode recognize
    python main.py --mode recognize --velocity
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


def run_recognition(use_velocity: bool = False):
    """
    Run the real-time ISL gesture recognition pipeline.
    
    Opens webcam, extracts landmarks, runs LSTM inference, and
    displays recognized words with confidence scores.
    
    Controls:
        'q' — Quit
        'c' — Clear sentence
        'r' — Reset recognizer
    
    Args:
        use_velocity: Whether to use velocity-enhanced features (must match training).
    """
    # ── Load model and vocabulary ────────────────────────────────────────────
    print("[INFO] Loading model...")
    model = load_model()
    if model is None:
        print("\n[ERROR] No trained model found!")
        print("  Please train a model first:")
        print("  1. Collect data: python main.py --mode collect --word HELLO")
        print("  2. Repeat for at least 3 words")
        print("  3. Train: python train.py")
        return
    
    print("[INFO] Loading vocabulary...")
    vocab = load_vocabulary()
    label_names = vocab['words']
    
    # Check if model was trained with velocity features
    if 'use_velocity' in vocab:
        use_velocity = vocab['use_velocity']
    
    print(f"[INFO] Vocabulary: {label_names}")
    print(f"[INFO] Velocity features: {'ON' if use_velocity else 'OFF'}")
    
    # ── Initialize components ────────────────────────────────────────────────
    extractor = LandmarkExtractor()
    recognizer = GestureRecognizer(model, label_names, use_velocity)
    fps_counter = FPSCounter()
    
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return
    
    print("\n" + "=" * 50)
    print("  ISL Real-Time Recognition")
    print("  Press 'q' to quit, 'c' to clear sentence")
    print("=" * 50 + "\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame.")
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # ── Preprocess ───────────────────────────────────────────────────
            processed = normalize_frame(frame)
            frame_rgb = convert_color(processed, 'RGB')
            
            # ── Extract landmarks ────────────────────────────────────────────
            landmarks, results = extractor.extract_landmarks_with_results(frame_rgb)
            
            # ── Draw landmarks on display frame ─────────────────────────────
            display = extractor.draw_landmarks(processed, results)
            
            # ── Run recognition (only if hands are detected) ─────────────────
            word = None
            confidence = 0.0
            
            if extractor.has_hands(results):
                word, confidence = recognizer.update(landmarks)
                
                if word is not None:
                    print(f"  ✓ Recognized: {word} ({confidence:.0%})")
            
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
                mode="RECOGNIZE"
            )
            
            # ── Show hand detection status ───────────────────────────────────
            if not extractor.has_hands(results):
                cv2.putText(display, "No hands detected - show your hands!", 
                            (config.FRAME_WIDTH // 2 - 200, config.FRAME_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("ISL Gesture Recognition", display)
            
            # ── Handle key input ─────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                recognizer.clear_sentence()
                print("  [CLEARED] Sentence reset.")
            elif key == ord('r'):
                recognizer.reset()
                print("  [RESET] Recognizer reset.")
    
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
  
  Real-time recognition:
    python main.py --mode recognize
    python main.py --mode recognize --velocity
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
    
    args = parser.parse_args()
    
    if args.mode == 'collect':
        if args.word is None:
            print("[ERROR] --word is required in collect mode!")
            print("  Example: python main.py --mode collect --word HELLO")
            sys.exit(1)
        
        collect_training_data(args.word, args.samples)
    
    elif args.mode == 'recognize':
        run_recognition(args.velocity)


if __name__ == "__main__":
    main()
