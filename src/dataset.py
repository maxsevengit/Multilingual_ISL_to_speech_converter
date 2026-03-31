"""
Dataset Management Module.

Handles data collection from webcam, dataset loading/saving,
and preparation of training data from collected gesture sequences.
"""

import os
import json
import time
import re
import cv2
import numpy as np
import config
from src.preprocessing import normalize_frame, convert_color
from src.landmark_extractor import LandmarkExtractor
from src.feature_engineer import create_sequence, sliding_windows, build_feature_vector
from src.utils import load_vocabulary, add_word_to_vocabulary, FPSCounter, draw_info_panel


def normalize_label_name(name: str) -> str:
    """
    Normalize label names so numbered and example-prefixed folders
    map to the same canonical word.
    """
    if name is None:
        return ""

    label = str(name).strip().upper().replace(" ", "_")

    # Strip common prefixes like "48._" or "EX._"
    label = re.sub(r"^\d+\._", "", label)
    label = re.sub(r"^EX\._", "", label)
    label = re.sub(r"^\d+[_\.]+", "", label)

    return label


def normalize_labels(X: np.ndarray, y: np.ndarray, label_names: list):
    """
    Merge duplicated labels after normalization and remap y accordingly.
    """
    if X is None or y is None or label_names is None:
        return X, y, label_names

    normalized = [normalize_label_name(name) for name in label_names]

    label_to_new_index = {}
    new_label_names = []
    old_to_new = {}

    for old_idx, norm_name in enumerate(normalized):
        if norm_name not in label_to_new_index:
            label_to_new_index[norm_name] = len(new_label_names)
            new_label_names.append(norm_name)
        old_to_new[old_idx] = label_to_new_index[norm_name]

    # Remap y
    y_new = np.array([old_to_new[int(idx)] for idx in y], dtype=np.int32)

    return X, y_new, new_label_names


def filter_labels(X: np.ndarray, y: np.ndarray, label_names: list, allowed: list):
    """
    Keep only labels present in the allowed list (after normalization).
    Remap labels to contiguous indices.
    """
    if X is None or y is None or label_names is None:
        return X, y, label_names

    allowed_set = {normalize_label_name(name) for name in allowed}
    normalized = [normalize_label_name(name) for name in label_names]
    normalized_set = set(normalized)

    # Build mapping for allowed labels (preserve allowed list order)
    new_label_names = []
    for name in allowed:
        norm = normalize_label_name(name)
        if norm in allowed_set and norm in normalized_set and norm not in new_label_names:
            new_label_names.append(norm)
    new_index = {name: i for i, name in enumerate(new_label_names)}

    keep_mask = np.array([normalized[int(idx)] in allowed_set for idx in y])
    X_filtered = X[keep_mask]
    y_filtered = y[keep_mask]

    y_new = np.array([new_index[normalized[int(idx)]] for idx in y_filtered], dtype=np.int32)

    return X_filtered, y_new, new_label_names


def collect_training_data(word: str, num_samples: int = None):
    """
    Interactive data collection mode: record gesture sequences via webcam.
    
    Opens the webcam and lets the user record multiple samples of a gesture.
    Each sample is a sequence of landmark frames captured over SEQUENCE_LENGTH frames.
    
    Controls:
        's' — Start recording a sample (after countdown)
        'r' — Reset / discard current recording
        'q' — Quit and save collected data
    
    Args:
        word: The ISL word/gesture to collect data for.
        num_samples: Number of samples to collect. Defaults to config.SAMPLES_PER_WORD.
    """
    if num_samples is None:
        num_samples = config.SAMPLES_PER_WORD
    
    word = word.upper()
    
    # Ensure word is in vocabulary
    add_word_to_vocabulary(word)
    
    # Setup save directory
    word_dir = os.path.join(config.DATA_DIR, word)
    os.makedirs(word_dir, exist_ok=True)
    
    # Count existing samples
    existing = len([f for f in os.listdir(word_dir) if f.endswith('.npy')])
    print(f"[INFO] Existing samples for '{word}': {existing}")
    
    # Initialize components
    extractor = LandmarkExtractor()
    fps_counter = FPSCounter()
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return
    
    sample_count = existing
    recording = False
    countdown_start = None
    frame_buffer = []
    
    print(f"\n{'='*50}")
    print(f"  Data Collection Mode")
    print(f"  Word: {word}")
    print(f"  Target: {num_samples} samples")
    print(f"  Press 's' to start recording, 'q' to quit")
    print(f"{'='*50}\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # ── Mirror for webcam (natural interaction) ──────────────────────
            frame = cv2.flip(frame, 1)
            
            # Preprocess
            frame = normalize_frame(frame)
            frame_rgb = convert_color(frame, 'RGB')
            
            # Extract landmarks (pass mirrored=True for standard webcam)
            features, results = extractor.extract_landmarks_with_results(frame_rgb, mirrored=True)
            
            # Draw landmarks on frame
            display = extractor.draw_landmarks(frame, results)
            
            fps = fps_counter.tick()
            
            # Handle countdown before recording
            if countdown_start is not None:
                elapsed = time.time() - countdown_start
                remaining = config.COLLECTION_COUNTDOWN - elapsed
                
                if remaining > 0:
                    # Show countdown
                    cv2.putText(display, str(int(remaining) + 1), 
                                (config.FRAME_WIDTH // 2 - 30, config.FRAME_HEIGHT // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                else:
                    # Start recording
                    recording = True
                    countdown_start = None
                    frame_buffer = []
                    print(f"  [REC] Recording sample {sample_count + 1}...")
            
            # Record frames
            if recording:
                frame_buffer.append(features)
                
                # Show recording indicator
                cv2.circle(display, (config.FRAME_WIDTH - 30, 75), 10, (0, 0, 255), -1)
                cv2.putText(display, f"REC {len(frame_buffer)}/{config.SEQUENCE_LENGTH}",
                            (config.FRAME_WIDTH - 160, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Check if sequence is complete
                if len(frame_buffer) >= config.SEQUENCE_LENGTH:
                    # Save the sample
                    sequence = np.array(frame_buffer[:config.SEQUENCE_LENGTH], dtype=np.float32)
                    save_path = os.path.join(word_dir, f"sample_{sample_count:04d}.npy")
                    np.save(save_path, sequence)
                    
                    sample_count += 1
                    recording = False
                    frame_buffer = []
                    
                    collected_this_session = sample_count - existing
                    print(f"  [SAVED] Sample {collected_this_session}/{num_samples} saved to {save_path}")
                    
                    if collected_this_session >= num_samples:
                        print(f"\n  [DONE] Collected {num_samples} new samples for '{word}'!")
                        break
            
            # Draw info panel
            display = draw_info_panel(
                display, mode="COLLECT",
                collecting_word=word, sample_count=sample_count,
                fps=fps
            )
            
            cv2.imshow("ISL Data Collection", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and not recording and countdown_start is None:
                countdown_start = time.time()
                print(f"  [COUNTDOWN] Starting in {config.COLLECTION_COUNTDOWN} seconds...")
            elif key == ord('r'):
                recording = False
                countdown_start = None
                frame_buffer = []
                print("  [RESET] Recording discarded.")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.release()
    
    print(f"\n[INFO] Total samples for '{word}': {sample_count}")


def load_dataset(data_dir: str = None):
    """
    Load all collected training data from disk.
    
    Expects directory structure:
        data/raw/WORD_NAME/sample_0000.npy
        data/raw/WORD_NAME/sample_0001.npy
        ...
    
    Args:
        data_dir: Path to data directory. Defaults to config.DATA_DIR.
    
    Returns:
        Tuple of (X, y, label_names) where:
          - X: numpy array of shape (num_samples, seq_length, num_features)
          - y: numpy array of integer labels
          - label_names: list of word names corresponding to label indices
    """
    if data_dir is None:
        data_dir = config.DATA_DIR
    
    if not os.path.exists(data_dir):
        print(f"[WARNING] Data directory not found: {data_dir}")
        return None, None, None
    
    X_all = []
    y_all = []
    label_names = []
    
    # Scan word directories
    word_dirs = sorted([d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d))])
    
    if not word_dirs:
        print("[WARNING] No word directories found in data directory.")
        return None, None, None
    
    for label_idx, word in enumerate(word_dirs):
        word_path = os.path.join(data_dir, word)
        sample_files = sorted([f for f in os.listdir(word_path) if f.endswith('.npy')])
        
        if not sample_files:
            continue
        
        loaded = 0
        skipped_zeros = 0
        skipped_shape = 0
        
        for sample_file in sample_files:
            sample_path = os.path.join(word_path, sample_file)
            sequence = np.load(sample_path)
            
            # Ensure correct shape
            if sequence.shape != (config.SEQUENCE_LENGTH, config.NUM_FEATURES):
                skipped_shape += 1
                continue
            
            # Filter out all-zeros samples (failed MediaPipe detection)
            if np.all(sequence == 0):
                skipped_zeros += 1
                continue
            
            # Also filter mostly-zero samples (< 10% non-zero values)
            nonzero_frac = np.count_nonzero(sequence) / sequence.size
            if nonzero_frac < 0.05:
                skipped_zeros += 1
                continue
            
            X_all.append(sequence)
            y_all.append(label_idx)
            loaded += 1
        
        if loaded > 0:
            label_names.append(word)
            # Fix label indices: use the actual label index (length - 1)
            # since we only append label_names when loaded > 0
            actual_idx = len(label_names) - 1
            # Re-assign the correct index for samples just added
            for i in range(len(y_all) - loaded, len(y_all)):
                y_all[i] = actual_idx
            
            status = f"  Loading '{word}': {loaded} real samples"
            if skipped_zeros > 0:
                status += f" (filtered {skipped_zeros} all-zeros)"
            if skipped_shape > 0:
                status += f" (skipped {skipped_shape} bad-shape)"
            print(status)
        else:
            print(f"  SKIPPING '{word}': 0 real samples ({skipped_zeros} all-zeros, {skipped_shape} bad-shape)")
    
    if not X_all:
        print("[WARNING] No valid samples found.")
        return None, None, None
    
    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int32)
    
    print(f"\n[INFO] Dataset loaded: {X.shape[0]} samples, {len(label_names)} classes")
    print(f"  Shape: X={X.shape}, y={y.shape}")
    print(f"  Classes: {label_names}")
    
    return X, y, label_names


def save_dataset_npz(X: np.ndarray, y: np.ndarray, label_names: list, 
                     path: str = None):
    """
    Save dataset as a compressed .npz file for quick loading.
    
    Args:
        X: Feature sequences.
        y: Labels.
        label_names: List of word names.
        path: Save path. Defaults to config.DATASET_PATH.
    """
    if path is None:
        path = config.DATASET_PATH
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, X=X, y=y, label_names=np.array(label_names))
    print(f"[INFO] Dataset saved to {path}")


def load_dataset_npz(path: str = None):
    """
    Load dataset from .npz file.
    
    Args:
        path: Path to .npz file. Defaults to config.DATASET_PATH.
    
    Returns:
        Tuple of (X, y, label_names).
    """
    if path is None:
        path = config.DATASET_PATH
    
    if not os.path.exists(path):
        return None, None, None
    
    data = np.load(path, allow_pickle=True)
    X = data['X']
    y = data['y']
    label_names = list(data['label_names'])
    
    print(f"[INFO] Dataset loaded from {path}")
    print(f"  Shape: X={X.shape}, y={y.shape}")
    print(f"  Classes: {label_names}")
    
    return X, y, label_names
