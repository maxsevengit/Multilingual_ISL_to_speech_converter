"""
Utility functions for ISL Gesture Recognition.

Drawing helpers, logging, vocabulary management, and 
miscellaneous helper functions.
"""

import json
import time
import cv2
import numpy as np
import config


def load_vocabulary(path: str = None) -> dict:
    """
    Load vocabulary mapping from JSON file.
    
    Args:
        path: Path to vocabulary JSON. Defaults to config.VOCAB_PATH.
    
    Returns:
        Dictionary with 'words', 'word_to_index', 'index_to_word' keys.
    """
    if path is None:
        path = config.VOCAB_PATH
    
    with open(path, 'r') as f:
        vocab = json.load(f)
    
    return vocab


def save_vocabulary(vocab: dict, path: str = None):
    """
    Save vocabulary mapping to JSON file.
    
    Args:
        vocab: Vocabulary dictionary.
        path: Output path. Defaults to config.VOCAB_PATH.
    """
    if path is None:
        path = config.VOCAB_PATH
    
    with open(path, 'w') as f:
        json.dump(vocab, f, indent=4)


def add_word_to_vocabulary(word: str, path: str = None) -> dict:
    """
    Add a new word to the vocabulary if it doesn't exist.
    
    Args:
        word: Word to add (will be uppercased).
        path: Vocabulary file path.
    
    Returns:
        Updated vocabulary dictionary.
    """
    vocab = load_vocabulary(path)
    word = word.upper()
    
    if word not in vocab['word_to_index']:
        new_index = len(vocab['words'])
        vocab['words'].append(word)
        vocab['word_to_index'][word] = new_index
        vocab['index_to_word'][str(new_index)] = word
        save_vocabulary(vocab, path)
    
    return vocab


class FPSCounter:
    """Tracks and smooths FPS for display."""
    
    def __init__(self, smoothing: int = 30):
        self.smoothing = smoothing
        self.timestamps = []
    
    def tick(self) -> float:
        """Record a frame and return smoothed FPS."""
        now = time.time()
        self.timestamps.append(now)
        
        # Keep only recent timestamps
        if len(self.timestamps) > self.smoothing:
            self.timestamps = self.timestamps[-self.smoothing:]
        
        if len(self.timestamps) < 2:
            return 0.0
        
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed == 0:
            return 0.0
        
        return (len(self.timestamps) - 1) / elapsed


def draw_info_panel(frame: np.ndarray, prediction: str = None, 
                    confidence: float = 0.0, sentence: list = None,
                    fps: float = 0.0, mode: str = "RECOGNIZE",
                    collecting_word: str = None, sample_count: int = 0) -> np.ndarray:
    """
    Draw an information overlay panel on the frame.
    
    Args:
        frame: BGR frame to draw on.
        prediction: Current predicted word.
        confidence: Prediction confidence (0-1).
        sentence: List of recognized words.
        fps: Current FPS.
        mode: Current mode ('RECOGNIZE' or 'COLLECT').
        collecting_word: Word being collected (in collect mode).
        sample_count: Number of samples collected.
    
    Returns:
        Frame with overlay drawn.
    """
    h, w = frame.shape[:2]
    output = frame.copy()
    
    # ── Top bar: Mode + FPS ──────────────────────────────────────────────────
    cv2.rectangle(output, (0, 0), (w, 50), (40, 40, 40), -1)
    cv2.putText(output, f"ISL Recognition | Mode: {mode}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
    cv2.putText(output, f"FPS: {fps:.1f}", (w - 130, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    if mode == "RECOGNIZE":
        # ── Prediction box ───────────────────────────────────────────────────
        if prediction:
            # Background
            cv2.rectangle(output, (10, 60), (350, 130), (30, 30, 30), -1)
            cv2.rectangle(output, (10, 60), (350, 130), (0, 255, 200), 2)
            
            # Word
            cv2.putText(output, prediction, (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 200), 3)
            
            # Confidence bar
            bar_width = int(200 * confidence)
            bar_color = (0, 255, 0) if confidence > 0.8 else (0, 200, 255) if confidence > 0.6 else (0, 100, 255)
            cv2.rectangle(output, (200, 80), (200 + bar_width, 100), bar_color, -1)
            cv2.rectangle(output, (200, 80), (400, 100), (100, 100, 100), 1)
            cv2.putText(output, f"{confidence:.0%}", (405, 97),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # ── Sentence output bar (bottom) ─────────────────────────────────────
        if sentence:
            cv2.rectangle(output, (0, h - 50), (w, h), (40, 40, 40), -1)
            sentence_text = " → ".join(sentence[-8:])  # Show last 8 words
            cv2.putText(output, sentence_text, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    elif mode == "COLLECT":
        # ── Collection info ──────────────────────────────────────────────────
        cv2.rectangle(output, (10, 60), (400, 140), (30, 30, 30), -1)
        cv2.rectangle(output, (10, 60), (400, 140), (0, 200, 255), 2)
        
        if collecting_word:
            cv2.putText(output, f"Collecting: {collecting_word}", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.putText(output, f"Samples: {sample_count}/{config.SAMPLES_PER_WORD}", 
                        (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        else:
            cv2.putText(output, "Press 's' to start recording", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # ── Controls help (bottom-right) ─────────────────────────────────────────
    help_text = "Q: Quit"
    if mode == "COLLECT":
        help_text = "S: Record | R: Reset | Q: Quit"
    cv2.putText(output, help_text, (w - 300, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    return output
