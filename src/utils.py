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
    All positions scale relative to frame size so it works with
    any resolution — landscape, portrait, or square.
    """
    h, w = frame.shape[:2]
    output = frame.copy()
    
    # Scale factor based on frame width (reference: 640px)
    s = max(w / 640, 0.5)
    pad = int(10 * s)
    
    # ── Top bar: Mode + FPS ──────────────────────────────────────────────────
    top_h = int(45 * s)
    cv2.rectangle(output, (0, 0), (w, top_h), (40, 40, 40), -1)
    cv2.putText(output, f"ISL | {mode}", (pad, int(28 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (0, 255, 200), max(1, int(2 * s)))
    fps_text = f"FPS: {fps:.0f}"
    cv2.putText(output, fps_text, (w - int(110 * s), int(28 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (200, 200, 200), max(1, int(s)))
    
    if mode in ("RECOGNIZE", "WEBCAM", "VIDEO"):
        # ── Prediction box ───────────────────────────────────────────────────
        if prediction:
            box_y1 = top_h + pad
            box_y2 = box_y1 + int(70 * s)
            box_x2 = min(w - pad, int(340 * s))
            
            # Semi-transparent background
            overlay = output.copy()
            cv2.rectangle(overlay, (pad, box_y1), (box_x2, box_y2), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
            cv2.rectangle(output, (pad, box_y1), (box_x2, box_y2), (0, 255, 200), 2)
            
            # Word — large and bold
            word_font_scale = min(1.2 * s, (box_x2 - pad - 10) / (len(prediction) * 22 + 1))
            word_font_scale = max(word_font_scale, 0.5)
            cv2.putText(output, prediction, (pad + int(8 * s), box_y2 - int(25 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, word_font_scale, (0, 255, 200),
                        max(2, int(2.5 * s)))
            
            # Confidence percentage
            conf_text = f"{confidence:.0%}"
            conf_color = (0, 255, 0) if confidence > 0.8 else (0, 200, 255)
            cv2.putText(output, conf_text, (pad + int(8 * s), box_y1 + int(20 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, conf_color,
                        max(1, int(1.5 * s)))
            
            # Confidence bar
            bar_y = box_y2 - int(12 * s)
            bar_x1 = box_x2 - int(110 * s)
            bar_x2 = box_x2 - int(10 * s)
            bar_fill = bar_x1 + int((bar_x2 - bar_x1) * confidence)
            cv2.rectangle(output, (bar_x1, bar_y), (bar_x2, bar_y + int(8 * s)),
                          (60, 60, 60), -1)
            cv2.rectangle(output, (bar_x1, bar_y), (bar_fill, bar_y + int(8 * s)),
                          conf_color, -1)
        
        # ── Sentence output bar (bottom) ─────────────────────────────────────
        if sentence:
            sent_h = int(45 * s)
            overlay = output.copy()
            cv2.rectangle(overlay, (0, h - sent_h), (w, h), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.8, output, 0.2, 0, output)
            sentence_text = " > ".join(sentence[-6:])
            cv2.putText(output, sentence_text, (pad, h - int(15 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (255, 255, 255),
                        max(1, int(1.5 * s)))
    
    elif mode == "COLLECT":
        box_y1 = top_h + pad
        box_y2 = box_y1 + int(70 * s)
        cv2.rectangle(output, (pad, box_y1), (int(380 * s), box_y2),
                      (30, 30, 30), -1)
        cv2.rectangle(output, (pad, box_y1), (int(380 * s), box_y2),
                      (0, 200, 255), 2)
        
        if collecting_word:
            cv2.putText(output, f"Collecting: {collecting_word}",
                        (pad + 10, box_y1 + int(30 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (0, 200, 255),
                        max(1, int(2 * s)))
            cv2.putText(output, f"Samples: {sample_count}/{config.SAMPLES_PER_WORD}",
                        (pad + 10, box_y2 - int(10 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (200, 200, 200),
                        max(1, int(s)))
        else:
            cv2.putText(output, "Press 's' to start",
                        (pad + 10, box_y1 + int(35 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (200, 200, 200),
                        max(1, int(1.5 * s)))
    
    return output
