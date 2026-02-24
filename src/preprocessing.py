"""
Image Preprocessing Module for ISL Gesture Recognition.

Handles frame normalization, color space conversion, and
data augmentation for training robustness.
"""

import cv2
import numpy as np
import config


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Resize frame to target resolution and apply lighting normalization.
    
    Args:
        frame: Raw BGR frame from webcam.
    
    Returns:
        Normalized BGR frame at target resolution.
    """
    # Resize to standard dimensions
    resized = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
    
    # Convert to LAB color space for lighting normalization
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_normalized = clahe.apply(l_channel)
    
    # Merge back and convert to BGR
    lab_normalized = cv2.merge([l_normalized, a_channel, b_channel])
    normalized = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)
    
    return normalized


def convert_color(frame: np.ndarray, target: str = 'RGB') -> np.ndarray:
    """
    Convert frame between BGR and RGB color spaces.
    
    Args:
        frame: Input image frame.
        target: Target color space ('RGB' or 'BGR').
    
    Returns:
        Color-converted frame.
    """
    if target.upper() == 'RGB':
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif target.upper() == 'BGR':
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"Unsupported target color space: {target}")


def augment_frame(frame: np.ndarray) -> np.ndarray:
    """
    Apply random augmentation to a frame for training data diversity.
    Applies random brightness, contrast adjustments, and horizontal flipping.
    
    Args:
        frame: Input BGR frame.
    
    Returns:
        Augmented BGR frame.
    """
    augmented = frame.copy()
    
    # Random brightness adjustment (-30 to +30)
    brightness = np.random.randint(-30, 31)
    augmented = np.clip(augmented.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
    
    # Random contrast adjustment (0.8 to 1.2)
    contrast = np.random.uniform(0.8, 1.2)
    mean = np.mean(augmented)
    augmented = np.clip((augmented.astype(np.float32) - mean) * contrast + mean, 0, 255).astype(np.uint8)
    
    # Random horizontal flip (50% chance) — Note: for sign language,
    # flipping mirrors left/right hand, so we also swap hand landmark data later
    if np.random.random() > 0.5:
        augmented = cv2.flip(augmented, 1)
    
    return augmented


def add_gaussian_noise(frame: np.ndarray, mean: float = 0, sigma: float = 10) -> np.ndarray:
    """
    Add Gaussian noise to a frame for training augmentation.
    
    Args:
        frame: Input BGR frame.
        mean: Mean of noise distribution.
        sigma: Standard deviation of noise distribution.
    
    Returns:
        Noisy BGR frame.
    """
    noise = np.random.normal(mean, sigma, frame.shape).astype(np.int16)
    noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy
