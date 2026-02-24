"""
Temporal Feature Engineering Module.

Converts raw landmark sequences into enriched feature vectors
with motion (velocity) information for temporal modeling.
"""

import numpy as np
import config


def create_sequence(landmark_buffer: list, seq_length: int = None) -> np.ndarray:
    """
    Create a fixed-length sequence from a buffer of landmark frames.
    Pads with zeros if buffer is shorter than seq_length, or 
    truncates from the beginning if longer.
    
    Args:
        landmark_buffer: List of landmark arrays, each of shape (NUM_FEATURES,).
        seq_length: Target sequence length. Defaults to config.SEQUENCE_LENGTH.
    
    Returns:
        Numpy array of shape (seq_length, NUM_FEATURES).
    """
    if seq_length is None:
        seq_length = config.SEQUENCE_LENGTH
    
    if len(landmark_buffer) == 0:
        return np.zeros((seq_length, config.NUM_FEATURES), dtype=np.float32)
    
    # Stack into 2D array
    buffer_array = np.array(landmark_buffer, dtype=np.float32)
    
    if len(buffer_array) >= seq_length:
        # Take the most recent seq_length frames
        return buffer_array[-seq_length:]
    else:
        # Pad with zeros at the beginning
        padding = np.zeros((seq_length - len(buffer_array), config.NUM_FEATURES), dtype=np.float32)
        return np.vstack([padding, buffer_array])


def compute_velocity(sequence: np.ndarray) -> np.ndarray:
    """
    Compute frame-to-frame velocity (first derivative) from a landmark sequence.
    The first frame velocity is set to zero.
    
    Args:
        sequence: Array of shape (seq_length, NUM_FEATURES).
    
    Returns:
        Velocity array of shape (seq_length, NUM_FEATURES).
    """
    velocity = np.zeros_like(sequence)
    velocity[1:] = sequence[1:] - sequence[:-1]
    return velocity


def compute_acceleration(sequence: np.ndarray) -> np.ndarray:
    """
    Compute frame-to-frame acceleration (second derivative) from a landmark sequence.
    
    Args:
        sequence: Array of shape (seq_length, NUM_FEATURES).
    
    Returns:
        Acceleration array of shape (seq_length, NUM_FEATURES).
    """
    velocity = compute_velocity(sequence)
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    return acceleration


def build_feature_vector(sequence: np.ndarray) -> np.ndarray:
    """
    Build an enriched feature vector by concatenating position and velocity.
    
    This doubles the feature dimension: each frame contains both the raw
    landmark positions and their frame-to-frame changes (motion cues).
    
    Args:
        sequence: Array of shape (seq_length, NUM_FEATURES).
    
    Returns:
        Enriched array of shape (seq_length, NUM_FEATURES * 2).
    """
    velocity = compute_velocity(sequence)
    return np.concatenate([sequence, velocity], axis=1)


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Normalize a sequence to zero mean and unit variance per feature.
    Handles zero-variance features gracefully.
    
    Args:
        sequence: Array of shape (seq_length, num_features).
    
    Returns:
        Normalized array of same shape.
    """
    mean = np.mean(sequence, axis=0)
    std = np.std(sequence, axis=0)
    
    # Avoid division by zero for constant features
    std[std < 1e-7] = 1.0
    
    return (sequence - mean) / std


def sliding_windows(landmarks_list: list, seq_length: int = None, 
                    step_size: int = None) -> list:
    """
    Generate overlapping sliding windows from a long landmark sequence.
    Useful for creating training samples from continuous recordings.
    
    Args:
        landmarks_list: List of landmark arrays from consecutive frames.
        seq_length: Window length. Defaults to config.SEQUENCE_LENGTH.
        step_size: Step between windows. Defaults to config.STEP_SIZE.
    
    Returns:
        List of numpy arrays, each of shape (seq_length, NUM_FEATURES).
    """
    if seq_length is None:
        seq_length = config.SEQUENCE_LENGTH
    if step_size is None:
        step_size = config.STEP_SIZE
    
    if len(landmarks_list) < seq_length:
        # Return single padded sequence
        return [create_sequence(landmarks_list, seq_length)]
    
    windows = []
    for start in range(0, len(landmarks_list) - seq_length + 1, step_size):
        window = np.array(landmarks_list[start:start + seq_length], dtype=np.float32)
        windows.append(window)
    
    return windows
