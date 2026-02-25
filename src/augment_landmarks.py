"""
Landmark-Level Data Augmentation Module.

Applies augmentations directly to extracted landmark sequences
(not images) to simulate different signers, body sizes, camera
angles, and signing speeds. This is the key to generalization.
"""

import numpy as np
import config


def augment_sequence(sequence: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    """
    Apply a random combination of augmentations to a landmark sequence.
    
    Args:
        sequence: Array of shape (seq_length, NUM_FEATURES).
        intensity: Augmentation strength multiplier (0.0–2.0).
    
    Returns:
        Augmented sequence of same shape.
    """
    aug = sequence.copy().astype(np.float32)
    
    # Each augmentation applied with some probability
    if np.random.random() < 0.7:
        aug = add_noise(aug, sigma=0.02 * intensity)
    
    if np.random.random() < 0.5:
        aug = scale_jitter(aug, range_=(0.8, 1.2), intensity=intensity)
    
    if np.random.random() < 0.5:
        aug = rotate_2d(aug, max_angle=15 * intensity)
    
    if np.random.random() < 0.5:
        aug = time_warp(aug, factor_range=(0.8, 1.2))
    
    if np.random.random() < 0.3:
        aug = landmark_dropout(aug, drop_rate=0.15 * intensity)
    
    if np.random.random() < 0.3:
        aug = hand_swap(aug)
    
    return aug


def add_noise(sequence: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    """
    Add Gaussian noise to landmark coordinates.
    Simulates detection jitter and signer variation.
    
    Only adds noise to non-zero landmarks (preserves missing-hand zeros).
    """
    noise = np.random.normal(0, sigma, sequence.shape).astype(np.float32)
    mask = (sequence != 0).astype(np.float32)
    return sequence + noise * mask


def scale_jitter(sequence: np.ndarray, range_: tuple = (0.8, 1.2),
                 intensity: float = 1.0) -> np.ndarray:
    """
    Randomly scale all landmarks by a uniform factor.
    Simulates different body sizes and camera distances.
    
    Even with scale-invariant normalization, slight scale jitter
    during training improves robustness.
    """
    low = 1.0 - (1.0 - range_[0]) * intensity
    high = 1.0 + (range_[1] - 1.0) * intensity
    scale = np.random.uniform(low, high)
    
    mask = (sequence != 0).astype(np.float32)
    return (sequence * scale) * mask + sequence * (1 - mask)


def rotate_2d(sequence: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Apply a random 2D rotation (around z-axis) to x,y coordinates.
    Simulates slightly tilted camera or body orientation.
    
    Operates on landmarks in groups of 3 (x, y, z), rotating only x and y.
    """
    angle = np.random.uniform(-max_angle, max_angle)
    rad = np.radians(angle)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    
    aug = sequence.copy()
    seq_len, num_features = aug.shape
    
    # Process each landmark's (x, y, z) triplet
    for i in range(0, num_features, 3):
        if i + 1 >= num_features:
            break
        x = aug[:, i].copy()
        y = aug[:, i + 1].copy()
        
        # Only rotate non-zero landmarks
        mask = (x != 0) | (y != 0)
        aug[mask, i] = x[mask] * cos_a - y[mask] * sin_a
        aug[mask, i + 1] = x[mask] * sin_a + y[mask] * cos_a
    
    return aug


def time_warp(sequence: np.ndarray, factor_range: tuple = (0.8, 1.2)) -> np.ndarray:
    """
    Stretch or compress the sequence in time.
    Simulates different signing speeds.
    
    Uses linear interpolation to resample the sequence.
    """
    seq_len, num_features = sequence.shape
    factor = np.random.uniform(*factor_range)
    
    # New temporal indices
    new_len = max(1, int(seq_len * factor))
    old_indices = np.linspace(0, seq_len - 1, new_len)
    
    # Interpolate each feature
    warped = np.zeros((new_len, num_features), dtype=np.float32)
    for f in range(num_features):
        warped[:, f] = np.interp(old_indices, np.arange(seq_len), sequence[:, f])
    
    # Resize back to original sequence length
    if new_len == seq_len:
        return warped
    
    result = np.zeros_like(sequence)
    if new_len > seq_len:
        # Truncate: take center portion
        start = (new_len - seq_len) // 2
        result = warped[start:start + seq_len]
    else:
        # Pad with zeros at the beginning
        result[seq_len - new_len:] = warped
    
    return result


def landmark_dropout(sequence: np.ndarray, drop_rate: float = 0.15) -> np.ndarray:
    """
    Randomly zero out entire landmarks (groups of 3: x,y,z) for some frames.
    Simulates partial hand occlusion.
    """
    aug = sequence.copy()
    seq_len, num_features = aug.shape
    num_landmarks = num_features // 3
    
    for t in range(seq_len):
        for lm in range(num_landmarks):
            if np.random.random() < drop_rate:
                idx = lm * 3
                aug[t, idx:idx + 3] = 0.0
    
    return aug


def hand_swap(sequence: np.ndarray) -> np.ndarray:
    """
    Swap left and right hand landmarks.
    Simulates left-handed vs right-handed signers.
    
    Assumes feature layout: [left_hand(63), right_hand(63), pose(36)]
    """
    aug = sequence.copy()
    lh = config.SINGLE_HAND_FEATURES  # 63
    
    left = aug[:, :lh].copy()
    right = aug[:, lh:2 * lh].copy()
    
    aug[:, :lh] = right
    aug[:, lh:2 * lh] = left
    
    return aug


def generate_augmented_batch(sequence: np.ndarray, num_augments: int = 5,
                              intensity: float = 1.0) -> list:
    """
    Generate multiple augmented versions of a single sequence.
    
    Args:
        sequence: Original landmark sequence (seq_length, NUM_FEATURES).
        num_augments: Number of augmented copies to generate.
        intensity: Augmentation strength.
    
    Returns:
        List of augmented sequences (includes the original).
    """
    batch = [sequence.copy()]
    
    for _ in range(num_augments):
        aug = augment_sequence(sequence, intensity=intensity)
        batch.append(aug)
    
    return batch
