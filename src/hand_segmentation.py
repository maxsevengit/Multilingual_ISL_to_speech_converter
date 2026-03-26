"""
Lightweight hand segmentation from MediaPipe landmarks.

This is not a heavy neural segmentation model; instead it uses the detected
hand landmark polygons to build a binary mask, then uses that mask to suppress
background pixels. For a laptop demo, this is fast and improves robustness
against cluttered backgrounds.
"""

from __future__ import annotations

import cv2
import numpy as np


def hand_mask_from_mediapipe_hands(results, frame_bgr: np.ndarray, dilate_px: int = 12) -> np.ndarray:
    """
    Create a hand mask from MediaPipe Hands results.

    Args:
        results: MediaPipe Hands results object.
        frame_bgr: Original frame (for shape reference).
        dilate_px: Dilation radius in pixels for a more generous mask.

    Returns:
        mask: uint8 mask of shape (H, W) with values {0, 255}.
    """
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if not getattr(results, "multi_hand_landmarks", None):
        return mask

    for hand_lms in results.multi_hand_landmarks:
        pts = []
        for lm in hand_lms.landmark:
            x = int(np.clip(lm.x * w, 0, w - 1))
            y = int(np.clip(lm.y * h, 0, h - 1))
            pts.append([x, y])
        pts = np.array(pts, dtype=np.int32)
        if pts.shape[0] < 3:
            continue
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)

    if dilate_px > 0 and np.any(mask):
        k = max(3, int(dilate_px) | 1)  # odd kernel size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def apply_mask(frame_bgr: np.ndarray, mask: np.ndarray, background: int = 0) -> np.ndarray:
    """
    Apply a binary mask to a frame.

    Args:
        frame_bgr: Input frame.
        mask: uint8 mask (H,W) values {0,255}.
        background: Background fill value (0 = black).

    Returns:
        Masked BGR frame.
    """
    if mask is None or mask.size == 0:
        return frame_bgr
    if mask.ndim == 2:
        m = mask[:, :, None]
    else:
        m = mask
    out = frame_bgr.copy()
    out[m == 0] = background
    return out

