"""
Hand & Pose Landmark Extraction Module.

Uses MediaPipe Tasks to extract hand landmarks from video frames.

Why Tasks API:
  - Newer MediaPipe Python builds (notably on newer Python versions)
    may not ship the legacy `mediapipe.solutions.*` APIs.
  - Tasks API is stable and works with downloadable `.task` model assets.
"""

from __future__ import annotations

import os
import numpy as np
import mediapipe as mp
import config


class LandmarkExtractor:
    """
    Extracts hand landmarks (2 hands max) and returns a flat feature vector.

    Output feature vector per frame:
      - Left hand:  21 landmarks × 3 coords = 63 values (zero-filled if missing)
      - Right hand: 21 landmarks × 3 coords = 63 values (zero-filled if missing)
      Total: 126 features per frame
    """
    
    def __init__(self):
        self.use_pose = bool(config.USE_POSE_LANDMARKS)
        if self.use_pose:
            raise NotImplementedError(
                "Pose landmarks are disabled by default and not implemented "
                "for the MediaPipe Tasks backend. Set USE_POSE_LANDMARKS=False."
            )

        if not os.path.exists(config.HAND_LANDMARKER_TASK_PATH):
            raise FileNotFoundError(
                f"Missing model asset: {config.HAND_LANDMARKER_TASK_PATH}\n"
                "Download it once (then it works offline):\n"
                "  https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/latest/hand_landmarker.task"
            )

        from mediapipe.tasks import python as mp_tasks_python
        from mediapipe.tasks.python import vision

        base_options = mp_tasks_python.BaseOptions(
            model_asset_path=config.HAND_LANDMARKER_TASK_PATH,
            delegate=mp_tasks_python.BaseOptions.Delegate.CPU,
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
        )
        self._vision = vision
        self._hand_landmarker = vision.HandLandmarker.create_from_options(options)

        # Timestamp for VIDEO-mode inference (ms)
        self._ts_ms = 0
        self._ts_step_ms = 33  # ~30 FPS
    
    def extract_landmarks(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Extract landmarks from an RGB frame.
        
        Args:
            frame_rgb: RGB image frame (H, W, 3).
        
        Returns:
            Flat numpy array of shape (NUM_FEATURES,) containing all landmark
            coordinates. Missing landmarks are zero-filled.
        """
        features, _ = self.extract_landmarks_with_results(frame_rgb)
        return features
    
    def extract_landmarks_with_results(self, frame_rgb: np.ndarray):
        """
        Extract landmarks and also return raw MediaPipe results for drawing.
        
        Args:
            frame_rgb: RGB image frame.
            
        Returns:
            Tuple of (features_array, mediapipe_results).
        """
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Use VIDEO-style API for consistent timestamps (works for webcam + video)
        self._ts_ms += self._ts_step_ms
        results = self._hand_landmarker.detect_for_video(image, self._ts_ms)

        left_hand, right_hand = self._extract_hands_from_tasks_results(results)
        left_hand, right_hand = self._normalize_hands_only(left_hand, right_hand)
        features = np.concatenate([left_hand, right_hand])
        return features, results
    
    def _extract_hand_landmarks(self, hand_landmarks) -> np.ndarray:
        """
        Extract 21 hand landmarks as a flat array.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks or None.
        
        Returns:
            Array of shape (63,). Zero-filled if no hand detected.
        """
        if hand_landmarks is None:
            return np.zeros(config.SINGLE_HAND_FEATURES, dtype=np.float32)
        
        landmarks = []
        for lm in hand_landmarks:
            landmarks.extend([lm.x, lm.y, lm.z])
        
        return np.array(landmarks, dtype=np.float32)

    def _extract_hands_from_tasks_results(self, results) -> tuple[np.ndarray, np.ndarray]:
        """Extract left/right hands from MediaPipe Tasks results."""
        left_hand = np.zeros(config.SINGLE_HAND_FEATURES, dtype=np.float32)
        right_hand = np.zeros(config.SINGLE_HAND_FEATURES, dtype=np.float32)

        hands = getattr(results, "hand_landmarks", None) or []
        handedness = getattr(results, "handedness", None) or []
        if not hands:
            return left_hand, right_hand

        for idx, hand_lms in enumerate(hands):
            label = None
            if idx < len(handedness) and handedness[idx]:
                # handedness[idx] is a list of Category-like objects
                try:
                    label = handedness[idx][0].category_name  # 'Left'/'Right'
                except Exception:
                    label = None

            arr = self._extract_hand_landmarks(hand_lms)
            if label == 'Left':
                left_hand = arr
            elif label == 'Right':
                right_hand = arr
            else:
                # Unknown: fill first empty slot
                if np.all(left_hand == 0):
                    left_hand = arr
                else:
                    right_hand = arr

        return left_hand, right_hand

    def _normalize_hands_only(self, left_hand: np.ndarray, right_hand: np.ndarray):
        """
        Hands-only normalization (no pose required):

        - Translate each detected hand by its wrist (landmark 0)
        - Scale by an approximate hand size (wrist→middle_mcp distance)

        This improves invariance for distance-to-camera changes on laptop webcams.
        """
        def normalize_one(hand: np.ndarray) -> np.ndarray:
            if np.all(hand == 0):
                return hand

            pts = hand.reshape(config.NUM_HAND_LANDMARKS, 3).copy()
            wrist = pts[0]
            pts -= wrist

            # Middle MCP is landmark 9 in MediaPipe Hands
            scale = float(np.linalg.norm(pts[9]))
            if scale < 1e-6:
                scale = 0.1
            pts /= scale
            return pts.flatten().astype(np.float32)

        return normalize_one(left_hand), normalize_one(right_hand)
    
    def draw_landmarks(self, frame_bgr: np.ndarray, results) -> np.ndarray:
        """
        Draw detected landmarks on a BGR frame for visualization.
        
        Args:
            frame_bgr: BGR image frame.
            results: MediaPipe Tasks results.
        
        Returns:
            BGR frame with landmarks drawn.
        """
        annotated = frame_bgr.copy()
        hands = getattr(results, "hand_landmarks", None) or []
        h, w = annotated.shape[:2]
        for hand in hands:
            for lm in hand:
                x = int(np.clip(lm.x * w, 0, w - 1))
                y = int(np.clip(lm.y * h, 0, h - 1))
                annotated[y:y + 2, x:x + 2] = (0, 255, 0)
        return annotated
    
    def has_hands(self, results) -> bool:
        """Check if at least one hand is detected in the results."""
        hands = getattr(results, "hand_landmarks", None) or []
        return bool(hands)
    
    def release(self):
        """Release MediaPipe resources."""
        if self._hand_landmarker is not None:
            self._hand_landmarker.close()
