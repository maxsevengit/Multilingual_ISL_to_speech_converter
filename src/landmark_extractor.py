"""
Hand Landmark Extraction Module.

Uses MediaPipe HandLandmarker (Tasks API) to detect hands.
Provides visualization for gesture recognition.
"""

from __future__ import annotations

import os
import numpy as np
import cv2
import config
import mediapipe as mp
from mediapipe.tasks.python.core import base_options as base_options_lib
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions


class LandmarkExtractor:
    """
    Extract raw left/right hand landmarks with MediaPipe HandLandmarker.

    Pose and face are not estimated. The returned vector is 126 floats
    (two hands Ã— 21 landmarks Ã— xyz) and is not normalized. Call
    prepare_model_input before the classifier.
    """
    
    def __init__(self):
        """Initialize HandLandmarker with fallback visualization."""
        print(f"[INFO] LandmarkExtractor initializing (Hand landmark mode)...")
        self.frame_count = 0
        self.landmarker = None
        
        try:
            model_path = "models/hand_landmarker.task"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            base_opts = base_options_lib.BaseOptions(model_asset_path=model_path)
            # Use HandLandmarker for hand detection with lowered confidence for back-of-hand
            from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
            opts = HandLandmarkerOptions(base_options=base_opts, num_hands=2,
                                        min_hand_detection_confidence=0.3,
                                        min_hand_presence_confidence=0.3,
                                        min_tracking_confidence=0.3)
            self.landmarker = HandLandmarker.create_from_options(opts)
            print(f"[INFO] HandLandmarker initialized with lowered confidence âœ“")
        except Exception as e:
            print(f"[ERROR] Failed to initialize landmarker: {e}")
            self.landmarker = None

    def extract_landmarks(self, frame_rgb: np.ndarray, mirrored: bool = False) -> np.ndarray:
        """Extract landmarks, return features only."""
        features, _ = self.extract_landmarks_with_results(frame_rgb, mirrored=mirrored)
        return features

    def extract_landmarks_with_results(self, frame_rgb: np.ndarray, mirrored: bool = False):
        """
        Detect hand landmarks using MediaPipe HandLandmarker.
        Returns raw coordinates. Normalization happens later in one shared function.
        
        Args:
            frame_rgb: Frame in RGB format (H, W, 3)
            mirrored: Whether the frame is horizontally flipped (webcam mode)
            
        Returns:
            (features_vector, results_object) where features_vector is raw
            hand landmarks of shape (126,).
        """
        self.frame_count += 1
        h, w = frame_rgb.shape[:2]
        
        # Create result object
        class DetectionResult:
            pass
        
        results = DetectionResult()
        
        # Try real MediaPipe hand detection
        if self.landmarker is not None:
            try:
                # Convert to MediaPipe Image format
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                detection_result = self.landmarker.detect(mp_image)

                results.hand_landmarks = []
                results.left_hand_landmarks = []
                results.right_hand_landmarks = []

                # Use handedness labels to correctly assign left vs right hand.
                # MediaPipe returns handedness from the *mirror-image* perspective
                # (i.e., what appears as "Left" in the frame is the signer's right).
                # We flip the label so features match anatomical left/right.
                if (hasattr(detection_result, 'hand_landmarks') and
                        detection_result.hand_landmarks):
                    results.hand_landmarks = detection_result.hand_landmarks
                    handedness_list = getattr(detection_result, 'handedness', [])

                    for hand_idx, hand_lms in enumerate(detection_result.hand_landmarks):
                        # Determine chirality from handedness, with fallback
                        chirality = 'Left'  # default fallback
                        if handedness_list and hand_idx < len(handedness_list):
                            cats = handedness_list[hand_idx]
                            if cats:
                                # category.category_name is 'Left' or 'Right'
                                raw_label = getattr(cats[0], 'category_name', 'Left')
                                
                                # Logic:
                                # 1. Training (Unmirrored): Right hand on left side -> MP says 'Left' -> Flip to 'Right'
                                # 2. Webcam (Mirrored): Right hand on right side -> MP says 'Right' -> DO NOT FLIP
                                if mirrored:
                                    chirality = raw_label
                                else:
                                    chirality = 'Right' if raw_label == 'Left' else 'Left'

                        if chirality == 'Left':
                            results.left_hand_landmarks = hand_lms
                        else:
                            results.right_hand_landmarks = hand_lms

            except Exception as e:
                print(f"[DEBUG] Hand detection error: {e}")
                results.hand_landmarks = []
                results.left_hand_landmarks = []
                results.right_hand_landmarks = []
        else:
            results.hand_landmarks = []
            results.left_hand_landmarks = []
            results.right_hand_landmarks = []
        
        if config.USE_POSE_LANDMARKS:
            raise RuntimeError(
                "USE_POSE_LANDMARKS is True, but LandmarkExtractor does not "
                "estimate pose. Refusing to emit a zero-filled pose block."
            )

        lh = self._extract_hand_landmarks(results.left_hand_landmarks)
        rh = self._extract_hand_landmarks(results.right_hand_landmarks)
        features = np.concatenate([lh, rh]).astype(np.float32)
        return features, results

    def _extract_hand_landmarks(self, landmarks) -> np.ndarray:
        """Extract hand landmarks as a flat array of shape (63,)."""
        if landmarks is None or len(landmarks) == 0:
            return np.zeros(63, dtype=np.float32)
        
        try:
            # Handle both NormalizedLandmark objects and dict-like structures
            points = []
            for lm in landmarks:
                if hasattr(lm, 'x') and hasattr(lm, 'y') and hasattr(lm, 'z'):
                    points.append([lm.x, lm.y, lm.z])
                elif isinstance(lm, dict):
                    points.append([lm.get('x', 0), lm.get('y', 0), lm.get('z', 0)])
            
            if len(points) == 0:
                return np.zeros(63, dtype=np.float32)

            points = np.array(points, dtype=np.float32)
            if points.shape[0] < config.NUM_HAND_LANDMARKS:
                padded = np.zeros((config.NUM_HAND_LANDMARKS, 3), dtype=np.float32)
                padded[:points.shape[0]] = points
                points = padded
            elif points.shape[0] > config.NUM_HAND_LANDMARKS:
                points = points[:config.NUM_HAND_LANDMARKS]
            return points.flatten()
        except Exception as e:
            print(f"[DEBUG] Hand landmark extraction error: {e}")
            return np.zeros(63, dtype=np.float32)


    def draw_landmarks(self, frame_bgr: np.ndarray, results: object) -> np.ndarray:
        """
        Draw detected hand landmarks. Pose and face are not drawn.
        
        Args:
            frame_bgr: Frame in BGR format
            results: Results object from detection
            
        Returns:
            Annotated frame with all landmarks drawn
        """
        if results is None:
            return frame_bgr
        
        annotated = frame_bgr.copy()
        h, w = frame_bgr.shape[:2]

        # Draw left hand in GREEN
        if hasattr(results, 'left_hand_landmarks') and results.left_hand_landmarks:
            self._draw_hand_connections(annotated, results.left_hand_landmarks, h, w, color=(0, 255, 0))
        
        # Draw right hand in BLUE
        if hasattr(results, 'right_hand_landmarks') and results.right_hand_landmarks:
            self._draw_hand_connections(annotated, results.right_hand_landmarks, h, w, color=(255, 0, 0))
        
        return annotated

    def _draw_hand_connections(self, frame: np.ndarray, landmarks, h: int, w: int, color=(0, 255, 0)):
        """
        Draw hand skeleton on frame with color-coded fingers.
        Each finger gets a distinct vibrant color.
        
        Args:
            frame: Frame to draw on
            landmarks: List of NormalizedLandmark objects (21 points)
            h, w: Frame dimensions
            color: Base color (used for left/right distinction)
        """
        if not landmarks or len(landmarks) == 0:
            return
        
        # Color palette for each finger (vibrant colors)
        FINGER_COLORS = {
            'thumb': (255, 0, 127),      # Magenta
            'index': (0, 255, 255),      # Cyan
            'middle': (0, 255, 0),       # Green
            'ring': (255, 255, 0),       # Yellow
            'pinky': (255, 0, 0),        # Red
            'palm': (255, 128, 0)        # Orange
        }
        
        # Hand landmarks structure (MediaPipe)
        # 0: wrist, 1-4: thumb, 5-8: index, 9-12: middle, 13-16: ring, 17-20: pinky
        FINGER_RANGES = {
            'thumb': [(0, 1), (1, 2), (2, 3), (3, 4)],
            'index': [(0, 5), (5, 6), (6, 7), (7, 8)],
            'middle': [(0, 9), (9, 10), (10, 11), (11, 12)],
            'ring': [(0, 13), (13, 14), (14, 15), (15, 16)],
            'pinky': [(0, 17), (17, 18), (18, 19), (19, 20)]
        }
        
        # Convert normalized landmarks to pixel coordinates
        points = []
        for lm in landmarks:
            try:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))
            except (AttributeError, TypeError):
                points.append((0, 0))
        
        # Draw each finger with its own color
        for finger_name, connections in FINGER_RANGES.items():
            finger_color = FINGER_COLORS[finger_name]
            
            for start_idx, end_idx in connections:
                if start_idx < len(points) and end_idx < len(points):
                    pt1 = points[start_idx]
                    pt2 = points[end_idx]
                    if pt1 != (0, 0) and pt2 != (0, 0):
                        # Thick lines for better visibility
                        cv2.line(frame, pt1, pt2, finger_color, 4)
        
        # Draw palm connections (wrist to each finger base)
        palm_color = FINGER_COLORS['palm']
        palm_connections = [(0, 5), (0, 9), (0, 13), (0, 17)]  # Wrist to finger bases
        for start_idx, end_idx in palm_connections:
            if start_idx < len(points) and end_idx < len(points):
                pt1 = points[start_idx]
                pt2 = points[end_idx]
                if pt1 != (0, 0) and pt2 != (0, 0):
                    cv2.line(frame, pt1, pt2, palm_color, 3)
        
        # Draw all landmarks as circles with gradient colors
        for i, point in enumerate(points):
            if point != (0, 0):
                # Color based on finger position
                if 1 <= i <= 4:  # Thumb
                    point_color = FINGER_COLORS['thumb']
                elif 5 <= i <= 8:  # Index
                    point_color = FINGER_COLORS['index']
                elif 9 <= i <= 12:  # Middle
                    point_color = FINGER_COLORS['middle']
                elif 13 <= i <= 16:  # Ring
                    point_color = FINGER_COLORS['ring']
                elif 17 <= i <= 20:  # Pinky
                    point_color = FINGER_COLORS['pinky']
                else:  # Wrist
                    point_color = FINGER_COLORS['palm']
                
                # Draw larger, more visible circles
                cv2.circle(frame, point, 6, point_color, -1)
                # Add white outline for contrast
                cv2.circle(frame, point, 6, (255, 255, 255), 1)


    def has_hands(self, results: object) -> bool:
        """Check if hands were detected in the results."""
        if results is None:
            return False
        
        try:
            has_left = (hasattr(results, 'left_hand_landmarks') and 
                       results.left_hand_landmarks is not None and 
                       len(results.left_hand_landmarks) > 0)
            has_right = (hasattr(results, 'right_hand_landmarks') and 
                        results.right_hand_landmarks is not None and 
                        len(results.right_hand_landmarks) > 0)
            return has_left or has_right
        except Exception:
            return False

    def release(self):
        """Release resources."""
        if self.landmarker is not None:
            self.landmarker = None
            print("[INFO] HandLandmarker released")

