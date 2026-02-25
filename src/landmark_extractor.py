"""
Hand & Pose Landmark Extraction Module.

Uses MediaPipe Holistic to extract hand and upper-body pose landmarks
from video frames. Returns normalized, flat feature vectors suitable
for temporal sequence modeling.
"""

import numpy as np
import mediapipe as mp
import config


class LandmarkExtractor:
    """
    Extracts hand and pose landmarks from video frames using MediaPipe Holistic.
    
    Produces a flat feature vector per frame:
      - Left hand:  21 landmarks × 3 coords = 63 values
      - Right hand: 21 landmarks × 3 coords = 63 values
      - Pose:       12 landmarks × 3 coords = 36 values (selected upper-body)
      Total: 162 features per frame
    """
    
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
    
    def extract_landmarks(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Extract landmarks from an RGB frame.
        
        Args:
            frame_rgb: RGB image frame (H, W, 3).
        
        Returns:
            Flat numpy array of shape (NUM_FEATURES,) containing all landmark
            coordinates. Missing landmarks are zero-filled.
        """
        results = self.holistic.process(frame_rgb)
        
        # Extract left hand landmarks
        left_hand = self._extract_hand_landmarks(results.left_hand_landmarks)
        
        # Extract right hand landmarks
        right_hand = self._extract_hand_landmarks(results.right_hand_landmarks)
        
        # Extract selected pose landmarks
        pose = self._extract_pose_landmarks(results.pose_landmarks)
        
        # Normalize relative to body center (mid-point of shoulders)
        left_hand, right_hand, pose = self._normalize_to_body(
            left_hand, right_hand, pose, results.pose_landmarks
        )
        
        # Concatenate into single feature vector
        features = np.concatenate([left_hand, right_hand, pose])
        
        return features
    
    def extract_landmarks_with_results(self, frame_rgb: np.ndarray):
        """
        Extract landmarks and also return raw MediaPipe results for drawing.
        
        Args:
            frame_rgb: RGB image frame.
            
        Returns:
            Tuple of (features_array, mediapipe_results).
        """
        results = self.holistic.process(frame_rgb)
        
        left_hand = self._extract_hand_landmarks(results.left_hand_landmarks)
        right_hand = self._extract_hand_landmarks(results.right_hand_landmarks)
        pose = self._extract_pose_landmarks(results.pose_landmarks)
        
        left_hand, right_hand, pose = self._normalize_to_body(
            left_hand, right_hand, pose, results.pose_landmarks
        )
        
        features = np.concatenate([left_hand, right_hand, pose])
        
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
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        
        return np.array(landmarks, dtype=np.float32)
    
    def _extract_pose_landmarks(self, pose_landmarks) -> np.ndarray:
        """
        Extract selected upper-body pose landmarks.
        
        Args:
            pose_landmarks: MediaPipe pose landmarks or None.
        
        Returns:
            Array of shape (POSE_FEATURES,). Zero-filled if no pose detected.
        """
        if pose_landmarks is None:
            return np.zeros(config.POSE_FEATURES, dtype=np.float32)
        
        landmarks = []
        for idx in config.POSE_LANDMARK_INDICES:
            lm = pose_landmarks.landmark[idx]
            landmarks.extend([lm.x, lm.y, lm.z])
        
        return np.array(landmarks, dtype=np.float32)
    
    def _normalize_to_body(self, left_hand, right_hand, pose, pose_landmarks):
        """
        Normalize all landmarks for translation AND scale invariance.
        
        1. Translate: subtract shoulder midpoint (translation invariance)
        2. Scale: divide by shoulder width (body-size invariance)
        
        This ensures the same sign produces identical features regardless
        of the signer's body size or distance from camera.
        """
        if pose_landmarks is None:
            return left_hand, right_hand, pose
        
        # Get shoulder landmarks
        left_shoulder = pose_landmarks.landmark[11]
        right_shoulder = pose_landmarks.landmark[12]
        
        # Translation reference: shoulder midpoint
        ref_x = (left_shoulder.x + right_shoulder.x) / 2
        ref_y = (left_shoulder.y + right_shoulder.y) / 2
        ref_z = (left_shoulder.z + right_shoulder.z) / 2
        ref_point = np.array([ref_x, ref_y, ref_z], dtype=np.float32)
        
        # Scale reference: shoulder width (Euclidean distance)
        shoulder_dist = np.sqrt(
            (left_shoulder.x - right_shoulder.x) ** 2 +
            (left_shoulder.y - right_shoulder.y) ** 2 +
            (left_shoulder.z - right_shoulder.z) ** 2
        )
        # Avoid division by zero; use fallback if shoulders not detected properly
        if shoulder_dist < 1e-6:
            shoulder_dist = 0.3  # reasonable default in normalized coords
        
        def normalize_array(arr, num_landmarks):
            if np.all(arr == 0):
                return arr
            reshaped = arr.reshape(num_landmarks, 3)
            reshaped -= ref_point       # translation invariance
            reshaped /= shoulder_dist   # scale invariance
            return reshaped.flatten().astype(np.float32)
        
        left_hand = normalize_array(left_hand, config.NUM_HAND_LANDMARKS)
        right_hand = normalize_array(right_hand, config.NUM_HAND_LANDMARKS)
        pose = normalize_array(pose, config.NUM_POSE_LANDMARKS)
        
        return left_hand, right_hand, pose
    
    def draw_landmarks(self, frame_bgr: np.ndarray, results) -> np.ndarray:
        """
        Draw detected landmarks on a BGR frame for visualization.
        
        Args:
            frame_bgr: BGR image frame.
            results: MediaPipe Holistic results.
        
        Returns:
            BGR frame with landmarks drawn.
        """
        annotated = frame_bgr.copy()
        
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw left hand
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Draw right hand
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        return annotated
    
    def has_hands(self, results) -> bool:
        """Check if at least one hand is detected in the results."""
        return (results.left_hand_landmarks is not None or 
                results.right_hand_landmarks is not None)
    
    def release(self):
        """Release MediaPipe resources."""
        self.holistic.close()
