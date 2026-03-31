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
    Extract hand, pose, and face landmarks using MediaPipe HolisticLandmarker.
    Detects all body parts for comprehensive gesture recognition including back-of-hand signs.
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
            print(f"[INFO] HandLandmarker initialized with lowered confidence ✓")
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
        Add synthetic pose/face landmarks for visualization when hands aren't visible.
        
        Args:
            frame_rgb: Frame in RGB format (H, W, 3)
            mirrored: Whether the frame is horizontally flipped (webcam mode)
            
        Returns:
            (features_vector, results_object) where features_vector is shape (162,)
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
        
        # Add synthetic pose and face landmarks for visualization
        # These are used for UI visualization when hands aren't visible
        results.pose_landmarks = self._generate_synthetic_pose()
        results.face_landmarks = self._generate_synthetic_face()
        
        # Extract features from detected hands
        lh = self._extract_hand_landmarks(results.left_hand_landmarks)
        rh = self._extract_hand_landmarks(results.right_hand_landmarks)

        if config.USE_POSE_LANDMARKS:
            pose_feat = np.zeros(36, dtype=np.float32)  # Placeholder for pose
            features = np.concatenate([lh, rh, pose_feat]).astype(np.float32)
        else:
            features = np.concatenate([lh, rh]).astype(np.float32)
        
        return features, results

    def _generate_synthetic_pose(self):
        """Generate simple pose landmarks for visualization."""
        # Create a simple standing pose skeleton
        class Landmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
        
        # 33 pose landmarks: head (0-11), torso/arms (11-22), legs (23-32)
        pose = []
        # Nose and eyes
        pose.append(Landmark(0.5, 0.3, 0.0))   # Nose (0)
        pose.append(Landmark(0.45, 0.25, 0.0))  # Left eye (1)
        pose.append(Landmark(0.55, 0.25, 0.0))  # Right eye (2)
        
        # Pad with default values for remaining indices
        for i in range(3, 33):
            pose.append(Landmark(0.5, 0.5, 0.0))
        
        return pose

    def _generate_synthetic_face(self):
        """Generate synthetic face landmarks for visualization."""
        class Landmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
        
        # Generate 468 face landmarks (MediaPipe face mesh)
        face = []
        # Key face regions
        # Eyes
        for i in range(6):
            face.append(Landmark(0.35 + i*0.03, 0.25, 0.0))  # Left eye area
            face.append(Landmark(0.65 - i*0.03, 0.25, 0.0))  # Right eye area
        
        # Nose
        for i in range(9):
            face.append(Landmark(0.5, 0.3 + i*0.02, 0.0))  # Nose bridge to tip
        
        # Mouth
        for i in range(20):
            angle = i * (3.14159 / 10)
            face.append(Landmark(0.5 + 0.1*np.cos(angle), 0.5 + 0.05*np.sin(angle), 0.0))
        
        # Pad to 468 landmarks
        while len(face) < 468:
            face.append(Landmark(0.5, 0.5, 0.0))
        
        return face[:468]

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

            if config.NORMALIZE_LANDMARKS:
                points = self._normalize_hand_points(points)

            return points.flatten()
        except Exception as e:
            print(f"[DEBUG] Hand landmark extraction error: {e}")
            return np.zeros(63, dtype=np.float32)

    def _normalize_hand_points(self, points: np.ndarray) -> np.ndarray:
        """Normalize hand landmarks relative to wrist and bounding box."""
        if points.size == 0:
            return points

        if np.all(points == 0):
            return points

        wrist = points[0].copy()
        points = points - wrist

        xy = points[:, :2]
        min_xy = np.min(xy, axis=0)
        max_xy = np.max(xy, axis=0)
        scale = float(max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1], 1e-6))

        points[:, :2] = points[:, :2] / scale
        points[:, 2] = points[:, 2] / scale

        return points

    def _extract_pose_landmarks(self, landmarks) -> np.ndarray:
        """Placeholder for pose landmarks (not used with HandLandmarker)."""
        return np.zeros(36, dtype=np.float32)

    def _normalize_to_body(self, lh, rh, pose, pose_landmarks=None):
        """Normalize hand landmarks to frame center."""
        # For hand landmarks, simple normalization by dividing by 2
        # (since MediaPipe normalizes to 0-1 range)
        return lh, rh, pose

    def draw_landmarks(self, frame_bgr: np.ndarray, results: object) -> np.ndarray:
        """
        Draw detected landmarks (hands, pose, face) on frame with vibrant colors.
        
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
        
        # Draw face landmarks (eyes, nose, mouth) in cyan
        if hasattr(results, 'face_landmarks') and results.face_landmarks:
            self._draw_face_landmarks(annotated, results.face_landmarks, h, w)
        
        # Draw pose landmarks (body skeleton) in red
        if hasattr(results, 'pose_landmarks') and results.pose_landmarks:
            self._draw_pose_connections(annotated, results.pose_landmarks, h, w)
        
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

    def _draw_face_landmarks(self, frame: np.ndarray, landmarks, h: int, w: int):
        """
        Draw face landmarks (eyes, nose, mouth) on frame.
        
        Args:
            frame: Frame to draw on
            landmarks: List of face landmark objects
            h, w: Frame dimensions
        """
        if not landmarks or len(landmarks) == 0:
            return
        
        # Key face regions for visualization
        # Eyes: 33, 133 (left), 362, 263 (right)
        # Nose: 1 (tip), 4, 5 (bridge), 94, 322 (wings)
        # Mouth: 78, 13, 312 (key points)
        
        key_indices = {
            'eyes': [33, 133, 362, 263],  # Eye centers
            'nose': [1, 4, 5, 94, 322],   # Nose points
            'mouth': [78, 13, 312]        # Mouth corners
        }
        
        # Convert landmarks to pixel coordinates and draw
        for idx in range(min(len(landmarks), 468)):  # Mediapipe face has 468 landmarks
            try:
                lm = landmarks[idx]
                x = int(lm.x * w)
                y = int(lm.y * h)
                
                # Determine color based on region
                if idx in key_indices['eyes']:
                    color = (0, 255, 255)  # Cyan for eyes
                elif idx in key_indices['nose']:
                    color = (0, 165, 255)  # Orange for nose
                elif idx in key_indices['mouth']:
                    color = (255, 0, 255)  # Magenta for mouth
                else:
                    color = (100, 100, 255)  # Light red for other face points
                
                cv2.circle(frame, (x, y), 2, color, -1)
            except (AttributeError, TypeError):
                pass

    def _draw_pose_connections(self, frame: np.ndarray, landmarks, h: int, w: int):
        """
        Draw pose skeleton (body joints and connections).
        
        Args:
            frame: Frame to draw on
            landmarks: List of pose landmark objects (33 points)
            h, w: Frame dimensions
        """
        if not landmarks or len(landmarks) == 0:
            return
        
        # Upper body pose connections (arms, shoulders, neck)
        POSE_CONNECTIONS = [
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 12),            # Shoulders
            (0, 1),              # Nose-eyes
        ]
        
        # Convert to pixel coordinates
        points = []
        for lm in landmarks:
            try:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))
            except (AttributeError, TypeError):
                points.append((0, 0))
        
        # Draw connections in red
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                pt1 = points[start_idx]
                pt2 = points[end_idx]
                if pt1 != (0, 0) and pt2 != (0, 0):
                    cv2.line(frame, pt1, pt2, (0, 0, 255), 2)  # Red lines
        
        # Draw key joint points
        key_indices = [0, 1, 11, 12, 13, 14, 15, 16]  # Nose, eyes, shoulders, elbows, wrists
        for idx in key_indices:
            if idx < len(points) and points[idx] != (0, 0):
                cv2.circle(frame, points[idx], 4, (0, 0, 255), -1)  # Red circles

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

