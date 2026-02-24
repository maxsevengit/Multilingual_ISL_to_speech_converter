"""
Tests for landmark extractor module.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.landmark_extractor import LandmarkExtractor


class TestLandmarkExtractor:
    """Tests for MediaPipe landmark extraction."""
    
    @pytest.fixture
    def extractor(self):
        """Create a LandmarkExtractor instance."""
        ext = LandmarkExtractor()
        yield ext
        ext.release()
    
    def test_output_shape(self, extractor):
        """Extracted features should have correct shape."""
        # Create a blank RGB frame
        frame_rgb = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        features = extractor.extract_landmarks(frame_rgb)
        assert features.shape == (config.NUM_FEATURES,)
    
    def test_no_hand_returns_zeros(self, extractor):
        """Blank frame (no hand) should return zero-filled vector."""
        frame_rgb = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        features = extractor.extract_landmarks(frame_rgb)
        # Hands should be zero-filled (pose might also be zeros for blank frame)
        left_hand = features[:config.SINGLE_HAND_FEATURES]
        right_hand = features[config.SINGLE_HAND_FEATURES:config.SINGLE_HAND_FEATURES*2]
        assert np.all(left_hand == 0)
        assert np.all(right_hand == 0)
    
    def test_output_dtype(self, extractor):
        """Features should be float32."""
        frame_rgb = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        features = extractor.extract_landmarks(frame_rgb)
        assert features.dtype == np.float32
    
    def test_extract_with_results(self, extractor):
        """extract_landmarks_with_results should return both features and results."""
        frame_rgb = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        features, results = extractor.extract_landmarks_with_results(frame_rgb)
        assert features.shape == (config.NUM_FEATURES,)
        assert results is not None
    
    def test_has_hands_blank_frame(self, extractor):
        """Blank frame should not have hands detected."""
        frame_rgb = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        _, results = extractor.extract_landmarks_with_results(frame_rgb)
        assert extractor.has_hands(results) == False
    
    def test_draw_landmarks_shape(self, extractor):
        """draw_landmarks should return same-shaped frame."""
        frame_bgr = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        frame_rgb = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        _, results = extractor.extract_landmarks_with_results(frame_rgb)
        annotated = extractor.draw_landmarks(frame_bgr, results)
        assert annotated.shape == frame_bgr.shape


class TestHandLandmarks:
    """Tests for internal hand landmark extraction."""
    
    @pytest.fixture
    def extractor(self):
        ext = LandmarkExtractor()
        yield ext
        ext.release()
    
    def test_none_hand_gives_zeros(self, extractor):
        """None hand landmarks should produce zero array."""
        result = extractor._extract_hand_landmarks(None)
        assert result.shape == (config.SINGLE_HAND_FEATURES,)
        assert np.all(result == 0)
    
    def test_none_pose_gives_zeros(self, extractor):
        """None pose landmarks should produce zero array."""
        result = extractor._extract_pose_landmarks(None)
        assert result.shape == (config.POSE_FEATURES,)
        assert np.all(result == 0)
