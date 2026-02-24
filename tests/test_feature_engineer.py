"""
Tests for feature engineering module.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.feature_engineer import (
    create_sequence, compute_velocity, compute_acceleration,
    build_feature_vector, normalize_sequence, sliding_windows
)


class TestCreateSequence:
    """Tests for sequence creation from landmark buffer."""
    
    def test_full_buffer(self):
        """Full buffer should return correct shape without padding."""
        buffer = [np.random.rand(config.NUM_FEATURES).astype(np.float32) 
                  for _ in range(config.SEQUENCE_LENGTH)]
        result = create_sequence(buffer)
        assert result.shape == (config.SEQUENCE_LENGTH, config.NUM_FEATURES)
    
    def test_short_buffer_padded(self):
        """Short buffer should be zero-padded at the beginning."""
        buffer = [np.ones(config.NUM_FEATURES, dtype=np.float32) for _ in range(5)]
        result = create_sequence(buffer, seq_length=10)
        assert result.shape == (10, config.NUM_FEATURES)
        # First 5 rows should be zeros (padding)
        np.testing.assert_array_equal(result[:5], 0)
        # Last 5 rows should be ones
        np.testing.assert_array_equal(result[5:], 1)
    
    def test_long_buffer_truncated(self):
        """Long buffer should take the last seq_length frames."""
        seq_len = 10
        buffer = [np.ones(config.NUM_FEATURES, dtype=np.float32) * i 
                  for i in range(20)]
        result = create_sequence(buffer, seq_length=seq_len)
        assert result.shape == (seq_len, config.NUM_FEATURES)
        # Should contain frames 10-19
        assert result[0][0] == 10.0
        assert result[-1][0] == 19.0
    
    def test_empty_buffer(self):
        """Empty buffer should return all zeros."""
        result = create_sequence([], seq_length=10)
        assert result.shape == (10, config.NUM_FEATURES)
        assert np.all(result == 0)


class TestComputeVelocity:
    """Tests for velocity computation."""
    
    def test_output_shape(self):
        """Velocity should have same shape as input."""
        seq = np.random.rand(config.SEQUENCE_LENGTH, config.NUM_FEATURES).astype(np.float32)
        vel = compute_velocity(seq)
        assert vel.shape == seq.shape
    
    def test_first_frame_zero(self):
        """First frame velocity should be zero."""
        seq = np.random.rand(10, 5).astype(np.float32)
        vel = compute_velocity(seq)
        np.testing.assert_array_equal(vel[0], 0)
    
    def test_constant_sequence(self):
        """Constant sequence should have zero velocity."""
        seq = np.ones((10, 5), dtype=np.float32)
        vel = compute_velocity(seq)
        np.testing.assert_array_equal(vel, 0)
    
    def test_linear_sequence(self):
        """Linear sequence should have constant velocity."""
        seq = np.arange(30).reshape(10, 3).astype(np.float32)
        vel = compute_velocity(seq)
        # Each frame increases by 3 in each feature
        expected_vel = np.full((10, 3), 3.0, dtype=np.float32)
        expected_vel[0] = 0  # First frame is zero
        np.testing.assert_array_almost_equal(vel, expected_vel)


class TestComputeAcceleration:
    """Tests for acceleration computation."""
    
    def test_output_shape(self):
        seq = np.random.rand(10, 5).astype(np.float32)
        acc = compute_acceleration(seq)
        assert acc.shape == seq.shape
    
    def test_constant_velocity_zero_accel(self):
        """Constant velocity (linear position) should give zero acceleration."""
        seq = np.arange(30).reshape(10, 3).astype(np.float32)
        acc = compute_acceleration(seq)
        # First two frames may have non-zero due to edge effects
        np.testing.assert_array_almost_equal(acc[2:], 0)


class TestBuildFeatureVector:
    """Tests for enriched feature vector construction."""
    
    def test_doubles_features(self):
        """Feature vector should double the number of features (pos + vel)."""
        seq = np.random.rand(config.SEQUENCE_LENGTH, config.NUM_FEATURES).astype(np.float32)
        enriched = build_feature_vector(seq)
        assert enriched.shape == (config.SEQUENCE_LENGTH, config.NUM_FEATURES * 2)
    
    def test_first_half_is_position(self):
        """First half of features should be the original positions."""
        seq = np.random.rand(10, 5).astype(np.float32)
        enriched = build_feature_vector(seq)
        np.testing.assert_array_equal(enriched[:, :5], seq)


class TestNormalizeSequence:
    """Tests for sequence normalization."""
    
    def test_zero_mean(self):
        """Normalized sequence should have zero mean per feature."""
        seq = np.random.rand(20, 5).astype(np.float32) * 100
        normalized = normalize_sequence(seq)
        means = np.mean(normalized, axis=0)
        np.testing.assert_array_almost_equal(means, 0, decimal=5)
    
    def test_unit_variance(self):
        """Normalized sequence should have unit variance per feature (if non-constant)."""
        seq = np.random.rand(20, 5).astype(np.float32) * 100
        normalized = normalize_sequence(seq)
        stds = np.std(normalized, axis=0)
        np.testing.assert_array_almost_equal(stds, 1, decimal=5)
    
    def test_constant_feature_handled(self):
        """Constant features should not cause division by zero."""
        seq = np.ones((20, 3), dtype=np.float32)
        normalized = normalize_sequence(seq)
        # Should not contain NaN or Inf
        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))


class TestSlidingWindows:
    """Tests for sliding window generation."""
    
    def test_window_count(self):
        """Should produce correct number of windows."""
        landmarks = [np.zeros(config.NUM_FEATURES) for _ in range(50)]
        windows = sliding_windows(landmarks, seq_length=30, step_size=10)
        # (50 - 30) / 10 + 1 = 3
        assert len(windows) == 3
    
    def test_window_shape(self):
        """Each window should have correct shape."""
        landmarks = [np.zeros(config.NUM_FEATURES) for _ in range(50)]
        windows = sliding_windows(landmarks, seq_length=30, step_size=10)
        for w in windows:
            assert w.shape == (30, config.NUM_FEATURES)
    
    def test_short_input_padded(self):
        """Input shorter than window should return single padded window."""
        landmarks = [np.zeros(config.NUM_FEATURES) for _ in range(5)]
        windows = sliding_windows(landmarks, seq_length=30, step_size=10)
        assert len(windows) == 1
        assert windows[0].shape == (30, config.NUM_FEATURES)
