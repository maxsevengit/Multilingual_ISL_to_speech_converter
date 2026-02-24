"""
Tests for image preprocessing module.
"""

import numpy as np
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import normalize_frame, convert_color, augment_frame, add_gaussian_noise
import config


class TestNormalizeFrame:
    """Tests for frame normalization."""
    
    def test_output_shape(self):
        """Normalized frame should have target dimensions."""
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        result = normalize_frame(frame)
        assert result.shape == (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3)
    
    def test_output_dtype(self):
        """Normalized frame should be uint8."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = normalize_frame(frame)
        assert result.dtype == np.uint8
    
    def test_pixel_range(self):
        """Pixel values should be in valid range [0, 255]."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = normalize_frame(frame)
        assert result.min() >= 0
        assert result.max() <= 255
    
    def test_dark_image_enhanced(self):
        """CLAHE should enhance a very dark image."""
        dark_frame = np.ones((480, 640, 3), dtype=np.uint8) * 20
        result = normalize_frame(dark_frame)
        # After CLAHE, the mean should be higher than the dark input
        assert result.mean() >= dark_frame.mean()


class TestConvertColor:
    """Tests for color conversion."""
    
    def test_bgr_to_rgb(self):
        """BGR → RGB should swap channels."""
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # Blue channel
        rgb = convert_color(bgr, 'RGB')
        assert rgb[:, :, 2].mean() == 255  # Blue should be in R position... no, B→R swap
        assert rgb[:, :, 0].mean() == 0 or rgb[:, :, 2].mean() == 255
    
    def test_roundtrip(self):
        """BGR → RGB → BGR should return the original."""
        original = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        rgb = convert_color(original, 'RGB')
        back = convert_color(rgb, 'BGR')
        np.testing.assert_array_equal(original, back)
    
    def test_invalid_target(self):
        """Invalid target should raise ValueError."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            convert_color(frame, 'HSV')


class TestAugmentFrame:
    """Tests for data augmentation."""
    
    def test_output_shape(self):
        """Augmented frame should have same shape as input."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = augment_frame(frame)
        assert result.shape == frame.shape
    
    def test_output_dtype(self):
        """Augmented frame should be uint8."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = augment_frame(frame)
        assert result.dtype == np.uint8
    
    def test_pixel_range(self):
        """Augmented pixels should stay in [0, 255]."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = augment_frame(frame)
        assert result.min() >= 0
        assert result.max() <= 255


class TestGaussianNoise:
    """Tests for Gaussian noise augmentation."""
    
    def test_output_shape(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = add_gaussian_noise(frame)
        assert result.shape == frame.shape
    
    def test_noise_added(self):
        """Noisy image should differ from original (with high probability)."""
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = add_gaussian_noise(frame, sigma=50)
        assert not np.array_equal(frame, result)
