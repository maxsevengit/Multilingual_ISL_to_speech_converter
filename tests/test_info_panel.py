"""OpenCV overlay changes that do not require a camera."""

import numpy as np

import config
from src.utils import draw_info_panel


def _panel(**kwargs):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    return draw_info_panel(
        frame,
        prediction="SUMMER",
        confidence=kwargs.get("confidence", 0.95),
        sentence=["SUMMER"],
        fps=12,
        mode="WEBCAM",
        model_name=kwargs.get("model_name", "TCN"),
        camera_status=kwargs.get("camera_status", "Camera: open"),
    )


def test_low_confidence_uses_a_different_color():
    high = _panel(confidence=0.95)
    low = _panel(confidence=config.CONFIDENCE_THRESHOLD - 0.1)
    amber = np.array([0, 140, 255])
    teal = np.array([0, 255, 200])
    low_amber = np.any(np.all(low == amber, axis=2))
    high_teal = np.any(np.all(high == teal, axis=2))
    high_amber = np.any(np.all(high == amber, axis=2))
    assert low_amber
    assert high_teal
    assert not high_amber


def test_model_name_and_camera_status_change_the_frame():
    named = _panel(model_name="LSTM", camera_status="Camera: open")
    other = _panel(model_name="GRU", camera_status="Camera: no signal")
    assert named.shape == other.shape
    assert not np.array_equal(named, other)
