"""Train and live feature transforms must be the same function and the same values."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.feature_engineer import live_features, prepare_model_input, training_features


def _raw_sequence(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sequence = rng.normal(loc=0.5, scale=0.1, size=(config.SEQUENCE_LENGTH, config.NUM_FEATURES))
    sequence = sequence.astype(np.float32)
    sequence[:, :3] = 0.4
    return sequence


def test_training_and_live_are_the_same_function():
    assert training_features is live_features
    assert training_features is prepare_model_input


def test_training_and_live_values_match():
    raw = _raw_sequence()
    train = training_features(raw, use_velocity=True)
    live = live_features(raw.copy(), use_velocity=True)
    assert train.shape == live.shape
    assert train.shape == (config.SEQUENCE_LENGTH, config.NUM_FEATURES * 2)
    np.testing.assert_allclose(train, live, rtol=1e-6, atol=1e-6)


def test_missing_hand_stays_finite():
    raw = _raw_sequence()
    raw[:, :config.SINGLE_HAND_FEATURES] = 0
    features = prepare_model_input(raw, use_velocity=True)
    assert features.shape[1] == config.MODEL_FEATURE_WIDTH
    assert np.isfinite(features).all()
    np.testing.assert_allclose(features[:, :config.SINGLE_HAND_FEATURES], 0)


def test_rejects_wrong_feature_width():
    raw = np.zeros((config.SEQUENCE_LENGTH, config.NUM_FEATURES + 36), dtype=np.float32)
    with pytest.raises(ValueError, match="Expected"):
        prepare_model_input(raw, use_velocity=True)


def test_rejects_wrong_rank():
    raw = np.zeros((config.NUM_FEATURES,), dtype=np.float32)
    with pytest.raises(ValueError, match="2D"):
        prepare_model_input(raw, use_velocity=False)


def test_rejects_empty_sequence():
    raw = np.zeros((0, config.NUM_FEATURES), dtype=np.float32)
    with pytest.raises(ValueError, match="no frames"):
        prepare_model_input(raw, use_velocity=False)
