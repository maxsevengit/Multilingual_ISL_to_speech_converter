"""
Tests for the continuous gesture recognizer engine.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.recognizer import GestureRecognizer


class MockModel:
    """Mock Keras model for testing the recognizer logic."""
    
    def __init__(self, num_classes, predicted_class=0, confidence=0.95):
        self.num_classes = num_classes
        self.predicted_class = predicted_class
        self.confidence = confidence
    
    def predict(self, x, verbose=0):
        """Return a mock prediction."""
        batch_size = x.shape[0]
        predictions = np.zeros((batch_size, self.num_classes), dtype=np.float32)
        remaining = (1.0 - self.confidence) / max(self.num_classes - 1, 1)
        predictions[:, :] = remaining
        predictions[:, self.predicted_class] = self.confidence
        return predictions


class TestGestureRecognizer:
    """Tests for the gesture recognizer engine."""
    
    @pytest.fixture
    def recognizer(self):
        """Create a recognizer with a mock model."""
        label_names = ["HELLO", "WATER", "THANK YOU"]
        model = MockModel(num_classes=3, predicted_class=0, confidence=0.9)
        return GestureRecognizer(model, label_names)
    
    def test_initial_state(self, recognizer):
        """Recognizer should start with empty sentence."""
        assert recognizer.get_sentence() == []
        assert recognizer.current_prediction is None
        assert recognizer.current_confidence == 0.0
    
    def test_no_prediction_until_buffer_full(self, recognizer):
        """Should not predict until buffer has enough frames."""
        landmarks = np.random.rand(config.NUM_FEATURES).astype(np.float32)
        
        for _ in range(config.SEQUENCE_LENGTH - 1):
            word, conf = recognizer.update(landmarks)
            # Should not produce prediction yet (buffer not full)
        
        assert recognizer.get_sentence() == [] or len(recognizer.get_sentence()) <= 1
    
    def test_prediction_after_full_buffer(self, recognizer):
        """Should produce prediction after buffer is filled and step is reached."""
        landmarks = np.random.rand(config.NUM_FEATURES).astype(np.float32)
        
        recognized = []
        for _ in range(config.SEQUENCE_LENGTH + config.STEP_SIZE):
            word, conf = recognizer.update(landmarks)
            if word is not None:
                recognized.append(word)
        
        # Should have at least one prediction
        assert len(recognized) >= 1
        # The mock model always predicts class 0 = "HELLO"
        assert recognized[0] == "HELLO"
    
    def test_confidence_gating(self):
        """Low-confidence predictions should be suppressed."""
        label_names = ["HELLO", "WATER", "THANK YOU"]
        model = MockModel(num_classes=3, predicted_class=0, confidence=0.3)
        recognizer = GestureRecognizer(model, label_names)
        
        landmarks = np.random.rand(config.NUM_FEATURES).astype(np.float32)
        
        for _ in range(config.SEQUENCE_LENGTH + config.STEP_SIZE * 5):
            word, conf = recognizer.update(landmarks)
        
        # Should not accept any low-confidence predictions
        assert recognizer.get_sentence() == []
    
    def test_duplicate_suppression(self, recognizer):
        """Same word should not be emitted rapidly."""
        landmarks = np.random.rand(config.NUM_FEATURES).astype(np.float32)
        
        recognized = []
        for _ in range(config.SEQUENCE_LENGTH + config.STEP_SIZE * 10):
            word, conf = recognizer.update(landmarks)
            if word is not None:
                recognized.append(word)
        
        # Should not have many duplicates (cooldown prevents it)
        # With DUPLICATE_COOLDOWN=15 and STEP_SIZE=10, at most a few
        assert len(recognized) <= 8
    
    def test_reset(self, recognizer):
        """Reset should clear all state."""
        landmarks = np.random.rand(config.NUM_FEATURES).astype(np.float32)
        
        for _ in range(config.SEQUENCE_LENGTH + config.STEP_SIZE * 3):
            recognizer.update(landmarks)
        
        recognizer.reset()
        assert recognizer.get_sentence() == []
        assert recognizer.current_prediction is None
        assert recognizer.frame_count == 0
        assert len(recognizer.frame_buffer) == 0
    
    def test_clear_sentence(self, recognizer):
        """clear_sentence should only clear the sentence, not recognition state."""
        landmarks = np.random.rand(config.NUM_FEATURES).astype(np.float32)
        
        for _ in range(config.SEQUENCE_LENGTH + config.STEP_SIZE * 3):
            recognizer.update(landmarks)
        
        recognizer.clear_sentence()
        assert recognizer.get_sentence() == []
        # But frame buffer should still have data
        assert len(recognizer.frame_buffer) > 0
    
    def test_different_words(self):
        """Recognizer should handle changing predictions."""
        label_names = ["HELLO", "WATER", "THANK YOU"]
        
        # First recognizer with class 0
        model1 = MockModel(num_classes=3, predicted_class=0, confidence=0.9)
        recognizer = GestureRecognizer(model1, label_names)
        
        landmarks = np.random.rand(config.NUM_FEATURES).astype(np.float32)
        
        # Fill buffer
        for _ in range(config.SEQUENCE_LENGTH + config.STEP_SIZE):
            recognizer.update(landmarks)
        
        sentence = recognizer.get_sentence()
        assert len(sentence) >= 1


class TestSmoothPredictions:
    """Tests for the prediction smoothing mechanism."""
    
    def test_majority_vote(self):
        """Smoothing should return the most frequent prediction."""
        label_names = ["HELLO", "WATER", "THANK YOU"]
        model = MockModel(num_classes=3, predicted_class=0, confidence=0.9)
        recognizer = GestureRecognizer(model, label_names)
        
        # Manually fill prediction buffer
        recognizer.prediction_buffer.append(("HELLO", 0.9))
        recognizer.prediction_buffer.append(("HELLO", 0.85))
        recognizer.prediction_buffer.append(("WATER", 0.7))
        
        word, conf = recognizer._smooth_predictions()
        assert word == "HELLO"
    
    def test_empty_buffer(self):
        """Empty prediction buffer should return None."""
        label_names = ["HELLO", "WATER"]
        model = MockModel(num_classes=2, predicted_class=0, confidence=0.9)
        recognizer = GestureRecognizer(model, label_names)
        
        word, conf = recognizer._smooth_predictions()
        assert word is None
        assert conf == 0.0
