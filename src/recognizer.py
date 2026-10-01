"""
Real-Time Continuous Gesture Recognition Engine.

Maintains a rolling buffer of landmark frames, runs inference on
overlapping temporal windows, and applies smoothing + confidence
gating to produce a clean word stream.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np

import config
from src.feature_engineer import create_sequence, live_features


@dataclass
class RecognitionResult:
    """One closed-set prediction. probabilities aligns with label_names."""

    gloss: str
    confidence: float
    probabilities: np.ndarray


class GestureRecognizer:
    """
    Continuous gesture recognition engine for real-time ISL translation.
    
    Features:
        - Rolling frame buffer for sliding window inference
        - Temporal smoothing via majority voting
        - Confidence gating (rejects low-confidence predictions)
        - Duplicate suppression (prevents repeated word emission)
        - Accumulated sentence output
    """
    
    def __init__(self, model, label_names: list, 
                 use_velocity_features: bool = False):
        """
        Initialize the recognizer.
        
        Args:
            model: Trained Keras LSTM model.
            label_names: List of word names (index-aligned with model output).
            use_velocity_features: Whether to append velocity features (must match training).
        """
        self.model = model
        self.label_names = label_names
        self.use_velocity_features = use_velocity_features
        
        # Rolling buffer of landmark frames
        self.frame_buffer = collections.deque(maxlen=config.SEQUENCE_LENGTH)
        
        # Smoothing buffer for recent predictions
        self.prediction_buffer = collections.deque(maxlen=config.SMOOTHING_WINDOW)
        
        # Frame counter for controlling inference frequency
        self.frame_count = 0
        
        # Duplicate suppression tracking
        self.last_word = None
        self.last_word_frame = -config.DUPLICATE_COOLDOWN
        
        # Accumulated sentence
        self.sentence = []
        
        # Latest prediction info
        self.current_prediction = None
        self.current_confidence = 0.0
        self._infer_fn = None
        self._compile_inference()

    def _compile_inference(self):
        """Warm one tf.function for batch-1 inference. Mocks keep model.predict."""
        input_shape = getattr(self.model, "input_shape", None)
        if not input_shape or not callable(self.model):
            return
        try:
            import tensorflow as tf
        except ImportError:
            return
        try:
            shape = tuple(1 if dim is None else int(dim) for dim in input_shape)
        except (TypeError, ValueError):
            return
        signature = tf.TensorSpec(shape=shape, dtype=tf.float32)
        model = self.model

        @tf.function(input_signature=[signature])
        def infer(batch):
            return model(batch, training=False)

        infer(tf.zeros(shape, dtype=tf.float32))
        self._infer_fn = infer

    def predict(self, sequence: np.ndarray) -> RecognitionResult:
        """
        Score one model-ready sequence.

        Returns the gloss, its confidence, and the full probability vector.
        The OpenCV loop and any later HTTP route both call this method.
        """
        array = np.asarray(sequence, dtype=np.float32)
        if array.ndim == 2:
            batch = array[np.newaxis, ...]
        elif array.ndim == 3 and array.shape[0] == 1:
            batch = array
        else:
            raise ValueError(
                "predict expects one sequence shaped (frames, features) "
                f"or (1, frames, features). Got {array.shape}."
            )

        input_shape = getattr(self.model, "input_shape", None)
        if input_shape is not None and len(input_shape) == 2:
            batch = batch.reshape((batch.shape[0], -1))
        expected_width = input_shape[-1] if input_shape is not None and len(input_shape) >= 2 else None
        if expected_width is not None and batch.shape[-1] != expected_width:
            raise ValueError(
                f"Checkpoint expects {expected_width} features. "
                f"Current pipeline produces {batch.shape[-1]} features. "
                "Model loading aborted due to incompatible feature specification."
            )

        if self._infer_fn is not None:
            import tensorflow as tf
            output = self._infer_fn(tf.constant(batch))
            probabilities = np.asarray(output)[0]
        else:
            probabilities = np.asarray(self.model.predict(batch, verbose=0))[0]

        probabilities = probabilities.astype(np.float32, copy=False)
        index = int(np.argmax(probabilities))
        return RecognitionResult(
            gloss=self.label_names[index],
            confidence=float(probabilities[index]),
            probabilities=probabilities,
        )
    
    def update(self, landmarks: np.ndarray):
        """
        Process a new frame's landmarks and potentially produce a recognition.
        
        Args:
            landmarks: Flat landmark array of shape (NUM_FEATURES,).
        
        Returns:
            Tuple of (word, confidence) if a new word is recognized, else (None, 0.0).
        """
        self.frame_buffer.append(landmarks)
        self.frame_count += 1
        
        # Only run inference every STEP_SIZE frames and when buffer is full enough
        if (self.frame_count % config.STEP_SIZE != 0 or 
            len(self.frame_buffer) < config.SEQUENCE_LENGTH):
            return None, 0.0
        
        # Same transform as training: normalize, then optional velocity.
        sequence = live_features(
            create_sequence(list(self.frame_buffer)),
            use_velocity=self.use_velocity_features,
        )
        result = self.predict(sequence)
        predicted_word = result.gloss
        confidence = result.confidence
        
        # Update current prediction for display
        self.current_prediction = predicted_word
        self.current_confidence = confidence
        
        # Add to prediction buffer for smoothing
        self.prediction_buffer.append((predicted_word, confidence))
        
        # Apply smoothing: majority vote among recent predictions
        smoothed_word, smoothed_confidence = self._smooth_predictions()
        
        # Confidence gating
        if smoothed_confidence < config.CONFIDENCE_THRESHOLD:
            return None, smoothed_confidence
        
        # Duplicate suppression
        if (smoothed_word == self.last_word and 
            self.frame_count - self.last_word_frame < config.DUPLICATE_COOLDOWN):
            return None, smoothed_confidence
        
        # Accept the word
        self.last_word = smoothed_word
        self.last_word_frame = self.frame_count
        
        # We don't want "IDLE" showing up in the user's sentence
        if smoothed_word != "IDLE":
            self.sentence.append(smoothed_word)
        
        return smoothed_word, smoothed_confidence
    
    def _smooth_predictions(self):
        """
        Apply majority voting over the prediction buffer.
        
        Returns:
            Tuple of (most_voted_word, average_confidence_of_that_word).
        """
        if not self.prediction_buffer:
            return None, 0.0
        
        # Count votes for each word
        word_votes = {}
        word_confidences = {}
        
        for word, conf in self.prediction_buffer:
            if word not in word_votes:
                word_votes[word] = 0
                word_confidences[word] = []
            word_votes[word] += 1
            word_confidences[word].append(conf)
        
        # Find the word with the most votes
        best_word = max(word_votes, key=word_votes.get)
        avg_confidence = np.mean(word_confidences[best_word])
        
        return best_word, float(avg_confidence)
    
    def get_sentence(self) -> list:
        """
        Get the accumulated list of recognized words.
        
        Returns:
            List of recognized word strings. Example: ["YOU", "WANT", "WATER"]
        """
        return self.sentence.copy()
    
    def get_current_prediction(self):
        """
        Get the latest raw prediction (before smoothing/gating).
        
        Returns:
            Tuple of (word, confidence).
        """
        return self.current_prediction, self.current_confidence
    
    def reset(self):
        """Reset all buffers and sentence output."""
        self.frame_buffer.clear()
        self.prediction_buffer.clear()
        self.frame_count = 0
        self.last_word = None
        self.last_word_frame = -config.DUPLICATE_COOLDOWN
        self.sentence = []
        self.current_prediction = None
        self.current_confidence = 0.0
    
    def clear_sentence(self):
        """Clear only the accumulated sentence, keep recognition state."""
        self.sentence = []
