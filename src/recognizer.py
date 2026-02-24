"""
Real-Time Continuous Gesture Recognition Engine.

Maintains a rolling buffer of landmark frames, runs inference on
overlapping temporal windows, and applies smoothing + confidence
gating to produce a clean word stream.
"""

import collections
import numpy as np
from tensorflow import keras
import config
from src.feature_engineer import create_sequence, build_feature_vector


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
    
    def __init__(self, model: keras.Model, label_names: list, 
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
        
        # Build sequence from buffer
        sequence = create_sequence(list(self.frame_buffer))
        
        # Optionally add velocity features
        if self.use_velocity_features:
            sequence = build_feature_vector(sequence)
        
        # Run model inference
        input_data = np.expand_dims(sequence, axis=0)  # Add batch dimension
        predictions = self.model.predict(input_data, verbose=0)[0]
        
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])
        predicted_word = self.label_names[predicted_idx]
        
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
