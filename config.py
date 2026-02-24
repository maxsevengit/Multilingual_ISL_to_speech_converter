"""
Configuration constants for ISL Gesture Recognition System.
All hyperparameters, paths, and thresholds are centralized here.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_PATH = os.path.join(BASE_DIR, "models", "isl_gesture_model.keras")
VOCAB_PATH = os.path.join(BASE_DIR, "vocab", "words.json")
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.npz")

# ─── Webcam ───────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ─── MediaPipe ────────────────────────────────────────────────────────────────
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# ─── Landmark Dimensions ─────────────────────────────────────────────────────
# Each hand: 21 landmarks × 3 (x,y,z) = 63
# Both hands: 63 × 2 = 126
# Pose (upper body selected): 12 landmarks × 3 = 36 (shoulders, elbows, wrists, hips)
NUM_HAND_LANDMARKS = 21
HAND_DIMS = 3  # x, y, z
SINGLE_HAND_FEATURES = NUM_HAND_LANDMARKS * HAND_DIMS  # 63
NUM_POSE_LANDMARKS = 12
POSE_FEATURES = NUM_POSE_LANDMARKS * HAND_DIMS  # 36
NUM_FEATURES = (SINGLE_HAND_FEATURES * 2) + POSE_FEATURES  # 162

# Selected pose landmark indices (upper body only)
POSE_LANDMARK_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
# 11=left_shoulder, 12=right_shoulder, 13=left_elbow, 14=right_elbow,
# 15=left_wrist, 16=right_wrist, 23=left_hip, 24=right_hip,
# 25=left_knee, 26=right_knee, 27=left_ankle, 28=right_ankle

# ─── Temporal / Sequence ─────────────────────────────────────────────────────
SEQUENCE_LENGTH = 30       # Number of frames per gesture window
STEP_SIZE = 10             # Sliding window step for continuous recognition

# ─── Model Architecture ──────────────────────────────────────────────────────
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DENSE_UNITS = 64
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 32

# ─── Recognition Engine ──────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.6    # Minimum confidence to accept a prediction
SMOOTHING_WINDOW = 5          # Number of recent predictions for majority vote
DUPLICATE_COOLDOWN = 15       # Frames before allowing same word again

# ─── Data Collection ─────────────────────────────────────────────────────────
SAMPLES_PER_WORD = 30         # Number of sequences to collect per word
COLLECTION_COUNTDOWN = 3      # Seconds countdown before recording starts
