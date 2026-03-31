"""
Configuration constants for ISL Gesture Recognition System.
All hyperparameters, paths, and thresholds are centralized here.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
INCLUDE_DIR = os.path.join(BASE_DIR, "data", "include_videos")
MODEL_PATH = os.path.join(BASE_DIR, "models", "isl_gesture_model.keras")
VOCAB_PATH = os.path.join(BASE_DIR, "vocab", "words.json")
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.npz")

# MediaPipe Tasks model assets (download once, then works offline)
HAND_LANDMARKER_TASK_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")

# ─── Webcam ───────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ─── MediaPipe ────────────────────────────────────────────────────────────────
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# ─── Segmentation (Novelty) ───────────────────────────────────────────────────
# If enabled, the recognizer will apply a lightweight hand mask to suppress
# background pixels before extracting landmarks. This tends to improve stability
# under cluttered backgrounds on laptop webcams.
ENABLE_HAND_SEGMENTATION = True

# ─── Landmark / Feature Configuration ─────────────────────────────────────────
# For a laptop demo with a small vocabulary, hands-only features are typically
# more stable than full-body Holistic pose (pose can add noise when partially
# visible or misdetected). You can still switch back to pose+hands if needed.
USE_POSE_LANDMARKS = True

# Normalize landmarks relative to wrist and hand bounding box.
NORMALIZE_LANDMARKS = True

# ─── Landmark Dimensions ──────────────────────────────────────────────────────
# Each hand: 21 landmarks × 3 (x,y,z) = 63
# Both hands: 63 × 2 = 126
# Pose (upper body selected): 12 landmarks × 3 = 36 (shoulders, elbows, wrists, hips)
NUM_HAND_LANDMARKS = 21
HAND_DIMS = 3  # x, y, z
SINGLE_HAND_FEATURES = NUM_HAND_LANDMARKS * HAND_DIMS  # 63
NUM_POSE_LANDMARKS = 12
POSE_FEATURES = NUM_POSE_LANDMARKS * HAND_DIMS  # 36
NUM_FEATURES = (SINGLE_HAND_FEATURES * 2) + (POSE_FEATURES if USE_POSE_LANDMARKS else 0)

# Selected pose landmark indices (upper body only)
POSE_LANDMARK_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
# 11=left_shoulder, 12=right_shoulder, 13=left_elbow, 14=right_elbow,
# 15=left_wrist, 16=right_wrist, 23=left_hip, 24=right_hip,
# 25=left_knee, 26=right_knee, 27=left_ankle, 28=right_ankle

# ─── Temporal / Sequence ─────────────────────────────────────────────────────
# NOTE: Must match the sequence length used in your dataset and trained model.
SEQUENCE_LENGTH = 30       # Shorter = more training samples from same video
STEP_SIZE = 10             # Sliding window step for continuous recognition

# ─── Model Architecture ──────────────────────────────────────────────────────
# MODEL_TYPE: 'lstm' (recommended), 'mlp' (legacy), 'tcn' (1D-CNN)
MODEL_TYPE = 'lstm'

MLP_UNITS_1 = 128          # First dense layer (MLP mode)
MLP_UNITS_2 = 64           # Second dense layer (MLP mode)
LSTM_UNITS_1 = 128         # First LSTM layer
LSTM_UNITS_2 = 64          # Second LSTM layer
TCN_FILTERS = 128          # TCN filters per layer
TCN_KERNEL_SIZE = 3        # TCN kernel size
DENSE_UNITS = 64           # Final dense layer before classifier
DROPOUT_RATE = 0.3         # Dropout rate
LEARNING_RATE = 0.001      # Adam learning rate
LABEL_SMOOTHING = 0.1      # Label smoothing (reduces overconfidence)
EPOCHS = 200               # Max epochs — early stopping handles early exit
BATCH_SIZE = 32            # Larger batch works better with LSTM
BALANCE_TARGET = 200       # Minimum samples per class after balancing

# ─── Recognition Engine ──────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60   # Closed-set classification
SMOOTHING_WINDOW = 7          # Rolling window for stable predictions
DUPLICATE_COOLDOWN = 15       # Frames before allowing same word again

# ─── Data Collection ─────────────────────────────────────────────────────────
SAMPLES_PER_WORD = 30         # Number of sequences to collect per word
COLLECTION_COUNTDOWN = 3      # Seconds countdown before recording starts

# ─── Target Vocabulary (11 Words — classes with real training data) ──────────
MAIN_WORDS = [
	"HELLO",
	"HOW_ARE_YOU",
	"ALRIGHT",
	"GOOD_MORNING",
	"GOOD_AFTERNOON",
	"SUMMER",
	"SPRING",
	"WINTER",
	"FALL",
	"SEASON",
	"MONSOON",
]

# ─── Future Words (need webcam data collection first) ────────────────────────
# These had 0 real samples (all-zeros from INCLUDE dataset). Collect via:
#   python main.py --mode collect --word SUNDAY
# Then add back to MAIN_WORDS and retrain.
FUTURE_WORDS = [
	"SUNDAY", "MONDAY", "TUESDAY", "WEDNESDAY",
	"THURSDAY", "FRIDAY", "SATURDAY", "TODAY",
]
