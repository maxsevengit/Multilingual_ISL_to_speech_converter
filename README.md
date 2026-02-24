# ISL Gesture Recognition System

**Real-Time Indian Sign Language (ISL) Translation using Computer Vision & Deep Learning**

A complete pipeline that captures live webcam video, extracts hand landmarks using MediaPipe, and classifies ISL word-level gestures using an LSTM neural network — outputting recognized words with confidence scores in real time.

## Features

- **Real-time webcam processing** with hand/pose landmark extraction
- **CLAHE-based image preprocessing** for lighting normalization
- **MediaPipe Holistic** for hand and upper-body pose detection
- **Bi-LSTM model** with batch normalization for gesture classification
- **Continuous recognition engine** with temporal smoothing, confidence gating, and duplicate suppression
- **Built-in data collection mode** to record your own ISL training data
- **20-word default ISL vocabulary**, easily extensible

## Project Structure

```
DIP Project/
├── config.py                 # All configuration & hyperparameters
├── main.py                   # Real-time recognition & data collection
├── train.py                  # Model training entry point
├── requirements.txt          # Python dependencies
├── README.md
├── data/raw/                 # Collected gesture data (per word)
├── models/                   # Saved trained model
├── vocab/words.json          # Word ↔ index mappings
├── src/
│   ├── preprocessing.py      # Frame normalization & augmentation
│   ├── landmark_extractor.py # MediaPipe landmark detection
│   ├── feature_engineer.py   # Temporal feature engineering
│   ├── dataset.py            # Data collection & loading
│   ├── model.py              # LSTM model definition & training
│   ├── recognizer.py         # Continuous recognition engine
│   └── utils.py              # Helpers (drawing, vocab, FPS)
└── tests/                    # Unit tests
```

## Quick Start

### 1. Install Dependencies

```bash
cd "DIP Project"
pip install -r requirements.txt
```

### 2. Collect Training Data

Record gesture samples for each ISL word you want to recognize:

```bash
# Record 30 samples of "HELLO"
python main.py --mode collect --word HELLO

# Record samples for more words
python main.py --mode collect --word WATER
python main.py --mode collect --word THANK_YOU
python main.py --mode collect --word YES
python main.py --mode collect --word NO
```

**Controls during collection:**
- `S` — Start recording (after 3-second countdown)
- `R` — Reset/discard current recording
- `Q` — Quit and save

### 3. Train the Model

```bash
python train.py

# With data augmentation (recommended):
python train.py --augment

# With velocity features:
python train.py --velocity
```

### 4. Run Real-Time Recognition

```bash
python main.py --mode recognize
```

**Controls during recognition:**
- `C` — Clear recognized sentence
- `R` — Reset recognizer
- `Q` — Quit

### Output Format

The system outputs a clean stream of recognized words:
```
["YOU", "WANT", "WATER"]
```

## Architecture

```
Webcam → Frame Normalization → MediaPipe Holistic → Landmark Extraction
    → Rolling Buffer → Sliding Window → LSTM Model → Confidence Gating
    → Temporal Smoothing → Duplicate Suppression → Word Stream
```

### Model Architecture

```
Input (30 frames × 162 features)
  → LSTM(128, return_sequences=True) + BatchNorm + Dropout(0.3)
  → LSTM(64) + BatchNorm + Dropout(0.3)
  → Dense(64, ReLU) + Dropout(0.3)
  → Dense(num_classes, Softmax)
```

### Feature Extraction (162 features per frame)

| Component | Landmarks | Features |
|-----------|-----------|----------|
| Left hand | 21 × (x,y,z) | 63 |
| Right hand | 21 × (x,y,z) | 63 |
| Upper body pose | 12 × (x,y,z) | 36 |
| **Total** | | **162** |

## Technology Stack

| Component | Library |
|-----------|---------|
| Hand detection | MediaPipe Holistic |
| Video capture | OpenCV |
| Deep learning | TensorFlow/Keras |
| Data handling | NumPy, scikit-learn |
| Visualization | OpenCV, Matplotlib |

## Running Tests

```bash
python -m pytest tests/ -v
```

## Configuration

All hyperparameters are in `config.py`:
- Webcam resolution, MediaPipe confidence thresholds
- Sequence length (30 frames), sliding window step (10 frames)
- LSTM units, dropout, learning rate, epochs
- Confidence threshold (0.6), smoothing window (5 predictions)
