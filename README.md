# ISL Gesture Recognition System

**Real-Time Indian Sign Language (ISL) Translation using Computer Vision & Deep Learning**

A complete pipeline that captures live webcam video, extracts hand landmarks using MediaPipe, and classifies ISL word-level gestures using an LSTM neural network — outputting recognized words with confidence scores in real time.

## Features

- **Real-time webcam processing** with hand/pose landmark extraction
- **CLAHE-based image preprocessing** for lighting normalization
- **MediaPipe Holistic** for hand and upper-body pose detection
- **Bi-LSTM model** with batch normalization for gesture classification
- **Continuous recognition engine** with temporal smoothing, confidence gating, and duplicate suppression
- **INCLUDE dataset integration** — process videos in `data/include_videos`
- **Built-in data collection mode** to record your own ISL training data and extend the vocabulary
- **Extensible vocabulary** — easily add new words to the existing model

## Project Structure

```
DIP Project/
├── config.py                 # All configuration & hyperparameters
├── download_dataset.py       # Download INCLUDE dataset from Zenodo
├── process_videos.py         # Convert videos → MediaPipe landmarks
├── train.py                  # Model training entry point
├── main.py                   # Real-time recognition & data collection
├── requirements.txt          # Python dependencies
├── data/
│   ├── include_videos/       # Downloaded INCLUDE video dataset
│   └── raw/                  # Processed landmark data (per word)
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
└── tests/                    # Unit tests (48 tests)
```

## Quick Start (Train on `include_videos` Only)

### 1. Install Dependencies

```bash
cd "DIP Project"
pip install -r requirements.txt
```

### 2. Process INCLUDE Videos

Process the videos in `data/include_videos` into landmark sequences:

```bash
python process_videos.py --input data/include_videos

# Optional limits (faster for testing):
python process_videos.py --input data/include_videos --max-words 20
```

### 3. Train the Model

```bash
python train.py --augment

# Or process INCLUDE videos and train in one command:
python train.py --process-include --augment
```

### 4. Run Real-Time Recognition

```bash
python main.py --mode recognize
```

Controls: `C` = Clear sentence, `R` = Reset, `Q` = Quit

### Optional: Record Your Own Data

Record gesture samples via webcam for each word:

```bash
python main.py --mode collect --word HELLO
python main.py --mode collect --word WATER
# Repeat for at least 3 words
```

Controls: `S` = Start recording, `R` = Reset, `Q` = Quit

### Add New Words (Extend Vocabulary)

After initial training, add new words anytime:

```bash
# Record new word via webcam
python main.py --mode collect --word NEW_WORD

# Re-train with all data (existing + new)
python train.py --augment --reload
```

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

## Dataset: INCLUDE

This project processes the videos already present in `data/include_videos` and trains from the derived landmark sequences.

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
python3 -m pytest tests/ -v   # 48 tests
```

## Configuration

All hyperparameters are in `config.py`:
- Webcam resolution, MediaPipe confidence thresholds
- Sequence length (30 frames), sliding window step (10 frames)
- LSTM units, dropout, learning rate, epochs
- Confidence threshold (0.6), smoothing window (5 predictions)

## Citation

```bibtex
@inproceedings{sridhar2020include,
  author = {Sridhar, Advaith and Ganesan, Rohith Gandhi and Kumar, Pratyush and Khapra, Mitesh},
  title = {INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition},
  year = {2020},
  publisher = {Association for Computing Machinery},
  doi = {10.1145/3394171.3413528},
  series = {MM '20}
}
```
