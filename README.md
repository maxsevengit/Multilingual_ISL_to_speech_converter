# ISL Gesture Recognition System

**Real-Time Indian Sign Language (ISL) Translation using Computer Vision & Deep Learning**

A complete pipeline that captures live webcam video, extracts hand landmarks using MediaPipe, and classifies ISL word-level gestures using an LSTM neural network — outputting recognized words with confidence scores in real time.

## Features

- **Real-time webcam processing** with hand/pose landmark extraction
- **CLAHE-based image preprocessing** for lighting normalization
- **MediaPipe Holistic** for hand and upper-body pose detection
- **Bi-LSTM model** with batch normalization for gesture classification
- **Continuous recognition engine** with temporal smoothing, confidence gating, and duplicate suppression
- **Pre-built INCLUDE dataset integration** — download & process public ISL dataset automatically
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

## Quick Start

### 1. Install Dependencies

```bash
cd "DIP Project"
pip install -r requirements.txt
```

### 2. Get Training Data

**Option A: Use INCLUDE Public Dataset (Recommended)**

Download and process the INCLUDE ISL dataset automatically:

```bash
# Step 1: Download videos from Zenodo (default: 5 categories, ~2-3 GB)
python download_dataset.py

# See available categories:
python download_dataset.py --list

# Download specific categories:
python download_dataset.py --categories Greetings Essentials Food

# Step 2: Process videos into landmark sequences
python process_videos.py --input data/include_videos

# Process with a word limit (faster for testing):
python process_videos.py --input data/include_videos --max-words 20
```

**Option B: Record Your Own Data**

Record gesture samples via webcam for each word:

```bash
python main.py --mode collect --word HELLO
python main.py --mode collect --word WATER
# Repeat for at least 3 words
```

Controls: `S` = Start recording, `R` = Reset, `Q` = Quit

**Option C: Use Any Video Dataset**

Place videos in `my_videos/WORD_NAME/video.mp4` format and process:

```bash
python process_videos.py --input my_videos/ --format generic
```

### 3. Train the Model

```bash
python train.py --augment

# With velocity features (optional):
python train.py --augment --velocity
```

### 4. Run Real-Time Recognition

```bash
python main.py --mode recognize
```

Controls: `C` = Clear sentence, `R` = Reset, `Q` = Quit

### 5. Add New Words (Extend Vocabulary)

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

This project uses the [INCLUDE dataset](https://zenodo.org/record/4010759) — a large-scale ISL dataset with 4,287 videos over 263 word signs from 15 categories, recorded by deaf students. The default download includes 5 categories covering the most common ISL words:

| Category | Example Words |
|----------|--------------|
| Greetings | Hello, Thank You, Sorry, Welcome |
| Essentials | Yes, No, Please, Help, Water |
| Questions | What, How, Where, When, Why |
| Family | Mother, Father, Brother, Sister |
| Feelings | Happy, Sad, Good, Bad, Angry |

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
