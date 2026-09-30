# ISL Gesture Recognition System

Isolated Indian Sign Language gloss classification. MediaPipe Hand Landmarker supplies raw hand landmarks. One normalization function feeds a temporal classifier. The shipped baseline is the TCN in `config.MODEL_TYPE`. LSTM, GRU, and a small Transformer were compared on the same INCLUDE Seasons split and are not the default.

## Features

- **MediaPipe Hand Landmarker** for both hands. Pose is not estimated.
- **One landmark normalization** (`normalize_hands_sequence`) shared by training and live inference
- **TCN baseline** (`config.MODEL_TYPE`). Other architectures remain available and are not the shipped default
- **Video-level train/validation/test split** so overlapping windows from one video stay together
- **INCLUDE dataset** as the isolated-sign training source
- **Model bundle** that refuses to load when the feature width, sequence length, vocabulary, or architecture disagree

## Project Structure

```
DIP Project/
├── config.py                 # All configuration & hyperparameters
├── prepare_dataset.py        # Videos -> raw windows + manifest + split
├── download_dataset.py       # Download INCLUDE from Zenodo (explicit command)
├── train.py                  # Train the configured model and write a bundle
├── data/
│   ├── include_videos/       # Downloaded INCLUDE videos (not committed)
│   ├── processed/            # Raw landmark windows (not committed)
│   ├── manifests/            # samples.jsonl and split.json (not committed)
│   └── augmented/            # Not used. Augmentation stays in memory
├── models/isl_baseline_v1/   # model.keras, config.json, vocabulary.json, metrics.json, split.json
├── vocab/words.json          # Demo vocabulary and the live feature spec
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

## Quick Start (Train on `include_videos` Only)

### 1. Install Dependencies

```bash
cd "DIP Project"
pip install -r requirements.txt
```

### 2. Download INCLUDE

The dataset is not downloaded by `pip install`.

```bash
python download_dataset.py --dataset include --list
python download_dataset.py --dataset include --categories Seasons
# Full release is about 57 GB:
python download_dataset.py --dataset include --all
```

### 3. Build the manifest

```bash
python prepare_dataset.py --dataset include
```

### 4. Train

```bash
python train.py --dataset include
```

### 5. Recognize

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
Video
  -> MediaPipe Hand Landmarker
  -> raw landmarks (126)
  -> normalize_hands_sequence()
  -> optional velocity
  -> 30-frame sequence
  -> model bundle
  -> confidence gate, temporal vote, duplicate suppression
```

### Model

The training default is the TCN in `config.MODEL_TYPE`. A trained bundle records that choice in `models/isl_baseline_v1/config.json`, and inference loads that file. `vocab/words.json` uses the same feature width.

```
Input (30 frames x 252 features with velocity, or 126 without)
  -> dilated causal Conv1D stack
  -> global average pool
  -> Dense
  -> softmax
```

With velocity off, the width is 126. Pose features are not included.

### Feature width

| Component | Values |
|-----------|--------|
| Left hand | 21 x (x, y, z) = 63 |
| Right hand | 21 x (x, y, z) = 63 |
| Raw total | 126 |
| With velocity | 252 |

## Dataset: INCLUDE

Primary dataset: [INCLUDE](https://doi.org/10.1145/3394171.3413528) (Sridhar, Ganesan, Kumar, Khapra, ACM Multimedia 2020).

The Zenodo record [4010759](https://zenodo.org/records/4010759) is the download used by `download_dataset.py`. Its record states:

- 4,292 videos (the paper reports 4,287; the record says 5 videos were added later)
- each video is one ISL sign
- official files `train.csv` (3,475) and `test.csv` (817); INCLUDE-50 has 766 train and 192 test videos
- license **CC-BY-4.0**
- signers are deaf students from St. Louis School for the Deaf, Adyar, Chennai

The paper reports 0.27 million frames, 263 word signs, and 15 categories. The public Zenodo files do not include a per-video signer id, so this repository splits by `video_id` (about 70/15/15, seed 42). A signer hold-out is used only when every manifest row has `signer_id` and there are at least three signers.

Videos are RGB and are compatible with MediaPipe Hand Landmarker. The paper's frame count implies roughly 60 frames per video, which fits a 30-frame window. Short videos are padded. Longer videos become overlapping windows that all inherit the source video's split.

iSign ([arXiv:2407.05404](https://arxiv.org/abs/2407.05404), CC-BY-NC-SA-4.0, gated research download) and ISLTranslate (Joshi, Agrawal, Modi, Findings of ACL 2023) are continuous or mixed translation resources. CISLR (Joshi et al., EMNLP 2022) is isolated but averages about 1.5 videos per word across 4,765 words, which does not fit this closed-set 30-frame classifier. They are not the training set.

Disk: category zips are about 0.8–1.8 GB each. `--all` is about 57 GB. Processed landmarks are much smaller than the videos and are gitignored.

Verify a download by checking that `data/include_videos/<Category>/<Word>/*.mp4` exists, then run `python prepare_dataset.py --dataset include`.

## Evaluation

Scores below are held-out test numbers written by `train.py`. Validation accuracy printed during training is not the model score.

### Baseline result

Command: `python train.py --dataset include`

Data: INCLUDE Seasons only (Zenodo `Seasons_1of1.zip`). 85 videos, 6 classes (`FALL`, `MONSOON`, `SEASON`, `SPRING`, `SUMMER`, `WINTER`). Public files do not include signer ids, so the split is by `video_id` (seed 42): 61 train / 12 validation / 12 test videos. Windows from one video stay in one split. Augmentation (5x) runs only on the training windows after the split: 299 train windows become 1,794 training samples. Validation windows: 63. Test windows: 58. Features: 30 frames by 252 values (126 raw hand landmarks plus velocity). Model: TCN. Bundle: `models/isl_baseline_v1/`.

| Metric | Value |
| --- | ---: |
| Test accuracy | 0.983 |
| Macro precision | 0.976 |
| Macro recall | 0.979 |
| Macro F1 | 0.976 |
| Test windows | 58 |
| Classes | 6 |
| Parameters | 287,814 |
| Model size | 3,555,367 bytes |
| CPU latency, batch 1, P50 | 158 ms |
| CPU latency, batch 1, P95 | 182 ms |

Per-class test F1: FALL 0.923, MONSOON 0.933, SEASON 1.000, SPRING 1.000, SUMMER 1.000, WINTER 1.000. The only test error is one MONSOON window predicted as FALL. Early stopping restored epoch 12. That epoch's validation accuracy was 0.810, which is lower than the test accuracy. The test set is 58 windows from 12 videos, so this score is a noisy estimate.

### Model comparison

LSTM, GRU, TCN, and a small Transformer were trained after the baseline, on the same manifest, the same video split, the same features, and the same epoch budget (200 max, early stopping patience 30, batch 32, Adam, seed 42). Each comparison run also called `keras.utils.set_random_seed(42)` before training. The baseline TCN above was trained before that call, so `models/comparison/tcn` is a second TCN initialization and is the TCN row in the table. Latency is one window at a time on CPU after five warmup calls.

| Model | Test Accuracy | Macro F1 | P50 CPU Latency | P95 CPU Latency | Parameters | Model Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LSTM | 0.966 | 0.965 | 173 ms | 267 ms | 476,870 | 5,784,353 bytes |
| GRU | 0.948 | 0.940 | 167 ms | 188 ms | 188,486 | 2,307,916 bytes |
| TCN | 0.931 | 0.927 | 158 ms | 180 ms | 287,814 | 3,555,368 bytes |
| Transformer | 0.966 | 0.961 | 228 ms | 272 ms | 527,386 | 6,575,721 bytes |

Per-class test F1 for those four runs:

| Class | LSTM | GRU | TCN | Transformer |
| --- | ---: | ---: | ---: | ---: |
| FALL | 1.000 | 0.923 | 1.000 | 1.000 |
| MONSOON | 0.933 | 0.933 | 0.933 | 0.933 |
| SEASON | 0.970 | 0.970 | 0.919 | 1.000 |
| SPRING | 1.000 | 1.000 | 1.000 | 0.957 |
| SUMMER | 1.000 | 0.941 | 0.769 | 1.000 |
| WINTER | 0.889 | 0.875 | 0.941 | 0.875 |

The seeded TCN scores 0.927 macro F1. The baseline TCN scores 0.976. That gap is larger than the gap between LSTM and the seeded TCN, so architecture ranking on 58 test windows is not stable across initializations.

### Production candidate

The production bundle stays `models/isl_baseline_v1` (TCN). In the seeded comparison, LSTM has the highest macro F1 (0.965) and the Transformer is close (0.961). LSTM's P95 latency is 267 ms against 180 ms for the seeded TCN, and the Transformer is slower still (P95 272 ms). GRU is smaller and close in latency, with macro F1 0.940. None of those trade-offs replaces the baseline checkpoint. A single 30-frame window takes about 160–270 ms on this CPU, which is the measured latency. It is not a frame-rate figure.

### Future work

Signer-independent evaluation needs signer ids, which this Seasons download does not provide. A larger INCLUDE subset, repeated seeds, and a held-out signer split would be required before treating the test score as stable. Continuous signing, sentence generation, and other languages are not part of this result.

## Running Tests

```bash
python -m pytest tests/ -q
```

## Technology Stack

| Component | Library |
|-----------|---------|
| Hand detection | MediaPipe Hand Landmarker |
| Video capture | OpenCV |
| Deep learning | TensorFlow/Keras |
| Data handling | NumPy, scikit-learn |

## Configuration

All hyperparameters are in `config.py`:
- Webcam resolution and MediaPipe hand-detection thresholds
- Sequence length (30 frames), sliding window step (10 frames)
- `MODEL_TYPE` (`tcn` by default), dropout, learning rate, epochs
- Confidence threshold (0.6), smoothing window (7 predictions)
- Split ratios 0.70 / 0.15 / 0.15 and seed 42

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
