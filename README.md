# ISL Gesture Recognition System

Isolated Indian Sign Language gloss classification. MediaPipe Hand Landmarker supplies raw hand landmarks. One normalization function feeds a temporal classifier. The production model is the GRU in `config.MODEL_TYPE`, chosen from a five-model comparison on the INCLUDE Seasons test split.

## Features

- **MediaPipe Hand Landmarker** for both hands. Pose is not estimated.
- **One landmark normalization** (`normalize_hands_sequence`) shared by training and live inference
- **GRU production model** (`config.MODEL_TYPE`). LSTM, TCN, MLP, and a one-block Transformer were trained on the same split
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

The training default is the GRU in `config.MODEL_TYPE`. A trained bundle records that choice in `models/isl_baseline_v1/config.json`, and inference loads that file.

```
Input (30 frames x 252 features with velocity, or 126 without)
  -> GRU(128) -> GRU(64)
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

Scores below are held-out test numbers written by `compare_models.py` into `models/comparison/*/metrics.json`. The production copy is `models/isl_baseline_v1/metrics.json`. Validation accuracy during training is not the model score.

Data: INCLUDE Seasons only. 85 videos, 6 classes. No signer ids in the public files, so the split is by `video_id` (seed 42): 61 train / 12 validation / 12 test videos. Test windows: 58. Class counts on that test set: FALL 6, MONSOON 8, SEASON 17, SPRING 11, SUMMER 8, WINTER 8. Features: 30 frames by 252 values. Same augmentation budget for every model (5x on the training windows only, 200 epochs, early stopping patience 30, batch 32). LSTM `recurrent_dropout` is 0 so the latency measurement does not force the slow LSTM kernel.

`python benchmark.py` prints the production model name, P50/P95 milliseconds, FPS, file size, test accuracy, and macro F1.

### Production result

GRU. Test accuracy 0.948. Macro precision 0.937. Macro recall 0.949. Macro F1 0.940. Parameters 188,486. Model size 2,307,916 bytes. Batch-1 CPU latency after warmup: P50 66 ms, P95 75 ms.

Per-class test F1: FALL 0.923, MONSOON 0.933, SEASON 0.970, SPRING 1.000, SUMMER 0.941, WINTER 0.875.

The test set is 58 windows from 12 videos, so these scores move when the initialization changes.

### Model comparison

| Model | Test Accuracy | Macro F1 | P50 CPU Latency | P95 CPU Latency | Parameters | Model Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LSTM | 0.948 | 0.936 | 64 ms | 70 ms | 476,870 | 5,784,354 bytes |
| GRU | 0.948 | 0.940 | 66 ms | 75 ms | 188,486 | 2,307,916 bytes |
| TCN | 0.931 | 0.927 | 66 ms | 81 ms | 287,814 | 3,555,368 bytes |
| MLP | 0.879 | 0.882 | 66 ms | 88 ms | 976,454 | 11,747,674 bytes |
| Transformer | 0.966 | 0.961 | 66 ms | 76 ms | 527,386 | 6,575,721 bytes |

Per-class test F1:

| Class | LSTM | GRU | TCN | MLP | Transformer |
| --- | ---: | ---: | ---: | ---: | ---: |
| FALL | 0.857 | 0.923 | 1.000 | 0.833 | 1.000 |
| MONSOON | 0.933 | 0.933 | 0.933 | 0.933 | 0.933 |
| SEASON | 1.000 | 0.970 | 0.919 | 0.848 | 1.000 |
| SPRING | 0.952 | 1.000 | 1.000 | 0.900 | 0.957 |
| SUMMER | 1.000 | 0.941 | 0.769 | 0.889 | 1.000 |
| WINTER | 0.875 | 0.875 | 0.941 | 0.889 | 0.875 |

The Bi-LSTM does not win this table. Its macro F1 is 0.936 against 0.940 for the GRU, and its P95 latency is 70 ms. With `recurrent_dropout` left at 0, the LSTM is not the slow model.

The Transformer has the highest macro F1 (0.961) and a P95 of 76 ms, which is slower than the GRU. It is one attention block on six classes, so it stays an ablation. The installed production model is the GRU.

### Future work

Signer-independent evaluation needs signer ids, which this Seasons download does not provide. A larger INCLUDE subset and repeated seeds would be required before treating the test score as stable.

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
