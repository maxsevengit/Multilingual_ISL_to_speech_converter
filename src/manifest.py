"""
Window manifest for processed landmark sequences.

Each 30-frame window keeps the source video id and frame range. The .npy
file alone is not the dataset index.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

import config
from src.feature_engineer import create_sequence


def window_records_from_frames(
    landmarks: list,
    video_id: str,
    label: str,
    seq_length: int = None,
    step_size: int = None,
    min_frames: int = 10,
    signer_id: str | None = None,
) -> list[tuple[dict, np.ndarray]]:
    """
    Cut one video into windows without dropping the video id.

    Returns (metadata, array) pairs. The array is raw landmarks, not normalized.
    """
    if seq_length is None:
        seq_length = config.SEQUENCE_LENGTH
    if step_size is None:
        step_size = config.STEP_SIZE

    n_frames = len(landmarks)
    if n_frames < min_frames:
        return []

    safe_label = _safe_name(label)
    safe_video = _safe_name(video_id)
    windows: list[tuple[dict, np.ndarray]] = []

    if n_frames <= seq_length:
        ranges = [(0, n_frames - 1, True)]
    else:
        ranges = [
            (start, start + seq_length - 1, False)
            for start in range(0, n_frames - seq_length + 1, step_size)
        ]

    for index, (start, end, padded) in enumerate(ranges):
        if padded:
            array = create_sequence(landmarks, seq_length)
        else:
            array = np.asarray(landmarks[start:end + 1], dtype=np.float32)
        sample_id = f"{safe_label}__{safe_video}__{index:04d}"
        record = {
            "sample_id": sample_id,
            "video_id": video_id,
            "label": label,
            "signer_id": signer_id,
            "start_frame": int(start),
            "end_frame": int(end),
            "padded": bool(padded),
            "split": None,
            "path": os.path.join(
                "data", "processed", safe_label, f"{sample_id}.npy"
            ).replace("\\", "/"),
        }
        windows.append((record, array.astype(np.float32)))
    return windows


def save_window(base_dir: str, record: dict, array: np.ndarray) -> str:
    """Write one raw window and return its absolute path."""
    relative = record["path"]
    absolute = os.path.join(base_dir, relative)
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    np.save(absolute, array.astype(np.float32))
    return absolute


def write_manifest(records: list[dict], path: str = None) -> str:
    if path is None:
        path = config.MANIFEST_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return path


def read_manifest(path: str = None) -> list[dict]:
    if path is None:
        path = config.MANIFEST_PATH
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_split(summary: dict, path: str = None) -> str:
    if path is None:
        path = config.SPLIT_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return path


def load_split_arrays(records: list[dict], base_dir: str = None):
    """
    Load raw windows grouped by split.

    Returns X_train, y_train, X_val, y_val, X_test, y_test, label_names.
    Labels are sorted so the index order is stable.
    """
    if base_dir is None:
        base_dir = config.BASE_DIR

    label_names = sorted({record["label"] for record in records})
    index = {name: i for i, name in enumerate(label_names)}
    buckets = {name: [] for name in ("train", "val", "test")}
    targets = {name: [] for name in ("train", "val", "test")}
    skipped = defaultdict(int)

    for record in records:
        split = record.get("split")
        if split not in buckets:
            skipped["bad_split"] += 1
            continue
        absolute = os.path.join(base_dir, record["path"])
        if not os.path.exists(absolute):
            skipped["missing_file"] += 1
            continue
        array = np.load(absolute)
        if array.shape != (config.SEQUENCE_LENGTH, config.NUM_FEATURES):
            skipped["bad_shape"] += 1
            continue
        if np.all(array == 0) or (np.count_nonzero(array) / array.size) < 0.05:
            skipped["empty"] += 1
            continue
        buckets[split].append(array.astype(np.float32))
        targets[split].append(index[record["label"]])

    def _stack(name):
        if not buckets[name]:
            return (
                np.empty((0, config.SEQUENCE_LENGTH, config.NUM_FEATURES), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )
        return (
            np.stack(buckets[name]).astype(np.float32),
            np.asarray(targets[name], dtype=np.int32),
        )

    X_train, y_train = _stack("train")
    X_val, y_val = _stack("val")
    X_test, y_test = _stack("test")
    return X_train, y_train, X_val, y_val, X_test, y_test, label_names, dict(skipped)


def _safe_name(value: str) -> str:
    cleaned = str(value).strip().replace(" ", "_").replace("\\", "_").replace("/", "__")
    cleaned = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in cleaned)
    return cleaned[:120] or "unknown"
