"""Manifest rows keep the source video and frame range."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.manifest import read_manifest, window_records_from_frames, write_manifest


def _frames(n: int) -> list:
    return [
        np.full(config.NUM_FEATURES, fill_value=i + 1, dtype=np.float32)
        for i in range(n)
    ]


def test_overlapping_windows_share_video_id():
    windows = window_records_from_frames(
        _frames(50), video_id="Greetings/HELLO/clip.mp4", label="HELLO"
    )
    assert len(windows) == 3
    video_ids = {record["video_id"] for record, _ in windows}
    assert video_ids == {"Greetings/HELLO/clip.mp4"}
    starts = [record["start_frame"] for record, _ in windows]
    assert starts == [0, 10, 20]
    assert windows[0][0]["end_frame"] == 29
    assert windows[0][1].shape == (config.SEQUENCE_LENGTH, config.NUM_FEATURES)


def test_short_video_is_padded_and_keeps_source_range():
    windows = window_records_from_frames(
        _frames(12), video_id="clip.mp4", label="HELLO", min_frames=10
    )
    assert len(windows) == 1
    record, array = windows[0]
    assert record["padded"] is True
    assert record["start_frame"] == 0
    assert record["end_frame"] == 11
    assert array.shape == (config.SEQUENCE_LENGTH, config.NUM_FEATURES)
    assert record["signer_id"] is None


def test_manifest_round_trip(tmp_path):
    windows = window_records_from_frames(_frames(30), video_id="v1", label="HELLO")
    path = tmp_path / "samples.jsonl"
    write_manifest([record for record, _ in windows], path=str(path))
    loaded = read_manifest(str(path))
    assert loaded[0]["video_id"] == "v1"
    assert loaded[0]["label"] == "HELLO"
    assert "sample_id" in loaded[0]
    assert loaded[0]["path"].startswith("data/processed/")
