"""Grouped splits must not leak videos, signers, or overlapping windows."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.splits import assert_no_leakage, assign_splits


def _window(video_id, label, start, signer_id=None, sample_suffix=""):
    return {
        "sample_id": f"{video_id}-{start}{sample_suffix}",
        "video_id": video_id,
        "label": label,
        "signer_id": signer_id,
        "start_frame": start,
        "end_frame": start + 29,
        "padded": False,
        "split": None,
        "path": "data/processed/x.npy",
    }


def _records():
    records = []
    for label in ("HELLO", "MONSOON"):
        for video_index in range(6):
            video_id = f"{label}/video_{video_index}.mp4"
            for start in (0, 10, 20):
                records.append(_window(video_id, label, start))
    return records


def test_video_splits_are_disjoint():
    assigned, summary = assign_splits(_records(), seed=42)
    assert summary["strategy"] == "video"
    assert_no_leakage(assigned)
    videos = summary["videos"]
    assert not (set(videos["train"]) & set(videos["val"]))
    assert not (set(videos["train"]) & set(videos["test"]))
    assert not (set(videos["val"]) & set(videos["test"]))


def test_overlapping_windows_stay_together():
    assigned, _ = assign_splits(_records(), seed=42)
    by_video = {}
    for record in assigned:
        by_video.setdefault(record["video_id"], set()).add(record["split"])
    assert all(len(splits) == 1 for splits in by_video.values())


def test_cross_split_overlap_is_rejected():
    bad = [
        _window("same.mp4", "HELLO", 0),
        _window("same.mp4", "HELLO", 10),
    ]
    bad[0]["split"] = "train"
    bad[1]["split"] = "test"
    with pytest.raises(AssertionError):
        assert_no_leakage(bad)


def test_signer_split_holds_signers_out():
    records = []
    for signer_index in range(6):
        for video_index in range(2):
            video_id = f"signer{signer_index}_video{video_index}.mp4"
            records.append(
                _window(video_id, "HELLO", 0, signer_id=f"signer-{signer_index}")
            )
            records.append(
                _window(video_id, "HELLO", 10, signer_id=f"signer-{signer_index}")
            )
    assigned, summary = assign_splits(records, seed=7)
    assert summary["strategy"] == "signer"
    assert_no_leakage(assigned)
    train_signers = {row["signer_id"] for row in assigned if row["split"] == "train"}
    test_signers = {row["signer_id"] for row in assigned if row["split"] == "test"}
    assert train_signers.isdisjoint(test_signers)
    assert test_signers
