"""
Video-level and signer-level dataset splits.

Windows are never assigned independently. Every window inherits the split of
its source video, and a signer is never placed in two splits when signer ids
are present on every record.
"""

from __future__ import annotations

import random
from collections import defaultdict

import config


def ranges_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    """Inclusive frame ranges overlap when each starts before the other ends."""
    return start_a <= end_b and start_b <= end_a


def _split_counts(n_videos: int) -> tuple[int, int, int]:
    """Return (n_train, n_val, n_test) video counts for one class."""
    if n_videos <= 0:
        return 0, 0, 0
    if n_videos == 1:
        return 1, 0, 0
    if n_videos == 2:
        return 1, 1, 0

    n_test = max(1, int(round(n_videos * config.TEST_RATIO)))
    n_val = max(1, int(round(n_videos * config.VAL_RATIO)))
    n_train = n_videos - n_val - n_test
    while n_train < 1:
        if n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break
        n_train = n_videos - n_val - n_test
    return n_train, n_val, n_test


def signer_ids_available(records: list[dict]) -> bool:
    """True only when every record has a non-empty signer id."""
    if not records:
        return False
    for record in records:
        signer = record.get("signer_id")
        if signer is None or str(signer).strip() == "":
            return False
    return True


def _assign_signers(records: list[dict], seed: int) -> dict[str, str] | None:
    """
    Put each signer entirely into one split.

    Returns None when fewer than three signers exist, so the caller can fall
    back to a video-level split and record that limitation.
    """
    signer_videos: dict[str, set[str]] = defaultdict(set)
    for record in records:
        signer_videos[str(record["signer_id"])].add(record["video_id"])

    signers = sorted(signer_videos)
    if len(signers) < 3:
        return None

    rng = random.Random(seed)
    rng.shuffle(signers)

    total = sum(len(videos) for videos in signer_videos.values())
    target_test = max(1, int(round(total * config.TEST_RATIO)))
    target_val = max(1, int(round(total * config.VAL_RATIO)))

    split_of = {
        signers[0]: "test",
        signers[1]: "val",
        signers[2]: "train",
    }
    counts = {
        "test": len(signer_videos[signers[0]]),
        "val": len(signer_videos[signers[1]]),
        "train": len(signer_videos[signers[2]]),
    }
    for signer in signers[3:]:
        n_videos = len(signer_videos[signer])
        if counts["test"] < target_test:
            choice = "test"
        elif counts["val"] < target_val:
            choice = "val"
        else:
            choice = "train"
        split_of[signer] = choice
        counts[choice] += n_videos
    return split_of


def assign_splits(records: list[dict], seed: int = None) -> tuple[list[dict], dict]:
    """
    Assign train/val/test on copies of the records.

    Signer-independent when every record has signer_id and there are at least
    three signers. Otherwise group by video id inside each label.
    """
    if seed is None:
        seed = config.SEED

    assigned = [dict(record) for record in records]
    if not assigned:
        return assigned, {
            "strategy": "empty",
            "seed": seed,
            "signer_ids_available": False,
            "videos": {"train": [], "val": [], "test": []},
        }

    if signer_ids_available(assigned):
        signer_split = _assign_signers(assigned, seed)
        if signer_split is not None:
            for record in assigned:
                record["split"] = signer_split[str(record["signer_id"])]
            summary = _summary(assigned, strategy="signer", seed=seed)
            summary["signer_ids_available"] = True
            summary["limitation"] = None
            return assigned, summary

    video_split: dict[str, str] = {}
    by_label: dict[str, list[str]] = defaultdict(list)
    seen = set()
    for record in assigned:
        video_id = record["video_id"]
        if video_id in seen:
            continue
        seen.add(video_id)
        by_label[record["label"]].append(video_id)

    rng = random.Random(seed)
    for label in sorted(by_label):
        videos = sorted(by_label[label])
        rng.shuffle(videos)
        n_train, n_val, n_test = _split_counts(len(videos))
        for video_id in videos[:n_train]:
            video_split[video_id] = "train"
        for video_id in videos[n_train:n_train + n_val]:
            video_split[video_id] = "val"
        for video_id in videos[n_train + n_val:n_train + n_val + n_test]:
            video_split[video_id] = "test"

    for record in assigned:
        record["split"] = video_split[record["video_id"]]

    summary = _summary(assigned, strategy="video", seed=seed)
    summary["signer_ids_available"] = False
    summary["limitation"] = (
        "Per-video signer ids are not available, so the split groups windows "
        "by video_id. A signer-independent split is used automatically when "
        "every manifest row has signer_id and at least three signers exist."
    )
    return assigned, summary


def _summary(records: list[dict], strategy: str, seed: int) -> dict:
    videos = {"train": set(), "val": set(), "test": set()}
    for record in records:
        videos[record["split"]].add(record["video_id"])
    return {
        "strategy": strategy,
        "seed": seed,
        "train_ratio": config.TRAIN_RATIO,
        "val_ratio": config.VAL_RATIO,
        "test_ratio": config.TEST_RATIO,
        "videos": {name: sorted(ids) for name, ids in videos.items()},
        "counts": {
            "videos": {name: len(ids) for name, ids in videos.items()},
            "windows": {
                name: sum(1 for record in records if record["split"] == name)
                for name in ("train", "val", "test")
            },
        },
    }


def assert_no_leakage(records: list[dict]) -> None:
    """
    Raise AssertionError if videos, signers, or overlapping windows cross splits.
    """
    videos = {"train": set(), "val": set(), "test": set()}
    signers = {"train": set(), "val": set(), "test": set()}
    by_video: dict[str, list[dict]] = defaultdict(list)

    for record in records:
        split = record.get("split")
        if split not in videos:
            raise AssertionError(f"Unknown split {split!r} on {record.get('sample_id')}.")
        videos[split].add(record["video_id"])
        signer = record.get("signer_id")
        if signer is not None and str(signer).strip() != "":
            signers[split].add(str(signer))
        by_video[record["video_id"]].append(record)

    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for left, right in pairs:
        shared = videos[left] & videos[right]
        if shared:
            raise AssertionError(
                f"Video leakage between {left} and {right}: {sorted(shared)[:5]}"
            )

    if any(signers.values()):
        shared_signers = signers["train"] & signers["test"]
        if shared_signers:
            raise AssertionError(
                f"Signer leakage between train and test: {sorted(shared_signers)[:5]}"
            )
        shared_val = signers["train"] & signers["val"]
        if shared_val:
            raise AssertionError(
                f"Signer leakage between train and val: {sorted(shared_val)[:5]}"
            )
        shared_val_test = signers["val"] & signers["test"]
        if shared_val_test:
            raise AssertionError(
                f"Signer leakage between val and test: {sorted(shared_val_test)[:5]}"
            )

    for video_id, windows in by_video.items():
        splits = {window["split"] for window in windows}
        if len(splits) > 1:
            raise AssertionError(
                f"Video {video_id} appears in multiple splits: {sorted(splits)}"
            )
        for i, left in enumerate(windows):
            for right in windows[i + 1:]:
                if left["split"] == right["split"]:
                    continue
                if ranges_overlap(
                    int(left["start_frame"]),
                    int(left["end_frame"]),
                    int(right["start_frame"]),
                    int(right["end_frame"]),
                ):
                    raise AssertionError(
                        f"Overlapping windows of {video_id} are in different splits."
                    )
