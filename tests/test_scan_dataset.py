"""INCLUDE folder scan must not count the same file twice on case-insensitive disks."""

import os

from process_videos import scan_generic_dataset, scan_include_dataset


def test_include_scan_dedupes_case_variants(tmp_path):
    word_dir = tmp_path / "Seasons" / "61. Summer"
    word_dir.mkdir(parents=True)
    clip = word_dir / "MVI_4565.MOV"
    clip.write_bytes(b"not-a-real-video")

    found = scan_include_dataset(str(tmp_path))

    assert list(found) == ["SUMMER"]
    assert len(found["SUMMER"]) == 1
    assert os.path.basename(found["SUMMER"][0]) == "MVI_4565.MOV"


def test_generic_scan_dedupes_case_variants(tmp_path):
    word_dir = tmp_path / "Hello"
    word_dir.mkdir()
    (word_dir / "clip.mp4").write_bytes(b"x")

    found = scan_generic_dataset(str(tmp_path))

    assert list(found) == ["HELLO"]
    assert len(found["HELLO"]) == 1
