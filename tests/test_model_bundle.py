"""Model bundle checks fail closed when the checkpoint does not match the pipeline."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.balance_dataset import balance_dataset
from src.model_bundle import BundleMismatchError, pipeline_spec, validate_bundle_compatibility
from train import augment_sequences


def _vocab(words):
    return {
        "words": words,
        "word_to_index": {word: index for index, word in enumerate(words)},
        "index_to_word": {str(index): word for index, word in enumerate(words)},
    }


def test_feature_width_mismatch_is_rejected():
    bundle = pipeline_spec()
    bundle["feature_width"] = 162
    with pytest.raises(BundleMismatchError, match="162"):
        validate_bundle_compatibility(bundle, _vocab(["HELLO", "MONSOON"]), pipeline_spec())


def test_sequence_length_mismatch_is_rejected():
    bundle = pipeline_spec()
    bundle["sequence_length"] = 16
    with pytest.raises(BundleMismatchError, match="sequence length"):
        validate_bundle_compatibility(bundle, _vocab(["HELLO"]), pipeline_spec())


def test_vocabulary_order_mismatch_is_rejected():
    bundle = pipeline_spec()
    vocab = _vocab(["HELLO", "MONSOON"])
    vocab["word_to_index"] = {"HELLO": 1, "MONSOON": 0}
    with pytest.raises(BundleMismatchError, match="order"):
        validate_bundle_compatibility(bundle, vocab, pipeline_spec())


def test_architecture_mismatch_is_rejected():
    bundle = pipeline_spec(model_type="lstm")
    with pytest.raises(BundleMismatchError, match="architecture"):
        validate_bundle_compatibility(
            bundle, _vocab(["HELLO"]), pipeline_spec(model_type="tcn")
        )


def test_matching_bundle_is_accepted():
    spec = pipeline_spec()
    validate_bundle_compatibility(spec, _vocab(["HELLO", "MONSOON"]), spec)


def test_balance_refuses_to_write_raw(tmp_path):
    with pytest.raises(RuntimeError, match="data/raw"):
        balance_dataset(data_dir=str(tmp_path), target=5)


def test_augmentation_does_not_write_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sequences = np.zeros((2, config.SEQUENCE_LENGTH, config.NUM_FEATURES), dtype=np.float32)
    sequences[:, :, 0] = 1
    labels = np.array([0, 1], dtype=np.int32)
    augmented_x, augmented_y = augment_sequences(sequences, labels, augment_factor=1, intensity=0.1)
    assert augmented_x.shape[0] == 4
    assert augmented_y.shape[0] == 4
    assert list(tmp_path.rglob("*.npy")) == []
