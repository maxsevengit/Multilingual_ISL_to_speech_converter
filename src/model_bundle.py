"""
Reproducible model bundle.

A checkpoint is loaded only when its config and vocabulary agree with the
live feature pipeline. Mismatches raise BundleMismatchError.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import config


class BundleMismatchError(RuntimeError):
    """The checkpoint does not match the current feature pipeline."""


@dataclass
class ModelBundle:
    model: object
    config: dict
    vocabulary: dict
    metrics: dict
    split: dict
    directory: str


def pipeline_spec(model_type: str = None, use_velocity: bool = None) -> dict:
    """Feature spec the current code actually produces."""
    if model_type is None:
        model_type = config.MODEL_TYPE
    if use_velocity is None:
        use_velocity = config.USE_VELOCITY
    feature_width = config.NUM_FEATURES * (2 if use_velocity else 1)
    return {
        "model_type": model_type,
        "sequence_length": config.SEQUENCE_LENGTH,
        "feature_width": feature_width,
        "raw_features": config.NUM_FEATURES,
        "use_velocity": bool(use_velocity),
        "use_pose_landmarks": False,
        "seed": config.SEED,
    }


def validate_bundle_compatibility(bundle_config: dict, vocabulary: dict, pipeline: dict) -> None:
    """
    Fail with an explicit message when the bundle and the live pipeline disagree.
    """
    problems = []

    bundle_width = int(bundle_config["feature_width"])
    pipeline_width = int(pipeline["feature_width"])
    if bundle_width != pipeline_width:
        problems.append(
            f"Checkpoint expects {bundle_width} features. "
            f"Current pipeline produces {pipeline_width} features."
        )

    bundle_seq = int(bundle_config["sequence_length"])
    pipeline_seq = int(pipeline["sequence_length"])
    if bundle_seq != pipeline_seq:
        problems.append(
            f"Checkpoint expects sequence length {bundle_seq}. "
            f"Current pipeline uses {pipeline_seq}."
        )

    if bundle_config.get("model_type") != pipeline.get("model_type"):
        problems.append(
            f"Checkpoint architecture is {bundle_config.get('model_type')!r}. "
            f"Current pipeline is configured as {pipeline.get('model_type')!r}."
        )

    if bool(bundle_config.get("use_velocity")) != bool(pipeline.get("use_velocity")):
        problems.append(
            "Checkpoint use_velocity="
            f"{bundle_config.get('use_velocity')!r}, "
            f"current pipeline use_velocity={pipeline.get('use_velocity')!r}."
        )

    words = list(vocabulary.get("words", []))
    word_to_index = vocabulary.get("word_to_index", {})
    if len(words) != len(word_to_index):
        problems.append(
            f"Vocabulary lists {len(words)} words but word_to_index has {len(word_to_index)} entries."
        )
    for position, word in enumerate(words):
        mapped = word_to_index.get(word)
        if mapped != position:
            problems.append(
                f"Vocabulary order mismatch for {word!r}: "
                f"list index {position}, word_to_index {mapped!r}."
            )
            break

    if problems:
        raise BundleMismatchError(
            "Model loading aborted due to incompatible feature specification.\n"
            + "\n".join(problems)
        )


def _read_json(path: str, default: dict | None = None) -> dict:
    if not os.path.exists(path):
        if default is not None:
            return default
        raise BundleMismatchError(f"Missing bundle file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_model_bundle(
    directory: str,
    model,
    bundle_config: dict,
    vocabulary: dict,
    metrics: dict,
    split_summary: dict,
) -> str:
    """Write model.keras plus the JSON sidecar files."""
    os.makedirs(directory, exist_ok=True)
    model_path = os.path.join(directory, "model.keras")
    model.save(model_path)
    _write_json(os.path.join(directory, "config.json"), bundle_config)
    _write_json(os.path.join(directory, "vocabulary.json"), vocabulary)
    _write_json(os.path.join(directory, "metrics.json"), metrics)
    _write_json(os.path.join(directory, "split.json"), split_summary)
    return directory


def load_model_bundle(directory: str = None, pipeline: dict = None) -> ModelBundle:
    """
    Load a bundle and refuse it when it does not match the live pipeline.
    """
    if directory is None:
        directory = config.BUNDLE_DIR
    if pipeline is None:
        pipeline = pipeline_spec()

    bundle_config = _read_json(os.path.join(directory, "config.json"))
    vocabulary = _read_json(os.path.join(directory, "vocabulary.json"))
    metrics = _read_json(os.path.join(directory, "metrics.json"), default={})
    split_summary = _read_json(os.path.join(directory, "split.json"), default={})
    validate_bundle_compatibility(bundle_config, vocabulary, pipeline)

    model_path = os.path.join(directory, "model.keras")
    if not os.path.exists(model_path):
        raise BundleMismatchError(f"Model file not found: {model_path}")

    from tensorflow import keras

    model = keras.models.load_model(model_path)
    input_shape = getattr(model, "input_shape", None)
    if input_shape is not None and len(input_shape) == 3:
        _, seq_length, feature_width = input_shape
        if seq_length != bundle_config["sequence_length"] or feature_width != bundle_config["feature_width"]:
            raise BundleMismatchError(
                "Model loading aborted due to incompatible feature specification.\n"
                f"model.keras input shape is (batch, {seq_length}, {feature_width}). "
                f"config.json declares sequence_length={bundle_config['sequence_length']}, "
                f"feature_width={bundle_config['feature_width']}."
            )
    output_shape = getattr(model, "output_shape", None)
    if output_shape is not None:
        n_classes = int(output_shape[-1])
        n_words = len(vocabulary.get("words", []))
        if n_classes != n_words:
            raise BundleMismatchError(
                "Model loading aborted due to incompatible feature specification.\n"
                f"model.keras has {n_classes} outputs. "
                f"vocabulary.json has {n_words} words."
            )

    return ModelBundle(
        model=model,
        config=bundle_config,
        vocabulary=vocabulary,
        metrics=metrics,
        split=split_summary,
        directory=directory,
    )


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
