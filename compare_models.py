"""
Train LSTM, GRU, TCN, MLP, and a small Transformer on the existing split.

Does not rebuild the manifest or reshuffle videos. Each run writes
models/comparison/<name>/metrics.json from the held-out test set.
The production bundle is the model with the best test macro F1, using
p95 CPU latency to break ties. The one-block Transformer stays an ablation
unless it is strictly better on both macro F1 and p95 latency.
"""

import json
import os
import shutil
import subprocess
import sys

from src.logutil import get_logger

log = get_logger("compare")

MODELS = ("lstm", "gru", "tcn", "mlp", "transformer")
ABLATION = "transformer"
ROOT = os.path.join("models", "comparison")
PRODUCTION = os.path.join("models", "isl_baseline_v1")
BUNDLE_FILES = (
    "model.keras",
    "config.json",
    "vocabulary.json",
    "metrics.json",
    "split.json",
    "experiment.json",
)


def select_production_model(rows: list[dict]) -> dict:
    """
    Pick a production model from held-out rows.

    rows items need model, macro_f1, and p95_cpu_latency_ms.
    Higher macro F1 wins. Equal F1 keeps the lower p95 latency.
    The Transformer is eligible only when it beats the other winner
    on both macro F1 and p95 latency.
    """
    if not rows:
        raise ValueError("No model metrics to compare.")
    candidates = [row for row in rows if row["model"] != ABLATION]
    if not candidates:
        candidates = list(rows)
    winner = max(candidates, key=lambda row: (row["macro_f1"], -row["p95_cpu_latency_ms"]))
    ablation = next((row for row in rows if row["model"] == ABLATION), None)
    if ablation is None:
        return winner
    better_f1 = ablation["macro_f1"] > winner["macro_f1"]
    better_latency = ablation["p95_cpu_latency_ms"] < winner["p95_cpu_latency_ms"]
    if better_f1 and better_latency:
        return ablation
    return winner


def _train_one(name: str) -> None:
    destination = os.path.join(ROOT, name)
    command = [
        sys.executable,
        "train.py",
        "--dataset",
        "include",
        "--model-type",
        name,
        "--bundle-dir",
        destination,
    ]
    log.info("Training %s into %s", name, destination)
    subprocess.run(command, check=True)


def _load(name: str) -> dict:
    path = os.path.join(ROOT, name, "metrics.json")
    if not os.path.exists(path):
        raise SystemExit(f"Missing held-out metrics: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if metrics.get("split") != "test":
        raise SystemExit(f"{path} is not a held-out test evaluation.")
    return metrics


def _install_production(name: str) -> None:
    source_dir = os.path.join(ROOT, name)
    os.makedirs(PRODUCTION, exist_ok=True)
    for filename in BUNDLE_FILES:
        source = os.path.join(source_dir, filename)
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(PRODUCTION, filename))
    config_path = "config.py"
    with open(config_path, "r", encoding="utf-8") as handle:
        text = handle.read()
    needle = "MODEL_TYPE = '"
    start = text.find(needle)
    if start == -1:
        raise SystemExit("config.py has no MODEL_TYPE assignment.")
    end = text.find("'", start + len(needle))
    updated = text[: start + len(needle)] + name + text[end:]
    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write(updated)


def _rows_from_disk() -> list[dict]:
    rows = []
    for name in MODELS:
        metrics = _load(name)
        rows.append({
            "model": name,
            "test_accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "per_class_precision": metrics.get("per_class_precision", {}),
            "per_class_recall": metrics.get("per_class_recall", {}),
            "per_class_f1": metrics.get("per_class_f1", {}),
            "class_counts": metrics.get("class_counts", {}),
            "confusion_matrix": metrics.get("confusion_matrix", []),
            "p50_cpu_latency_ms": metrics["p50_cpu_latency_ms"],
            "p95_cpu_latency_ms": metrics["p95_cpu_latency_ms"],
            "parameters": metrics["parameters"],
            "model_size_bytes": metrics["model_size_bytes"],
            "test_samples": metrics["test_samples"],
            "num_classes": metrics["num_classes"],
            "seed": metrics.get("seed"),
            "n_train_videos": metrics.get("n_train_videos"),
            "n_val_videos": metrics.get("n_val_videos"),
            "n_test_videos": metrics.get("n_test_videos"),
            "ablation": name == ABLATION,
        })
    return rows


def main():
    os.makedirs(ROOT, exist_ok=True)
    for name in MODELS:
        _train_one(name)
    rows = _rows_from_disk()
    winner = select_production_model(rows)
    _install_production(winner["model"])
    report = {
        "production_model": winner["model"],
        "reason": (
            "Highest held-out macro F1. Equal scores keep the lower p95 CPU latency. "
            "The Transformer is installed only when it wins both macro F1 and p95 latency."
        ),
        "models": rows,
    }
    out_path = os.path.join(ROOT, "comparison.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    header = (
        f"{'Model':<13} {'Accuracy':>10} {'Macro F1':>10} "
        f"{'P50 ms':>10} {'P95 ms':>10} {'Params':>10} {'Size':>12}"
    )
    log.info(header)
    for row in rows:
        log.info(
            "%-13s %10.4f %10.4f %10.2f %10.2f %10d %12d",
            row["model"],
            row["test_accuracy"],
            row["macro_f1"],
            row["p50_cpu_latency_ms"],
            row["p95_cpu_latency_ms"],
            row["parameters"],
            row["model_size_bytes"],
        )
    log.info("Production model: %s", winner["model"])
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
