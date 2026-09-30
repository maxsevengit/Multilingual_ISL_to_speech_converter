"""
Read held-out test metrics for the four comparison architectures.

Each model must already have been trained with train.py into
models/comparison/<name>/ using the same INCLUDE Seasons manifest.
This script does not resplit the data and does not train.
"""

import json
import os
import sys

MODELS = ("lstm", "gru", "tcn", "transformer")
ROOT = os.path.join("models", "comparison")


def _load(name: str) -> dict:
    path = os.path.join(ROOT, name, "metrics.json")
    if not os.path.exists(path):
        raise SystemExit(f"Missing held-out metrics: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if metrics.get("split") != "test":
        raise SystemExit(f"{path} is not a held-out test evaluation.")
    return metrics


def main():
    rows = []
    for name in MODELS:
        metrics = _load(name)
        rows.append({
            "model": name,
            "test_accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "per_class_f1": metrics.get("per_class_f1", {}),
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
        })

    out_path = os.path.join(ROOT, "comparison.json")
    os.makedirs(ROOT, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    header = (
        f"{'Model':<13} {'Accuracy':>10} {'Macro F1':>10} "
        f"{'P50 ms':>10} {'P95 ms':>10} {'Params':>10} {'Size':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['model']:<13} {row['test_accuracy']:10.4f} {row['macro_f1']:10.4f} "
            f"{row['p50_cpu_latency_ms']:10.2f} {row['p95_cpu_latency_ms']:10.2f} "
            f"{row['parameters']:10d} {row['model_size_bytes']:12d}"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
