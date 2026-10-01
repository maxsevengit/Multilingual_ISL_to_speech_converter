"""
Print batch-1 CPU latency and the held-out scores stored with a bundle.

    python benchmark.py
    python benchmark.py --bundle models/comparison/lstm
"""

import argparse
import json
import os
import time

import numpy as np

from src.logutil import get_logger

log = get_logger("benchmark")


def _read_metrics(bundle: str) -> dict:
    path = os.path.join(bundle, "metrics.json")
    with open(path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if metrics.get("split") != "test":
        raise SystemExit(f"{path} is not a held-out test evaluation.")
    return metrics


def _measure(bundle: str, metrics: dict) -> tuple[float, float]:
    """Warm the bundle, then time batch-1 inference. Falls back to stored latency."""
    model_path = os.path.join(bundle, "model.keras")
    if not os.path.exists(model_path):
        return metrics["p50_cpu_latency_ms"], metrics["p95_cpu_latency_ms"]
    from tensorflow import keras

    model = keras.models.load_model(model_path)
    sequence = metrics.get("sequence_length", model.input_shape[1])
    width = metrics.get("feature_width", model.input_shape[2])
    sample = np.zeros((1, sequence, width), dtype=np.float32)
    for _ in range(5):
        model.predict(sample, verbose=0)
    samples = []
    for _ in range(30):
        start = time.perf_counter()
        model.predict(sample, verbose=0)
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    p50 = samples[len(samples) // 2]
    p95 = samples[min(len(samples) - 1, int(round(0.95 * (len(samples) - 1))))]
    return float(p50), float(p95)


def main():
    parser = argparse.ArgumentParser(description="Report model latency and test scores")
    parser.add_argument("--bundle", default=os.path.join("models", "isl_baseline_v1"))
    parser.add_argument("--stored-latency", action="store_true",
                        help="Print latency from metrics.json instead of measuring again")
    args = parser.parse_args()

    metrics = _read_metrics(args.bundle)
    if args.stored_latency:
        p50 = float(metrics["p50_cpu_latency_ms"])
        p95 = float(metrics["p95_cpu_latency_ms"])
    else:
        p50, p95 = _measure(args.bundle, metrics)
    size = int(metrics.get("model_size_bytes") or os.path.getsize(os.path.join(args.bundle, "model.keras")))
    fps = 1000.0 / p50 if p50 else 0.0
    name = metrics.get("model_type", "unknown")
    log.info("model %s", name)
    log.info("p50_ms %.2f", p50)
    log.info("p95_ms %.2f", p95)
    log.info("fps %.2f", fps)
    log.info("size_bytes %d", size)
    log.info("test_accuracy %.4f", metrics["accuracy"])
    log.info("macro_f1 %.4f", metrics["macro_f1"])


if __name__ == "__main__":
    main()
