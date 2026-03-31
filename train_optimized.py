#!/usr/bin/env python3
"""
Convenience wrapper — delegates to train.py with recommended settings.

Equivalent to:
    python train.py --model-type lstm --balance-target 200 --augment-factor 10

Usage:
    python train_optimized.py
"""

import sys
import subprocess

if __name__ == "__main__":
    # Pass any extra args through
    extra = sys.argv[1:]
    cmd = [
        sys.executable, "train.py",
        "--model-type", "lstm",
        "--balance-target", "200",
        "--augment-factor", "10",
    ] + extra
    print(f"[train_optimized] Running: {' '.join(cmd)}\n")
    sys.exit(subprocess.call(cmd))
