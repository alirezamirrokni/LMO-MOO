#!/usr/bin/env python3
"""Run clipping-strength ablations for MultiMNIST.
Tests three clipping thresholds and the no-clipping variant.
"""
import subprocess, sys

configs = [
    ("clip_0p25", ["--clipping", "task", "--clip-value", "0.25"]),
    ("clip_0p5",  ["--clipping", "task", "--clip-value", "0.5"]),
    ("clip_1p0",  ["--clipping", "task", "--clip-value", "1.0"]),
    ("no_clip",   ["--clipping", "none"]),
]

for name, extra in configs:
    cmd = [
        sys.executable, "run_multimnist.py",
        "--method", "ours",
        "--tag", f"clip_ablation_{name}",
        "--seed", "42",
        "--output-root", "results/multimnist_clip_ablation",
        *extra,
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
