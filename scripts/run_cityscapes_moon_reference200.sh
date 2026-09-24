#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python run.py --dataset cityscapes --method moon --epochs 200 --lr 1e-4 --method-params-lr 1e-4 --moon-logit-decay 0.05 --output-root results "$@"
