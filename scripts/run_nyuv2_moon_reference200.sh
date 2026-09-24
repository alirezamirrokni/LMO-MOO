#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python run.py --dataset nyuv2 --method moon --epochs 200 --lr 1e-3 --method-params-lr 1e-5 --moon-logit-decay 1e-3 --output-root results "$@"
