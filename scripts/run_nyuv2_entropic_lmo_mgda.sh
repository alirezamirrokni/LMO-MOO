#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Configuration recorded in results/nyuv2/paper_lmo_moo_sd42.stats.
python run.py --dataset nyuv2 --method entropic-lmo-mgda --seed 42 --epochs 200 --lr 0.0025 --method-params-lr 0.0001 --lmo-alpha 0.1 --muon-ns-steps 5 --entropy-tau 0 --weight-kappa 0 --lmo-lr-schedule legacy --output-root results "$@"
