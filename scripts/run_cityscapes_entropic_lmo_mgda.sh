#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Configuration recorded in results/cityscapes/paper_lmo_moo_sd42.stats.
python run.py --dataset cityscapes --method entropic-lmo-mgda --seed 42 --epochs 200 --lr 0.005 --method-params-lr 0.0001 --lmo-alpha 0.1 --muon-ns-steps 5 --entropy-tau 0 --weight-kappa 0 --lmo-lr-schedule late150 --output-root results "$@"
