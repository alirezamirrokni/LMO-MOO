#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python run_multimnist_suite.py --methods ours "$@"
