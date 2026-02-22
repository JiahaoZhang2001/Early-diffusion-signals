#!/usr/bin/env bash
set -euo pipefail

echo "[reproduce] Python version:"
python3 --version

echo "[reproduce] Working directory:"
pwd

echo "[reproduce] Preparing output directories..."
mkdir -p thesis_outputs/tables thesis_outputs/figures thesis_outputs/logs

echo "[reproduce] Running pipeline..."
python3 main.py --data_dir Data --out_dir thesis_outputs 2>&1 | tee thesis_outputs/logs/run.log

echo "Done. Outputs in thesis_outputs/ ..."
