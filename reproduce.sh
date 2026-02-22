#!/usr/bin/env bash
set -euo pipefail

echo "[reproduce] Python version:"
python3 --version

echo "[reproduce] Preparing output directories..."
mkdir -p thesis_outputs/tables thesis_outputs/figures thesis_outputs/logs thesis_outputs/captions

echo "[reproduce] Running pipeline..."
python3 main.py --data_dir Data --out_dir thesis_outputs

echo "[reproduce] Done. Outputs saved under:"
echo "  thesis_outputs/tables"
echo "  thesis_outputs/figures"
echo "  thesis_outputs/logs"
echo "  thesis_outputs/captions"
