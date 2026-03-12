#!/usr/bin/env bash
set -euo pipefail

echo "[reproduce] Python version:"
python3 --version

echo "[reproduce] Working directory:"
pwd

echo "[reproduce] Preparing output directories..."
mkdir -p thesis_outputs/tables thesis_outputs/figures thesis_outputs/logs
mkdir -p "new data"

echo "[reproduce] Step 1/4: Twitter15/16 comparative benchmark"
python3 main.py --data_dir Data --out_dir thesis_outputs 2>&1 | tee thesis_outputs/logs/run.log

echo "[reproduce] Step 2/4: FibVID harmonized main analysis"
python3 scripts/run_fibvid_main_pipeline.py

echo "[reproduce] Step 3/4: FibVID time-window robustness"
python3 scripts/run_fibvid_timewin_compare.py

echo "[reproduce] Step 4/4: FibVID native graph supplementary analysis"
python3 scripts/run_fibvid_graph_native.py

echo "[reproduce] Done."
echo "[reproduce] Twitter benchmark outputs: thesis_outputs/"
echo "[reproduce] FibVID outputs: new data/"
