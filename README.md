# Early Diffusion Signals for Predicting Reach and Veracity in Social Media Rumor Cascades

This repository reproduces the final thesis analysis. The current design uses FibVID as the main dataset, keeps Twitter15/16 as a comparative benchmark, treats size-based $k$-windows as the main specification, and reports time-window and native-graph analyses as supplementary checks.

## Research question

The project studies a timing problem in misinformation response: harmful cascades can grow before reliable verification is available. The main empirical question is whether signals observed very early in diffusion can help with two related prediction tasks:

1. eventual reach, and
2. veracity / misinformation risk.

The central idea is to go beyond early volume alone. The pipeline combines early volume and timing, residualized structural shape features, Hawkes-inspired dynamic summaries, and structure--tempo interaction terms, then tests how much each bundle adds under fixed early windows.

## What the repository reproduces

The final analysis has four parts.

1. Twitter15/16 benchmark pipeline under the original rumor-tree format.
2. FibVID main analysis under a harmonized cascade-compatible representation.
3. FibVID time-window robustness analysis.
4. FibVID native graph analysis used as a supplementary veracity check.

The main paper conclusion from this pipeline is a task asymmetry: early diffusion is much more informative for eventual reach than for veracity. In the FibVID main analysis, reach is strong under the harmonized $k$-window design, while veracity remains modest and is better interpreted as weak risk screening rather than strong classification.

## Current main findings

The final implemented pipeline supports four main takeaways.

1. On FibVID under the harmonized size-based design, reach is strongly predictable from early diffusion. The main OLS specification reaches about `R^2 = 0.886` at `k=180`, and the interaction-full specification peaks around `R^2 = 0.861` at `k=120`.
2. Veracity is much weaker. In the same FibVID main analysis, the best baseline logit result reaches about `AUC = 0.591` at `k=180`, while richer feature bundles do not produce consistent gains over that baseline.
3. Twitter15/16 show the same broad ordering under the original rumor-tree benchmark: veracity is moderate and reach is stronger. The best interaction-full benchmark results are about `AUC = 0.657` for veracity and `R^2 = 0.719` for reach.
4. Time-window results do not overturn the main pattern, but they are weaker than the size-based design, especially for reach. A supplementary native-graph FibVID analysis recovers some additional veracity signal, which suggests that veracity is more sensitive to representation choice than reach.

## Comparative snapshot

The repository still keeps a small visual comparison between the older Twitter15/16 benchmark outputs and the revised FibVID-centered analysis.

- In the original Twitter15/16 benchmark, veracity peaks around AUC `0.657`, while reach peaks around `R^2 = 0.719`.
- In the revised FibVID main analysis, veracity remains modest, but reach becomes even stronger under the harmonized $k$-window specification.

### Veracity comparison

Left: original Twitter15/16 benchmark output. Right: revised FibVID main analysis.

<p align="center">
  <img src="README_files/F1K_k_veracity_primary.png" alt="Twitter15/16 veracity benchmark" width="46%" />
  <img src="README_files/F1K_k_veracity_primary_fibvid.png" alt="FibVID veracity main analysis" width="46%" />
</p>

### Reach comparison

Left: original Twitter15/16 benchmark output. Right: revised FibVID main analysis.

<p align="center">
  <img src="README_files/F2K_k_reach_primary.png" alt="Twitter15/16 reach benchmark" width="46%" />
  <img src="README_files/F2K_k_reach_primary_fibvid.png" alt="FibVID reach main analysis" width="46%" />
</p>

## Method overview

The final pipeline uses the following logic.

1. Define an early observation window either by the first `k` posts or by elapsed time `T`.
2. Build feature bundles in a fixed sequence:
   - baseline tempo features,
   - residualized structural shape features,
   - dynamic features,
   - structure--tempo interactions.
3. Evaluate the same bundles across two tasks:
   - veracity classification,
   - reach prediction.
4. Compare the main FibVID analysis with:
   - the original Twitter15/16 benchmark,
   - time-window robustness checks,
   - a supplementary native-graph veracity check on FibVID.

For FibVID, the main pipeline uses a harmonized cascade-compatible representation. In practice, this means claim-level propagation records are converted into a single-root cascade format so that the same early-window pipeline can be applied consistently across FibVID and Twitter15/16. The native-graph script keeps the original claim-level propagation graph and is used only as a supplementary representation check.

## Main output locations

Running the full reproduction writes to two places.

- `thesis_outputs/`
  Twitter15/16 benchmark outputs from the original tree-based pipeline.
- `new data/`
  FibVID main outputs, time-window outputs, native-graph outputs, and cross-result summaries.

Important generated files include:

- `thesis_outputs/tables/results_k_primary.csv`
- `thesis_outputs/tables/permutation_test_k60.csv`
- `new data/tables/results_k_primary.csv`
- `new data/tables/results_timewin_primary_fibvid.csv`
- `new data/fibvid_graph_native/graph_veracity_results.csv`
- `new data/full_results_comparison.md`

## Which script produces what

- `python3 main.py --data_dir Data --out_dir thesis_outputs`
  Runs the original Twitter15/16 benchmark pipeline.

- `python3 scripts/run_fibvid_main_pipeline.py`
  Converts FibVID into a cascade-compatible format, runs the main harmonized analysis, and writes the main FibVID tables/figures to `new data/`.

- `python3 scripts/run_fibvid_timewin_compare.py`
  Runs the FibVID time-window robustness analysis and writes time-window tables/figures to `new data/`.

- `python3 scripts/run_fibvid_graph_native.py`
  Runs the native-graph FibVID analysis used as a supplementary veracity check.

## Data requirements

See `Data/README.md` for the expected folder layout.

The current full analysis expects both:

- Twitter15/16 tree files under `Data/rumor_detection_acl2017/`
- FibVID files under `merry555-FibVID-14b95c3/`

The FibVID scripts expect the following released tables:

- `merry555-FibVID-14b95c3/claim_propagation/claim_propagation.csv`
- `merry555-FibVID-14b95c3/news_claim/news_claim.csv`

## Environment

Python tested: `3.12.2`

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`xgboost` is optional. The code falls back automatically when it is unavailable.

## Reproduce the final analysis

Use the top-level reproduction script:

```bash
bash reproduce.sh
```

This script runs, in order:

1. the Twitter15/16 benchmark pipeline,
2. the FibVID harmonized main pipeline,
3. the FibVID time-window comparison,
4. the FibVID native graph analysis.

If you only want one component, the direct entry points are:

```bash
python3 main.py --data_dir Data --out_dir thesis_outputs
python3 scripts/run_fibvid_main_pipeline.py
python3 scripts/run_fibvid_timewin_compare.py
python3 scripts/run_fibvid_graph_native.py
```

## Manuscript source

The current thesis source is under:

- `tex/main.tex`

The active manuscript reflects the same final design documented in this README: FibVID as the main dataset, Twitter15/16 as a benchmark, size-based windows as the main specification, and time-window / native-graph analyses as supplementary checks.

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── reproduce.sh
├── main.py
├── scripts/
│   ├── run_fibvid_main_pipeline.py
│   ├── run_fibvid_timewin_compare.py
│   └── run_fibvid_graph_native.py
├── thesis_pipeline/
├── Data/
│   ├── README.md
│   └── rumor_detection_acl2017/
├── thesis_outputs/    # generated Twitter benchmark outputs
└── new data/          # generated FibVID outputs and comparison summaries
```

## Notes on scope

This repository is meant to reproduce the final implemented analysis, not every exploratory design considered during drafting. In the final paper:

- FibVID is the main empirical setting.
- Twitter15/16 remain as a comparative benchmark.
- Time windows are a robustness specification.
- The native graph analysis is supplementary and is mainly used to check veracity sensitivity to representation.

## Citation

Please cite this project using `CITATION.cff`.
