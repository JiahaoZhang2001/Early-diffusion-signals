# Data Setup

The final repository uses two data sources with different roles.

## 1. Twitter15/16 benchmark data

This is the original rumor-tree benchmark used for the comparative analysis. The expected layout is:

```text
Data/
└── rumor_detection_acl2017/
    ├── twitter15/
    │   ├── label.txt
    │   └── tree/
    │       ├── <cascade_id>.txt
    │       └── ...
    └── twitter16/
        ├── label.txt
        └── tree/
            ├── <cascade_id>.txt
            └── ...
```

Required files:

- `Data/rumor_detection_acl2017/twitter15/label.txt`
- `Data/rumor_detection_acl2017/twitter16/label.txt`
- `Data/rumor_detection_acl2017/twitter15/tree/*.txt`
- `Data/rumor_detection_acl2017/twitter16/tree/*.txt`

These files are used by:

- `python3 main.py --data_dir Data --out_dir thesis_outputs`

## 2. FibVID main data

FibVID is the main dataset in the final thesis analysis. The scripts expect the released folder under:

```text
merry555-FibVID-14b95c3/
├── claim_propagation/
│   └── claim_propagation.csv
└── news_claim/
    └── news_claim.csv
```

Required files:

- `merry555-FibVID-14b95c3/claim_propagation/claim_propagation.csv`
- `merry555-FibVID-14b95c3/news_claim/news_claim.csv`

These files are used by:

- `python3 scripts/run_fibvid_main_pipeline.py`
- `python3 scripts/run_fibvid_timewin_compare.py`
- `python3 scripts/run_fibvid_graph_native.py`

## What each script writes

- `main.py` writes the Twitter15/16 benchmark outputs to `thesis_outputs/`
- `run_fibvid_main_pipeline.py` writes the harmonized FibVID main analysis to `new data/`
- `run_fibvid_timewin_compare.py` adds FibVID time-window robustness outputs to `new data/`
- `run_fibvid_graph_native.py` writes the native graph supplementary results to `new data/fibvid_graph_native/`

## Reproduce everything

From the repository root:

```bash
bash reproduce.sh
```

This runs the benchmark and the final FibVID analyses in sequence.
