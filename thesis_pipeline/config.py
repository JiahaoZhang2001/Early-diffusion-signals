from pathlib import Path
from typing import List

import numpy as np

SEED = 42
RNG = np.random.default_rng(SEED)

ROOT = Path(".").resolve()
DATA_ROOT = ROOT / "Data" / "rumor_detection_acl2017"
OUT_ROOT = ROOT / "thesis_outputs"
OUT_TABLES = OUT_ROOT / "tables"
OUT_FIGURES = OUT_ROOT / "figures"
OUT_CAPTIONS = OUT_ROOT / "captions"
OUT_LOGS = OUT_ROOT / "logs"

TIME_WINDOWS = [10, 20, 30, 45, 60, 90, 120, 180]
K_WINDOWS = [10, 20, 30, 45, 60, 90, 120, 180]
T_CURVE = [10, 20, 30, 45, 60, 90, 120, 180]
LABELS = {"true", "false", "unverified", "non-rumor"}

BASELINE_TIME = ["early_n_nodes", "early_growth_rate", "time_to_10", "time_to_20", "obs_horizon"]
BASELINE_K = ["early_n_nodes", "early_growth_rate_k", "obs_horizon"]
RAW_STRUCTURE_SHAPE = [
    "leaf_fraction",
    "internal_fraction",
    "depth_mean_norm",
    "depth_std_norm",
    "depth_entropy_norm",
    "depth_p90_norm",
    "degree_gini",
    "branching_entropy_norm",
    "subtree_imbalance",
    "virality_norm",
]
STRUCTURE_BUNDLE_C = [f"{c}_resid" for c in RAW_STRUCTURE_SHAPE]
DYNAMIC_BUNDLE_H = [
    "hawkes_mu_hat",
    "hawkes_alpha_hat",
    "hawkes_branching_ratio",
    "tempo_cv_interarrival",
]

SAMPLE_VIRALITY_N = 150

# Rounded tuned tree params (nearest 5/10), with ccp_alpha kept unchanged.
TUNED_TREE_CLS_PARAMS = {
    "criterion": "entropy",
    "max_depth": 10,
    "min_samples_leaf": 35,
    "min_samples_split": 75,
    "ccp_alpha": 7.787658410143284e-06,
    "random_state": SEED,
    "class_weight": "balanced",
}

TUNED_TREE_REG_PARAMS = {
    "criterion": "friedman_mse",
    "max_depth": 10,
    "min_samples_leaf": 35,
    "min_samples_split": 10,
    "ccp_alpha": 0.00037348188749214417,
    "random_state": SEED,
}


def resolve_dataset_root(data_dir: Path) -> Path:
    data_dir = data_dir.resolve()
    nested = data_dir / "rumor_detection_acl2017"
    if nested.exists():
        return nested
    return data_dir


def configure_paths(data_dir: Path, out_dir: Path) -> None:
    global ROOT, DATA_ROOT, OUT_ROOT, OUT_TABLES, OUT_FIGURES, OUT_CAPTIONS, OUT_LOGS
    ROOT = Path(".").resolve()
    DATA_ROOT = resolve_dataset_root(data_dir)
    OUT_ROOT = out_dir.resolve()
    OUT_TABLES = OUT_ROOT / "tables"
    OUT_FIGURES = OUT_ROOT / "figures"
    OUT_CAPTIONS = OUT_ROOT / "captions"
    OUT_LOGS = OUT_ROOT / "logs"


def ensure_dirs() -> None:
    for p in [OUT_ROOT, OUT_TABLES, OUT_FIGURES, OUT_CAPTIONS, OUT_LOGS]:
        p.mkdir(parents=True, exist_ok=True)
