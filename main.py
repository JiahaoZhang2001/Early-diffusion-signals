#!/usr/bin/env python3
import ast
import argparse
import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

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


def save_figure(fig: plt.Figure, stem: str) -> List[str]:
    png = OUT_FIGURES / f"{stem}.png"
    pdf = OUT_FIGURES / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [str(png), str(pdf)]


def write_caption(stem: str, text: str) -> str:
    path = OUT_CAPTIONS / f"{stem}.txt"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return str(path)


@dataclass
class Cascade:
    dataset: str
    cascade_id: str
    label: str
    y_false: int
    final_size: int
    y_reach: float
    root: str
    adj: Dict[str, List[str]]
    delays: Dict[str, float]
    bfs_order: List[str]


def node_key(uid: str, tweet_id: str, delay: float) -> str:
    return f"{uid}|{tweet_id}|{delay:.6f}"


def parse_label_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or ":" not in s:
                continue
            label, cid = s.split(":", 1)
            out[cid.strip()] = label.strip().lower()
    return out


def parse_edge_line(line: str) -> Tuple[Tuple[str, str, float], Tuple[str, str, float]]:
    if "->" not in line:
        raise ValueError("missing arrow")
    left, right = line.split("->", 1)
    l_obj = ast.literal_eval(left.strip())
    r_obj = ast.literal_eval(right.strip())
    if not (isinstance(l_obj, list) and len(l_obj) == 3 and isinstance(r_obj, list) and len(r_obj) == 3):
        raise ValueError("endpoint format invalid")
    l = (str(l_obj[0]), str(l_obj[1]), float(l_obj[2]))
    r = (str(r_obj[0]), str(r_obj[1]), float(r_obj[2]))
    if l[2] < 0 or r[2] < 0:
        raise ValueError("negative delay")
    return l, r


def bfs_tree_projection(adj: Dict[str, List[str]], root: str) -> Dict[str, List[str]]:
    tree: Dict[str, List[str]] = defaultdict(list)
    visited = {root}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                tree[u].append(v)
                q.append(v)
    return tree


def bfs_depths(adj: Dict[str, List[str]], root: str) -> Tuple[List[str], Dict[str, int]]:
    order: List[str] = []
    depths: Dict[str, int] = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj.get(u, []):
            if v not in depths:
                depths[v] = depths[u] + 1
                q.append(v)
    return order, depths


def structural_virality_proxy(nodes: List[str], undirected_adj: Dict[str, List[str]], rng: np.random.Generator) -> float:
    n = len(nodes)
    if n <= 1:
        return 0.0
    if n > SAMPLE_VIRALITY_N:
        sampled = list(rng.choice(nodes, size=SAMPLE_VIRALITY_N, replace=False))
    else:
        sampled = list(nodes)
    sampled_set = set(sampled)

    def bfs_dist(src: str) -> Dict[str, int]:
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in undirected_adj.get(u, []):
                if v in sampled_set and v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    dvals: List[int] = []
    for i, u in enumerate(sampled):
        du = bfs_dist(u)
        for v in sampled[i + 1 :]:
            if v in du:
                dvals.append(du[v])
    if not dvals:
        return 0.0
    return float(np.mean(dvals))


def finite_or_nan(x: float) -> float:
    if x is None:
        return np.nan
    if isinstance(x, (float, int)) and np.isfinite(x):
        return float(x)
    return np.nan


def normalized_entropy(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    total = float(arr.sum())
    if total <= 0:
        return 0.0
    p = arr / total
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    h = -float(np.sum(p * np.log(p)))
    return float(h / np.log(float(p.size)))


def gini_coefficient(values: Sequence[float]) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.size == 0:
        return 0.0
    arr = np.maximum(arr, 0.0)
    s = float(arr.sum())
    if s <= 0:
        return 0.0
    idx = np.arange(1, arr.size + 1, dtype=float)
    return float(np.clip((2.0 * np.sum(idx * arr) / (arr.size * s)) - ((arr.size + 1.0) / arr.size), 0.0, 1.0))


def estimate_hawkes_params(event_times: Sequence[float], horizon: float) -> Dict[str, float]:
    t = np.sort(np.asarray(event_times, dtype=float))
    t = t[np.isfinite(t)]
    horizon = float(max(horizon, 1e-6))
    if t.size <= 2:
        return {
            "hawkes_mu_hat": float(t.size / horizon),
            "hawkes_alpha_hat": 0.0,
            "hawkes_branching_ratio": 0.0,
            "tempo_cv_interarrival": 0.0,
        }

    inter = np.diff(t)
    inter = inter[inter > 0]
    if inter.size == 0:
        inter = np.array([horizon], dtype=float)
    mean_inter = float(np.mean(inter))
    std_inter = float(np.std(inter))
    cv_inter = float(std_inter / (mean_inter + 1e-9))
    beta_hat = 1.0 / max(mean_inter, 1e-6)

    n_bins = int(np.clip(np.sqrt(t.size) * 2.0, 8, 32))
    edges = np.linspace(0.0, horizon, n_bins + 1)
    dt = float(np.mean(np.diff(edges)))
    counts, _ = np.histogram(t, bins=edges)

    starts = edges[:-1]
    excitation = np.zeros(n_bins, dtype=float)
    for i, s in enumerate(starts):
        prev = t[t < s]
        if prev.size:
            excitation[i] = float(np.sum(np.exp(-beta_hat * (s - prev))))

    y = counts.astype(float) / max(dt, 1e-6)
    x = np.column_stack([np.ones_like(excitation), excitation])
    theta, *_ = np.linalg.lstsq(x, y, rcond=None)
    mu_hat = float(max(theta[0], 0.0))
    alpha_hat = float(max(theta[1], 0.0))
    branching_ratio = float(np.clip(alpha_hat / max(beta_hat, 1e-9), 0.0, 2.0))
    return {
        "hawkes_mu_hat": finite_or_nan(mu_hat),
        "hawkes_alpha_hat": finite_or_nan(alpha_hat),
        "hawkes_branching_ratio": finite_or_nan(branching_ratio),
        "tempo_cv_interarrival": finite_or_nan(cv_inter),
    }


def build_subgraph_adj(full_adj: Dict[str, List[str]], obs_set: set) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    for u in obs_set:
        for v in full_adj.get(u, []):
            if v in obs_set:
                out[u].append(v)
    return out


def compute_structure_features(
    obs_nodes: List[str],
    obs_adj: Dict[str, List[str]],
    root: str,
    delays: Dict[str, float],
    obs_horizon: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    obs_nodes = list(dict.fromkeys(obs_nodes))
    n = len(obs_nodes)
    if n == 0:
        return {
            "leaf_fraction": 1.0,
            "internal_fraction": 0.0,
            "depth_mean_norm": 0.0,
            "depth_std_norm": 0.0,
            "depth_entropy_norm": 0.0,
            "depth_p90_norm": 0.0,
            "degree_gini": 0.0,
            "branching_entropy_norm": 0.0,
            "subtree_imbalance": 0.0,
            "virality_norm": 0.0,
            "hawkes_mu_hat": 0.0,
            "hawkes_alpha_hat": 0.0,
            "hawkes_branching_ratio": 0.0,
            "tempo_cv_interarrival": 0.0,
        }

    out_deg = {u: 0 for u in obs_nodes}
    obs_set = set(obs_nodes)
    for u in obs_nodes:
        out_deg[u] = sum(1 for v in obs_adj.get(u, []) if v in obs_set)

    order, depths = bfs_depths(obs_adj, root)
    depth_vals = np.asarray([depths[u] for u in order if u in depths], dtype=float)
    depth_scale = max(np.log2(n + 1.0), 1.0)
    depth_mean_norm = float(np.mean(depth_vals) / depth_scale) if depth_vals.size else 0.0
    depth_std_norm = float(np.std(depth_vals) / depth_scale) if depth_vals.size else 0.0
    depth_p90_norm = float(np.percentile(depth_vals, 90) / max(float(depth_vals.max()), 1.0)) if depth_vals.size else 0.0
    depth_counts = Counter(depth_vals.tolist())
    depth_entropy_norm = normalized_entropy(list(depth_counts.values()))

    out_deg_vals = np.asarray([out_deg.get(u, 0) for u in obs_nodes], dtype=float)
    leaves = sum(1 for u in obs_nodes if out_deg.get(u, 0) == 0)
    internal = n - leaves
    leaf_fraction = float(leaves / n)
    internal_fraction = float(internal / n)
    leaf_fraction = float(np.clip(leaf_fraction, 0.0, 1.0))
    internal_fraction = float(np.clip(internal_fraction, 0.0, 1.0))
    degree_gini = gini_coefficient(out_deg_vals)
    branching_entropy_norm = normalized_entropy([x for x in out_deg_vals if x > 0])

    subtree_size: Dict[str, int] = {u: 1 for u in order}
    for u in reversed(order):
        for v in obs_adj.get(u, []):
            if v in subtree_size:
                subtree_size[u] += subtree_size[v]
    node_imbalances: List[float] = []
    for u in order:
        children = [v for v in obs_adj.get(u, []) if v in subtree_size]
        if len(children) < 2:
            continue
        child_sizes = np.asarray([subtree_size[v] for v in children], dtype=float)
        imb = float((child_sizes.max() - child_sizes.min()) / max(child_sizes.sum(), 1e-6))
        node_imbalances.append(imb)
    subtree_imbalance = float(np.mean(node_imbalances)) if node_imbalances else 0.0

    undirected: Dict[str, List[str]] = defaultdict(list)
    for u in obs_nodes:
        for v in obs_adj.get(u, []):
            if v in obs_set:
                undirected[u].append(v)
                undirected[v].append(u)
    virality = structural_virality_proxy(obs_nodes, undirected, rng)
    virality_norm = float(np.clip(virality / max(n - 1.0, 1.0), 0.0, 2.0))

    event_times = np.sort(np.asarray([delays.get(nid, np.nan) for nid in obs_nodes], dtype=float))
    event_times = event_times[np.isfinite(event_times)]
    hawkes = estimate_hawkes_params(event_times, obs_horizon)

    feats = {
        "leaf_fraction": finite_or_nan(leaf_fraction),
        "internal_fraction": finite_or_nan(internal_fraction),
        "depth_mean_norm": finite_or_nan(depth_mean_norm),
        "depth_std_norm": finite_or_nan(depth_std_norm),
        "depth_entropy_norm": finite_or_nan(depth_entropy_norm),
        "depth_p90_norm": finite_or_nan(depth_p90_norm),
        "degree_gini": finite_or_nan(degree_gini),
        "branching_entropy_norm": finite_or_nan(branching_entropy_norm),
        "subtree_imbalance": finite_or_nan(subtree_imbalance),
        "virality_norm": finite_or_nan(virality_norm),
        **hawkes,
    }
    return {k: float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for k, v in feats.items()}


def parse_cascade(dataset: str, cascade_id: str, label: str, tree_path: Path) -> Tuple[Optional[Cascade], Optional[str]]:
    try:
        lines = [x.strip() for x in tree_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    except Exception as e:
        return None, f"read_error: {e}"
    if not lines:
        return None, "empty_tree"

    adj_raw: Dict[str, List[str]] = defaultdict(list)
    delays: Dict[str, float] = {}
    roots: List[Tuple[str, str, float]] = []

    for i, line in enumerate(lines, start=1):
        try:
            p, c = parse_edge_line(line)
        except Exception as e:
            return None, f"parse_error_line_{i}: {e}"

        if p[0] == "ROOT" and p[1] == "ROOT":
            roots.append(c)
            p_key = "__ROOT__"
        else:
            p_key = node_key(*p)
            delays[p_key] = p[2]

        c_key = node_key(*c)
        delays[c_key] = c[2]
        adj_raw[p_key].append(c_key)

    if not roots:
        return None, "missing_root_edge"

    root_match = None
    for r in roots:
        if r[1] == cascade_id:
            root_match = r
            break
    if root_match is None:
        root_match = roots[0]
        if root_match[1] != cascade_id:
            return None, f"root_mismatch:{root_match[1]}"

    root_key = node_key(*root_match)
    delays[root_key] = root_match[2]

    full_adj: Dict[str, List[str]] = defaultdict(list)
    for u, vs in adj_raw.items():
        if u == "__ROOT__":
            for v in vs:
                if v == root_key:
                    full_adj[root_key]
            continue
        for v in vs:
            full_adj[u].append(v)

    tree_adj = bfs_tree_projection(full_adj, root_key)
    bfs_order, _ = bfs_depths(tree_adj, root_key)
    if not bfs_order:
        return None, "unreachable_root"

    bfs_set = set(bfs_order)
    tree_adj_clean: Dict[str, List[str]] = defaultdict(list)
    for u in bfs_order:
        for v in tree_adj.get(u, []):
            if v in bfs_set:
                tree_adj_clean[u].append(v)

    final_size = len(bfs_order)
    y_reach = float(np.log1p(final_size))

    return (
        Cascade(
            dataset=dataset,
            cascade_id=cascade_id,
            label=label,
            y_false=1 if label == "false" else 0,
            final_size=final_size,
            y_reach=y_reach,
            root=root_key,
            adj=tree_adj_clean,
            delays={k: delays[k] for k in bfs_order},
            bfs_order=bfs_order,
        ),
        None,
    )


def make_time_feature_rows(cascades: List[Cascade], windows: Sequence[int], rng: np.random.Generator) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for c in cascades:
        for t in windows:
            obs_nodes = [n for n in c.bfs_order if c.delays[n] <= t]
            if c.root not in obs_nodes:
                obs_nodes.append(c.root)
            obs_nodes = list(dict.fromkeys(obs_nodes))
            obs_set = set(obs_nodes)
            obs_adj = build_subgraph_adj(c.adj, obs_set)

            s_feats = compute_structure_features(obs_nodes, obs_adj, c.root, c.delays, float(t), rng)
            sorted_delays = sorted(c.delays[n] for n in obs_nodes)
            row = {
                "dataset": c.dataset,
                "cascade_id": c.cascade_id,
                "label": c.label,
                "y_false": c.y_false,
                "final_size": c.final_size,
                "y_reach": c.y_reach,
                "window_type": "time",
                "window_value": int(t),
                "early_n_nodes": len(obs_nodes),
                "early_growth_rate": len(obs_nodes) / float(t),
                "obs_horizon": float(t),
                "time_to_10": sorted_delays[9] if len(sorted_delays) >= 10 else np.nan,
                "time_to_20": sorted_delays[19] if len(sorted_delays) >= 20 else np.nan,
                **s_feats,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.replace([np.inf, -np.inf], np.nan)


def make_k_feature_rows(cascades: List[Cascade], windows: Sequence[int], rng: np.random.Generator) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for c in cascades:
        for k in windows:
            obs_nodes = c.bfs_order[:k]
            if not obs_nodes:
                obs_nodes = [c.root]
            obs_set = set(obs_nodes)
            obs_adj = build_subgraph_adj(c.adj, obs_set)
            max_delay = max(c.delays[n] for n in obs_nodes)
            obs_horizon = max(float(max_delay), 1.0)
            s_feats = compute_structure_features(obs_nodes, obs_adj, c.root, c.delays, obs_horizon, rng)
            row = {
                "dataset": c.dataset,
                "cascade_id": c.cascade_id,
                "label": c.label,
                "y_false": c.y_false,
                "final_size": c.final_size,
                "y_reach": c.y_reach,
                "window_type": "k",
                "window_value": int(k),
                "early_n_nodes": len(obs_nodes),
                "early_growth_rate_k": len(obs_nodes) / obs_horizon,
                "obs_horizon": obs_horizon,
                **s_feats,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.replace([np.inf, -np.inf], np.nan)


def residualize_on_log_volume(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    out["log_volume"] = np.log1p(out["early_n_nodes"].astype(float))
    for col in feature_cols:
        resid_col = f"{col}_resid"
        out[resid_col] = np.nan
        for _, g in out.groupby(["window_type", "window_value"], sort=False):
            idx = g.index
            x = out.loc[idx, "log_volume"].to_numpy(dtype=float)
            y = out.loc[idx, col].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() == 0:
                continue
            resid = np.full_like(y, np.nan, dtype=float)
            if mask.sum() >= 3 and np.unique(x[mask]).size >= 2:
                slope, intercept = np.polyfit(x[mask], y[mask], deg=1)
                pred = intercept + slope * x[mask]
                resid[mask] = y[mask] - pred
            else:
                resid[mask] = y[mask] - float(np.mean(y[mask]))
            out.loc[idx, resid_col] = resid
        out[resid_col] = out[resid_col].fillna(0.0)
    return out


def add_structure_tempo_interactions(df: pd.DataFrame, struct_cols: Sequence[str], tempo_col: str, prefix: str) -> List[str]:
    if tempo_col not in df.columns:
        return []
    out_cols: List[str] = []
    t = df[tempo_col].astype(float)
    t_norm = (t - t.mean()) / (t.std(ddof=0) + 1e-9)
    for c in struct_cols:
        if c not in df.columns:
            continue
        s = df[c].astype(float)
        s_norm = (s - s.mean()) / (s.std(ddof=0) + 1e-9)
        name = f"{prefix}_{c}_x_{tempo_col}"
        df[name] = s_norm * t_norm
        out_cols.append(name)
    return out_cols


def volume_dependence_diagnostics(df: pd.DataFrame, raw_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (wtype, wval), g in df.groupby(["window_type", "window_value"], sort=True):
        x = np.log1p(g["early_n_nodes"].astype(float).to_numpy())
        for col in raw_cols:
            resid_col = f"{col}_resid"
            y_raw = g[col].astype(float).to_numpy() if col in g.columns else np.array([])
            y_res = g[resid_col].astype(float).to_numpy() if resid_col in g.columns else np.array([])
            mask_raw = np.isfinite(x) & np.isfinite(y_raw)
            mask_res = np.isfinite(x) & np.isfinite(y_res)
            corr_raw = float(np.corrcoef(x[mask_raw], y_raw[mask_raw])[0, 1]) if mask_raw.sum() >= 3 else np.nan
            corr_res = float(np.corrcoef(x[mask_res], y_res[mask_res])[0, 1]) if mask_res.sum() >= 3 else np.nan
            rows.append(
                {
                    "window_type": wtype,
                    "window_value": int(wval),
                    "feature": col,
                    "corr_with_log_volume_raw": corr_raw,
                    "corr_with_log_volume_resid": corr_res,
                    "n": int(len(g)),
                }
            )
    return pd.DataFrame(rows)


def make_preprocessor(features: List[str], use_missing_ind_for_time_to: bool) -> ColumnTransformer:
    time_to_cols = [c for c in features if c in {"time_to_10", "time_to_20"}]
    other_cols = [c for c in features if c not in time_to_cols]
    transformers = []
    if time_to_cols and use_missing_ind_for_time_to:
        transformers.append(("time_to", SimpleImputer(strategy="median", add_indicator=True), time_to_cols))
    elif time_to_cols:
        transformers.append(("time_to", SimpleImputer(strategy="median"), time_to_cols))
    if other_cols:
        transformers.append(("other", SimpleImputer(strategy="median"), other_cols))
    return ColumnTransformer(transformers=transformers, sparse_threshold=0)


def model_pipeline(task: str, model_name: str, features: List[str], xgb_available: bool) -> Pipeline:
    pre = make_preprocessor(features, use_missing_ind_for_time_to=True)

    if task == "veracity":
        if model_name == "logit":
            clf = LogisticRegression(
                class_weight="balanced",
                solver="liblinear",
                max_iter=5000,
                random_state=SEED,
            )
            return Pipeline([("pre", pre), ("scaler", StandardScaler()), ("model", clf)])
        if model_name == "rf":
            clf = RandomForestClassifier(
                n_estimators=500,
                random_state=SEED,
                class_weight="balanced_subsample",
                n_jobs=1,
            )
            return Pipeline([("pre", pre), ("model", clf)])
        if model_name == "xgb":
            if xgb_available:
                from xgboost import XGBClassifier

                clf = XGBClassifier(
                    n_estimators=500,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    random_state=SEED,
                    eval_metric="logloss",
                )
                return Pipeline([("pre", pre), ("model", clf)])
            clf = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                random_state=SEED,
            )
            return Pipeline([("pre", pre), ("model", clf)])

    if task == "reach":
        if model_name == "ols":
            reg = LinearRegression()
            return Pipeline([("pre", pre), ("scaler", StandardScaler()), ("model", reg)])
        if model_name == "rfreg":
            reg = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=1)
            return Pipeline([("pre", pre), ("model", reg)])
        if model_name == "xgbreg":
            if xgb_available:
                from xgboost import XGBRegressor

                reg = XGBRegressor(
                    n_estimators=500,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    random_state=SEED,
                )
                return Pipeline([("pre", pre), ("model", reg)])
            reg = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                random_state=SEED,
            )
            return Pipeline([("pre", pre), ("model", reg)])

    raise ValueError(f"Unknown task/model combination: {task}/{model_name}")


def get_cv_splits(df: pd.DataFrame, task: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    n = len(df)
    idx = np.arange(n)
    if task == "veracity":
        y = df["y_false"].to_numpy().astype(int)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        return [(tr, te) for tr, te in skf.split(idx.reshape(-1, 1), y)]
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    return [(tr, te) for tr, te in kf.split(idx)]


def mean_and_se(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def eval_veracity_scores(
    df: pd.DataFrame,
    features: List[str],
    model_name: str,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    xgb_available: bool,
) -> np.ndarray:
    x = df[features]
    y = df["y_false"].to_numpy().astype(int)
    scores: List[float] = []
    for tr, te in splits:
        pipe = model_pipeline("veracity", model_name, features, xgb_available)
        pipe.fit(x.iloc[tr], y[tr])
        if hasattr(pipe, "predict_proba"):
            p = pipe.predict_proba(x.iloc[te])[:, 1]
        else:
            p = pipe.decision_function(x.iloc[te])
        if len(np.unique(y[te])) < 2:
            continue
        scores.append(float(roc_auc_score(y[te], p)))
    return np.asarray(scores, dtype=float)


def eval_reach_scores(
    df: pd.DataFrame,
    features: List[str],
    model_name: str,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    xgb_available: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    x = df[features]
    y = df["y_reach"].to_numpy().astype(float)
    r2s: List[float] = []
    maes: List[float] = []
    for tr, te in splits:
        pipe = model_pipeline("reach", model_name, features, xgb_available)
        pipe.fit(x.iloc[tr], y[tr])
        pred = pipe.predict(x.iloc[te])
        fold_r2 = float(r2_score(y[te], pred))
        if not np.isfinite(fold_r2):
            fold_r2 = -1.0
        fold_r2 = float(np.clip(fold_r2, -1.0, 1.0))
        r2s.append(fold_r2)
        maes.append(float(mean_absolute_error(y[te], pred)))
    return np.asarray(r2s, dtype=float), np.asarray(maes, dtype=float)


def eval_veracity(df: pd.DataFrame, features: List[str], model_name: str, splits: List[Tuple[np.ndarray, np.ndarray]], xgb_available: bool) -> Tuple[float, float]:
    scores = eval_veracity_scores(df, features, model_name, splits, xgb_available)
    return mean_and_se(scores)


def eval_reach(df: pd.DataFrame, features: List[str], model_name: str, splits: List[Tuple[np.ndarray, np.ndarray]], xgb_available: bool) -> Tuple[float, float, float, float]:
    r2s, maes = eval_reach_scores(df, features, model_name, splits, xgb_available)
    r2_mean, r2_se = mean_and_se(r2s)
    mae_mean, mae_se = mean_and_se(maes)
    return r2_mean, r2_se, mae_mean, mae_se


def paired_t_test(x0: Sequence[float], x1: Sequence[float]) -> Tuple[float, float, float, float]:
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    n = min(x0.size, x1.size)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    d = (x1[:n] - x0[:n]).astype(float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    delta = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    if sd <= 1e-12:
        t_stat = 0.0 if abs(delta) <= 1e-12 else float(np.sign(delta) * 1e6)
        p_val = 1.0 if abs(delta) <= 1e-12 else 0.0
        return delta, t_stat, p_val, float("nan")
    t_stat = delta / (sd / np.sqrt(d.size))
    try:
        from scipy.stats import t as t_dist

        p_val = float(2.0 * t_dist.sf(abs(t_stat), df=d.size - 1))
    except Exception:
        p_val = float(math.erfc(abs(t_stat) / math.sqrt(2.0)))
    effect = delta / sd
    return delta, float(t_stat), p_val, float(effect)


def add_n_under_ticks(ax: plt.Axes, xvals: Sequence[int], n_map: Dict[int, int]) -> None:
    ax.set_xticks(list(xvals))
    labels = [f"{x}\nN={n_map.get(int(x), 0)}" for x in xvals]
    ax.set_xticklabels(labels)


def set_publication_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 120,
        }
    )


def plot_three_lines(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    se_col: str,
    lines: Sequence[str],
    title: str,
    ylabel: str,
    chance_line: Optional[float],
    n_map: Dict[int, int],
) -> plt.Figure:
    colors = {
        "baseline": "#1F4E79",
        "structure_only": "#B36A00",
        "full": "#2E7D32",
        "interaction_full": "#7A1F5C",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    xvals = sorted(df[x_col].unique())
    for line in lines:
        d = df[df["feature_set"] == line].sort_values(x_col)
        x = d[x_col].to_numpy()
        y = d[y_col].to_numpy()
        se = d[se_col].to_numpy()
        ci95 = 1.96 * se
        ax.errorbar(
            x,
            y,
            yerr=ci95,
            marker="o",
            linewidth=1.8,
            capsize=3.5,
            markersize=4.5,
            label=line.replace("_", "-"),
            color=colors.get(line, None),
        )
    if chance_line is not None:
        ax.axhline(chance_line, color="#666666", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Window")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.2, linestyle=":")
    add_n_under_ticks(ax, xvals, n_map)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def permutation_test(
    df: pd.DataFrame,
    features: List[str],
    task: str,
    model_name: str,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    xgb_available: bool,
    n_permutations: int,
    rng: np.random.Generator,
) -> Tuple[float, np.ndarray, float]:
    x = df[features]

    if task == "veracity":
        y = df["y_false"].to_numpy().astype(int)

        def score_fn(y_in: np.ndarray) -> float:
            vals = []
            for tr, te in splits:
                pipe = model_pipeline("veracity", model_name, features, xgb_available)
                pipe.fit(x.iloc[tr], y_in[tr])
                p = pipe.predict_proba(x.iloc[te])[:, 1]
                if len(np.unique(y_in[te])) < 2:
                    continue
                vals.append(float(roc_auc_score(y_in[te], p)))
            return float(np.mean(vals))

    else:
        y = df["y_reach"].to_numpy().astype(float)

        def score_fn(y_in: np.ndarray) -> float:
            vals = []
            for tr, te in splits:
                pipe = model_pipeline("reach", model_name, features, xgb_available)
                pipe.fit(x.iloc[tr], y_in[tr])
                pred = pipe.predict(x.iloc[te])
                fold_r2 = float(r2_score(y_in[te], pred))
                if not np.isfinite(fold_r2):
                    fold_r2 = -1.0
                fold_r2 = float(np.clip(fold_r2, -1.0, 1.0))
                vals.append(fold_r2)
            return float(np.mean(vals))

    observed = score_fn(y)
    null_scores = np.zeros(n_permutations, dtype=float)
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        null_scores[i] = score_fn(y_perm)
    p_value = float(np.mean(null_scores >= observed))
    return observed, null_scores, p_value


def main(data_dir: Path, out_dir: Path) -> None:
    configure_paths(data_dir, out_dir)
    ensure_dirs()
    set_publication_style()
    created_files: List[str] = []
    run_log_lines: List[str] = []
    exclusions: List[str] = []

    run_log_lines.append(f"Run start: {datetime.now().isoformat(timespec='seconds')}")

    # STEP 0 Parse + audit
    datasets = [("twitter15", DATA_ROOT / "twitter15"), ("twitter16", DATA_ROOT / "twitter16")]
    cascades: List[Cascade] = []
    audit_rows: List[Dict[str, Any]] = []
    label_counts = Counter()

    for dname, droot in datasets:
        labels = parse_label_file(droot / "label.txt")
        tree_dir = droot / "tree"

        for cid, label in labels.items():
            label_counts[label] += 1
            final_size = 0
            parse_ok = False

            if label not in LABELS:
                exclusions.append(f"{dname}:{cid} excluded: invalid_label={label}")
                audit_rows.append({"cascade_id": cid, "dataset": dname, "label": label, "final_size": final_size, "parse_ok": parse_ok})
                continue

            tree_path = tree_dir / f"{cid}.txt"
            if not tree_path.exists():
                exclusions.append(f"{dname}:{cid} excluded: missing_tree_file")
                audit_rows.append({"cascade_id": cid, "dataset": dname, "label": label, "final_size": final_size, "parse_ok": parse_ok})
                continue

            cascade, err = parse_cascade(dname, cid, label, tree_path)
            if cascade is None:
                exclusions.append(f"{dname}:{cid} excluded: {err}")
                audit_rows.append({"cascade_id": cid, "dataset": dname, "label": label, "final_size": final_size, "parse_ok": parse_ok})
                continue

            parse_ok = True
            final_size = cascade.final_size
            cascades.append(cascade)
            audit_rows.append({"cascade_id": cid, "dataset": dname, "label": label, "final_size": final_size, "parse_ok": parse_ok})

    audit_df = pd.DataFrame(audit_rows).sort_values(["dataset", "cascade_id"]).reset_index(drop=True)
    p_audit = OUT_TABLES / "dataset_audit.csv"
    audit_df.to_csv(p_audit, index=False)
    created_files.append(str(p_audit))

    audit_txt_lines = ["Data audit summary", f"Total labeled: {len(audit_rows)}", f"Parsed OK: {len(cascades)}", f"Exclusions: {len(exclusions)}", "", "Counts by label:"]
    for k in sorted(label_counts.keys()):
        audit_txt_lines.append(f"- {k}: {label_counts[k]}")
    audit_txt_lines.append("")
    audit_txt_lines.append("Exclusions:")
    if exclusions:
        audit_txt_lines.extend([f"- {x}" for x in exclusions])
    else:
        audit_txt_lines.append("- None")
    p_audit_txt = OUT_LOGS / "data_audit.txt"
    p_audit_txt.write_text("\n".join(audit_txt_lines) + "\n", encoding="utf-8")
    created_files.append(str(p_audit_txt))

    # Feature tables for windows
    time_df = make_time_feature_rows(cascades, TIME_WINDOWS, RNG)
    k_df = make_k_feature_rows(cascades, K_WINDOWS, RNG)
    time_df = residualize_on_log_volume(time_df, RAW_STRUCTURE_SHAPE)
    k_df = residualize_on_log_volume(k_df, RAW_STRUCTURE_SHAPE)
    time_inter_cols = add_structure_tempo_interactions(time_df, STRUCTURE_BUNDLE_C, "early_growth_rate", "int_time")
    k_inter_cols = add_structure_tempo_interactions(k_df, STRUCTURE_BUNDLE_C, "early_growth_rate_k", "int_k")

    vol_diag = pd.concat(
        [
            volume_dependence_diagnostics(time_df, RAW_STRUCTURE_SHAPE),
            volume_dependence_diagnostics(k_df, RAW_STRUCTURE_SHAPE),
        ],
        ignore_index=True,
    )
    p_vol_diag = OUT_TABLES / "volume_dependence_diagnostics.csv"
    vol_diag.to_csv(p_vol_diag, index=False)
    created_files.append(str(p_vol_diag))

    time_df = time_df.sort_values(["window_value", "dataset", "cascade_id"]).reset_index(drop=True)
    k_df = k_df.sort_values(["window_value", "dataset", "cascade_id"]).reset_index(drop=True)

    # Shared folds per time window
    veracity_splits_by_t: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    reach_splits_by_t: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    n_used_by_t: Dict[int, int] = {}
    coverage_by_t: Dict[int, float] = {}

    for t in TIME_WINDOWS:
        df_t = time_df[time_df["window_value"] == t].sort_values(["dataset", "cascade_id"]).reset_index(drop=True)
        veracity_splits_by_t[t] = get_cv_splits(df_t, "veracity")
        reach_splits_by_t[t] = get_cv_splits(df_t, "reach")
        n_used_by_t[t] = len(df_t)
        coverage_by_t[t] = float((df_t["early_n_nodes"] >= 1).mean())

    veracity_splits_by_k: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    reach_splits_by_k: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    n_used_by_k: Dict[int, int] = {}
    coverage_by_k: Dict[int, float] = {}
    for k in K_WINDOWS:
        df_k = k_df[k_df["window_value"] == k].sort_values(["dataset", "cascade_id"]).reset_index(drop=True)
        veracity_splits_by_k[k] = get_cv_splits(df_k, "veracity")
        reach_splits_by_k[k] = get_cv_splits(df_k, "reach")
        n_used_by_k[k] = len(df_k)
        coverage_by_k[k] = float((df_k["early_n_nodes"] >= 1).mean())

    use_xgb = os.environ.get("THESIS_USE_XGBOOST", "0") == "1"
    if use_xgb:
        try:
            import xgboost  # noqa: F401
            xgb_available = True
            xgb_note = "xgboost available"
        except Exception:
            xgb_available = False
            xgb_note = "xgboost unavailable; fallback to HistGradientBoosting models"
    else:
        xgb_available = False
        xgb_note = "xgboost disabled for this run (OpenMP SHM limitation); using GradientBoosting fallback"

    # STEP 1 Primary K-window results with residualized shape + dynamics + interactions
    step1_rows: List[Dict[str, Any]] = []
    step1_tests: List[Dict[str, Any]] = []
    for k in K_WINDOWS:
        df_k = k_df[k_df["window_value"] == k].sort_values(["dataset", "cascade_id"]).reset_index(drop=True)

        feature_sets = {
            "baseline": BASELINE_K,
            "structure_only": STRUCTURE_BUNDLE_C,
            "full": BASELINE_K + STRUCTURE_BUNDLE_C + DYNAMIC_BUNDLE_H,
            "interaction_full": BASELINE_K + STRUCTURE_BUNDLE_C + DYNAMIC_BUNDLE_H + k_inter_cols,
        }

        for fs_name, features in feature_sets.items():
            auc_m, auc_se = eval_veracity(df_k, features, "logit", veracity_splits_by_k[k], xgb_available)
            step1_rows.append(
                {
                    "task": "veracity",
                    "window_type": "k",
                    "window_value": k,
                    "feature_set": fs_name,
                    "model": "logit",
                    "metric": "auc",
                    "mean": auc_m,
                    "se": auc_se,
                    "N_used": len(df_k),
                }
            )

            r2_m, r2_se, mae_m, mae_se = eval_reach(df_k, features, "ols", reach_splits_by_k[k], xgb_available)
            step1_rows.append(
                {
                    "task": "reach",
                    "window_type": "k",
                    "window_value": k,
                    "feature_set": fs_name,
                    "model": "OLS",
                    "metric": "r2",
                    "mean": r2_m,
                    "se": r2_se,
                    "N_used": len(df_k),
                }
            )
            step1_rows.append(
                {
                    "task": "reach",
                    "window_type": "k",
                    "window_value": k,
                    "feature_set": fs_name,
                    "model": "OLS",
                    "metric": "mae",
                    "mean": mae_m,
                    "se": mae_se,
                    "N_used": len(df_k),
                }
            )

        # paired tests on identical folds: baseline vs interaction_full
        v_base = eval_veracity_scores(df_k, feature_sets["baseline"], "logit", veracity_splits_by_k[k], xgb_available)
        v_int = eval_veracity_scores(df_k, feature_sets["interaction_full"], "logit", veracity_splits_by_k[k], xgb_available)
        d_auc, t_auc, p_auc, e_auc = paired_t_test(v_base, v_int)
        step1_tests.append(
            {
                "task": "veracity",
                "metric": "auc",
                "window_type": "k",
                "window_value": k,
                "delta_mean": d_auc,
                "t_stat": t_auc,
                "p_value": p_auc,
                "effect_size_dz": e_auc,
                "n_folds": min(len(v_base), len(v_int)),
            }
        )

        r_base, _ = eval_reach_scores(df_k, feature_sets["baseline"], "ols", reach_splits_by_k[k], xgb_available)
        r_int, _ = eval_reach_scores(df_k, feature_sets["interaction_full"], "ols", reach_splits_by_k[k], xgb_available)
        d_r2, t_r2, p_r2, e_r2 = paired_t_test(r_base, r_int)
        step1_tests.append(
            {
                "task": "reach",
                "metric": "r2",
                "window_type": "k",
                "window_value": k,
                "delta_mean": d_r2,
                "t_stat": t_r2,
                "p_value": p_r2,
                "effect_size_dz": e_r2,
                "n_folds": min(len(r_base), len(r_int)),
            }
        )

    step1_df = pd.DataFrame(step1_rows)
    p_step1 = OUT_TABLES / "results_k_primary.csv"
    step1_df.to_csv(p_step1, index=False)
    created_files.append(str(p_step1))
    p_step1_test = OUT_TABLES / "k_primary_paired_ttests.csv"
    pd.DataFrame(step1_tests).to_csv(p_step1_test, index=False)
    created_files.append(str(p_step1_test))

    fig1_df = step1_df[(step1_df["task"] == "veracity") & (step1_df["metric"] == "auc")]
    fig1 = plot_three_lines(
        fig1_df,
        x_col="window_value",
        y_col="mean",
        se_col="se",
        lines=["baseline", "structure_only", "full", "interaction_full"],
        title="Veracity (AUC) vs K-window",
        ylabel="ROC-AUC",
        chance_line=0.5,
        n_map=n_used_by_k,
    )
    created_files.extend(save_figure(fig1, "F1K_k_veracity_primary"))

    fig2_df = step1_df[(step1_df["task"] == "reach") & (step1_df["metric"] == "r2")]
    fig2 = plot_three_lines(
        fig2_df,
        x_col="window_value",
        y_col="mean",
        se_col="se",
        lines=["baseline", "structure_only", "full", "interaction_full"],
        title="Reach (R^2) vs K-window",
        ylabel="R^2",
        chance_line=None,
        n_map=n_used_by_k,
    )
    created_files.extend(save_figure(fig2, "F2K_k_reach_primary"))

    created_files.append(
        write_caption(
            "F1K",
            "Primary size-window veracity results. Features include baseline tempo, residualized shape descriptors, Hawkes-inspired dynamics, and explicit structure-tempo interactions. Points show mean CV AUC and bars show 95% CI from SE.",
        )
    )
    created_files.append(
        write_caption(
            "F2K",
            "Primary size-window reach results. Size-based windows hold early volume fixed while testing whether residualized structural shape and dynamics add signal beyond baseline tempo. Error bars are 95% CIs from SE.",
        )
    )

    # STEP 2 Full-only model family comparison on K windows
    step2_rows: List[Dict[str, Any]] = []
    full_features_k = BASELINE_K + STRUCTURE_BUNDLE_C + DYNAMIC_BUNDLE_H + k_inter_cols
    for k in K_WINDOWS:
        df_k = k_df[k_df["window_value"] == k].sort_values(["dataset", "cascade_id"]).reset_index(drop=True)

        for m in ["logit", "rf", "xgb"]:
            auc_m, auc_se = eval_veracity(df_k, full_features_k, m, veracity_splits_by_k[k], xgb_available)
            step2_rows.append(
                {
                    "task": "veracity",
                    "window_k": k,
                    "model": m,
                    "metric": "auc",
                    "mean": auc_m,
                    "se": auc_se,
                    "N_used": len(df_k),
                }
            )

        for m in ["ols", "rfreg", "xgbreg"]:
            r2_m, r2_se, mae_m, mae_se = eval_reach(df_k, full_features_k, m, reach_splits_by_k[k], xgb_available)
            step2_rows.append(
                {
                    "task": "reach",
                    "window_k": k,
                    "model": m,
                    "metric": "r2",
                    "mean": r2_m,
                    "se": r2_se,
                    "N_used": len(df_k),
                }
            )
            step2_rows.append(
                {
                    "task": "reach",
                    "window_k": k,
                    "model": m,
                    "metric": "mae",
                    "mean": mae_m,
                    "se": mae_se,
                    "N_used": len(df_k),
                }
            )

    step2_df = pd.DataFrame(step2_rows)
    p_step2 = OUT_TABLES / "model_family_comparison_k_full.csv"
    step2_df.to_csv(p_step2, index=False)
    created_files.append(str(p_step2))

    fig3, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    colors = {"logit": "#1F4E79", "rf": "#B36A00", "xgb": "#2E7D32", "ols": "#1F4E79", "rfreg": "#B36A00", "xgbreg": "#2E7D32"}

    for m in ["logit", "rf", "xgb"]:
        d = step2_df[(step2_df["task"] == "veracity") & (step2_df["metric"] == "auc") & (step2_df["model"] == m)].sort_values("window_k")
        x = d["window_k"].to_numpy()
        y = d["mean"].to_numpy()
        se = d["se"].to_numpy()
        axes[0].errorbar(x, y, yerr=1.96 * se, marker="o", linewidth=1.8, capsize=3.0, label=m, color=colors[m])

    for m in ["ols", "rfreg", "xgbreg"]:
        d = step2_df[(step2_df["task"] == "reach") & (step2_df["metric"] == "r2") & (step2_df["model"] == m)].sort_values("window_k")
        x = d["window_k"].to_numpy()
        y = d["mean"].to_numpy()
        se = d["se"].to_numpy()
        axes[1].errorbar(x, y, yerr=1.96 * se, marker="o", linewidth=1.8, capsize=3.0, label=m, color=colors[m])

    axes[0].set_title("A: Veracity AUC (Full Features)")
    axes[1].set_title("B: Reach R^2 (Full Features)")
    axes[0].set_ylabel("AUC")
    axes[1].set_ylabel("R^2")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.2, linestyle=":")
        add_n_under_ticks(ax, K_WINDOWS, n_used_by_k)
        ax.set_xlabel("K window (nodes)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
    fig3.tight_layout()
    created_files.extend(save_figure(fig3, "F3_model_family_full_only"))

    created_files.append(
        write_caption(
            "F3",
            "Model-family comparison in fixed-size windows using residualized structural shape, Hawkes-inspired dynamics, and interaction terms. Error bars are 95% CIs from SE.",
        )
    )

    # STEP 3 Delta gain by model (full - baseline) on K windows
    step3_rows: List[Dict[str, Any]] = []
    for k in K_WINDOWS:
        df_k = k_df[k_df["window_value"] == k].sort_values(["dataset", "cascade_id"]).reset_index(drop=True)

        for m in ["logit", "rf", "xgb"]:
            b_mean, _ = eval_veracity(df_k, BASELINE_K, m, veracity_splits_by_k[k], xgb_available)
            f_mean = float(
                step2_df[
                    (step2_df["task"] == "veracity")
                    & (step2_df["window_k"] == k)
                    & (step2_df["model"] == m)
                    & (step2_df["metric"] == "auc")
                ]["mean"].iloc[0]
            )
            step3_rows.append(
                {
                    "task": "veracity",
                    "window_k": k,
                    "model": m,
                    "metric": "auc",
                    "baseline_mean": b_mean,
                    "full_mean": f_mean,
                    "delta": f_mean - b_mean,
                }
            )

        for m in ["ols", "rfreg", "xgbreg"]:
            b_r2, _, _, _ = eval_reach(df_k, BASELINE_K, m, reach_splits_by_k[k], xgb_available)
            f_r2 = float(
                step2_df[
                    (step2_df["task"] == "reach")
                    & (step2_df["window_k"] == k)
                    & (step2_df["model"] == m)
                    & (step2_df["metric"] == "r2")
                ]["mean"].iloc[0]
            )
            step3_rows.append(
                {
                    "task": "reach",
                    "window_k": k,
                    "model": m,
                    "metric": "r2",
                    "baseline_mean": b_r2,
                    "full_mean": f_r2,
                    "delta": f_r2 - b_r2,
                }
            )

    step3_df = pd.DataFrame(step3_rows)
    p_step3 = OUT_TABLES / "delta_gain_by_model_k.csv"
    step3_df.to_csv(p_step3, index=False)
    created_files.append(str(p_step3))

    fig4, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    width = 0.22

    v_df = step3_df[step3_df["task"] == "veracity"].copy()
    models_v = ["logit", "rf", "xgb"]
    x = np.arange(len(K_WINDOWS))
    for i, m in enumerate(models_v):
        d = v_df[v_df["model"] == m].sort_values("window_k")
        axes[0].bar(x + (i - 1) * width, d["delta"].to_numpy(), width=width, label=m)
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(k) for k in K_WINDOWS])
    axes[0].set_title("A: ΔAUC (Full - Baseline)")
    axes[0].set_xlabel("K window (nodes)")
    axes[0].set_ylabel("ΔAUC")
    axes[0].legend()

    r_df = step3_df[step3_df["task"] == "reach"].copy()
    models_r = ["ols", "rfreg", "xgbreg"]
    for i, m in enumerate(models_r):
        d = r_df[r_df["model"] == m].sort_values("window_k")
        axes[1].bar(x + (i - 1) * width, d["delta"].to_numpy(), width=width, label=m)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(k) for k in K_WINDOWS])
    axes[1].set_title("B: ΔR^2 (Full - Baseline)")
    axes[1].set_xlabel("K window (nodes)")
    axes[1].set_ylabel("ΔR^2")
    axes[1].legend()

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.2, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig4.tight_layout()
    created_files.extend(save_figure(fig4, "F4_delta_gain_by_model"))

    created_files.append(
        write_caption(
            "F4",
            "Performance gain from adding residualized structure, dynamics, and structure-tempo interactions in fixed-size windows. Panel A reports ΔAUC for veracity; Panel B reports ΔR^2 for reach.",
        )
    )

    # STEP 4A Permutation test at K=60 (full features)
    df60 = k_df[k_df["window_value"] == 60].sort_values(["dataset", "cascade_id"]).reset_index(drop=True)

    ver_model_perm = "logit"
    observed_auc, null_auc, p_auc = permutation_test(
        df60,
        full_features_k,
        task="veracity",
        model_name=ver_model_perm,
        splits=veracity_splits_by_k[60],
        xgb_available=xgb_available,
        n_permutations=200,
        rng=np.random.default_rng(SEED + 101),
    )

    observed_r2, null_r2, p_r2 = permutation_test(
        df60,
        full_features_k,
        task="reach",
        model_name="ols",
        splits=reach_splits_by_k[60],
        xgb_available=xgb_available,
        n_permutations=200,
        rng=np.random.default_rng(SEED + 202),
    )

    perm_summary = pd.DataFrame(
        [
            {
                "task": "veracity",
                "model": ver_model_perm,
                "metric": "auc",
                "window_type": "k",
                "window_value": 60,
                "observed_score": observed_auc,
                "null_mean": float(np.mean(null_auc)),
                "null_std": float(np.std(null_auc)),
                "p_value": p_auc,
                "n_permutations": 200,
            },
            {
                "task": "reach",
                "model": "ols",
                "metric": "r2",
                "window_type": "k",
                "window_value": 60,
                "observed_score": observed_r2,
                "null_mean": float(np.mean(null_r2)),
                "null_std": float(np.std(null_r2)),
                "p_value": p_r2,
                "n_permutations": 200,
            },
        ]
    )
    p_perm_summary = OUT_TABLES / "permutation_test_k60.csv"
    perm_summary.to_csv(p_perm_summary, index=False)
    created_files.append(str(p_perm_summary))

    perm_null_rows = []
    for i, s in enumerate(null_auc):
        perm_null_rows.append({"task": "veracity", "metric": "auc", "perm_idx": i, "null_score": float(s)})
    for i, s in enumerate(null_r2):
        perm_null_rows.append({"task": "reach", "metric": "r2", "perm_idx": i, "null_score": float(s)})
    perm_null_df = pd.DataFrame(perm_null_rows)
    p_perm_null = OUT_TABLES / "permutation_null_scores_k60.csv"
    perm_null_df.to_csv(p_perm_null, index=False)
    created_files.append(str(p_perm_null))

    fig5, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(null_auc, bins=20, color="#4C78A8", alpha=0.8)
    axes[0].axvline(observed_auc, color="red", linewidth=2)
    axes[0].set_title(f"Veracity AUC null (p={p_auc:.3f})")
    axes[0].set_xlabel("CV AUC")
    axes[0].set_ylabel("Count")

    axes[1].hist(null_r2, bins=20, color="#F58518", alpha=0.8)
    axes[1].axvline(observed_r2, color="red", linewidth=2)
    axes[1].set_title(f"Reach R^2 null (p={p_r2:.3f})")
    axes[1].set_xlabel("CV R^2")
    axes[1].set_ylabel("Count")

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
    fig5.tight_layout()
    created_files.extend(save_figure(fig5, "F5_permutation_test_signal"))

    created_files.append(
        write_caption(
            "F5",
            "Permutation test at K=60 with residualized shape, dynamics, and interactions. Histograms show null score distributions from 200 label/target shuffles under fixed CV splits.",
        )
    )

    # STEP 4B Time-window supplementary curve using dynamic + interaction features
    lc_rows: List[Dict[str, Any]] = []
    time_curve_df = make_time_feature_rows(cascades, T_CURVE, np.random.default_rng(SEED + 404))
    time_curve_df = residualize_on_log_volume(time_curve_df, RAW_STRUCTURE_SHAPE)
    time_curve_inter_cols = add_structure_tempo_interactions(time_curve_df, STRUCTURE_BUNDLE_C, "early_growth_rate", "int_curve")
    full_features_time = BASELINE_TIME + STRUCTURE_BUNDLE_C + DYNAMIC_BUNDLE_H + time_curve_inter_cols
    n_used_curve: Dict[int, int] = {}
    for t in T_CURVE:
        dft = time_curve_df[time_curve_df["window_value"] == t].sort_values(["dataset", "cascade_id"]).reset_index(drop=True)
        n_used_curve[t] = len(dft)
        v_splits = get_cv_splits(dft, "veracity")
        r_splits = get_cv_splits(dft, "reach")

        auc_m, auc_se = eval_veracity(dft, full_features_time, "logit", v_splits, xgb_available)
        lc_rows.append({"task": "veracity", "window_minutes": t, "model": "logit", "metric": "auc", "mean": auc_m, "se": auc_se, "N_used": len(dft)})

        r2_m, r2_se, _, _ = eval_reach(dft, full_features_time, "ols", r_splits, xgb_available)
        lc_rows.append({"task": "reach", "window_minutes": t, "model": "ols", "metric": "r2", "mean": r2_m, "se": r2_se, "N_used": len(dft)})

    lc_df = pd.DataFrame(lc_rows)
    p_lc = OUT_TABLES / "learning_curve_time.csv"
    lc_df.to_csv(p_lc, index=False)
    created_files.append(str(p_lc))

    fig6, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

    d_auc = lc_df[(lc_df["task"] == "veracity") & (lc_df["metric"] == "auc")].sort_values("window_minutes")
    x = d_auc["window_minutes"].to_numpy()
    y = d_auc["mean"].to_numpy()
    se = d_auc["se"].to_numpy()
    axes[0].errorbar(x, y, yerr=1.96 * se, marker="o", linewidth=1.8, capsize=3.0, color="#1F4E79")
    axes[0].set_title("A: Learning Curve (Veracity AUC)")
    axes[0].set_ylabel("AUC")

    d_r2 = lc_df[(lc_df["task"] == "reach") & (lc_df["metric"] == "r2")].sort_values("window_minutes")
    x2 = d_r2["window_minutes"].to_numpy()
    y2 = d_r2["mean"].to_numpy()
    se2 = d_r2["se"].to_numpy()
    axes[1].errorbar(x2, y2, yerr=1.96 * se2, marker="o", linewidth=1.8, capsize=3.0, color="#B36A00")
    axes[1].set_title("B: Learning Curve (Reach R^2)")
    axes[1].set_ylabel("R^2")

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.2, linestyle=":")
        add_n_under_ticks(ax, T_CURVE, n_used_curve)
        ax.set_xlabel("Time window (minutes)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig6.tight_layout()
    created_files.extend(save_figure(fig6, "F6_learning_curve_time"))

    created_files.append(
        write_caption(
            "F6",
            "Supplementary time-window signal accumulation with residualized structure, dynamics, and interactions. Error bars are 95% CIs from SE.",
        )
    )

    # STEP 5 K-window appendix A1-A2
    step5_rows: List[Dict[str, Any]] = []
    cov_rows: List[Dict[str, Any]] = []
    k_exclusion_log: List[str] = []

    for k in K_WINDOWS:
        dfk = k_df[k_df["window_value"] == k].sort_values(["dataset", "cascade_id"]).reset_index(drop=True)
        n_total = len(dfk)
        n_ge_k = int((dfk["final_size"] >= k).sum())
        cov = float(n_ge_k / n_total) if n_total else np.nan
        cov_rows.append({"k": k, "coverage": cov, "n_total": n_total, "n_ge_k": n_ge_k})

        v_splits = get_cv_splits(dfk, "veracity")
        r_splits = get_cv_splits(dfk, "reach")

        feature_sets = {
            "baseline": BASELINE_K,
            "structure_only": STRUCTURE_BUNDLE_C,
            "full": BASELINE_K + STRUCTURE_BUNDLE_C + DYNAMIC_BUNDLE_H,
            "interaction_full": BASELINE_K + STRUCTURE_BUNDLE_C + DYNAMIC_BUNDLE_H + k_inter_cols,
        }

        for fs_name, features in feature_sets.items():
            auc_m, auc_se = eval_veracity(dfk, features, "logit", v_splits, xgb_available)
            step5_rows.append(
                {
                    "task": "veracity",
                    "window_type": "k",
                    "window_value": k,
                    "feature_set": fs_name,
                    "model": "logit",
                    "metric": "auc",
                    "mean": auc_m,
                    "se": auc_se,
                    "N_used": len(dfk),
                }
            )

            r2_m, r2_se, mae_m, mae_se = eval_reach(dfk, features, "ols", r_splits, xgb_available)
            step5_rows.append(
                {
                    "task": "reach",
                    "window_type": "k",
                    "window_value": k,
                    "feature_set": fs_name,
                    "model": "OLS",
                    "metric": "r2",
                    "mean": r2_m,
                    "se": r2_se,
                    "N_used": len(dfk),
                }
            )
            step5_rows.append(
                {
                    "task": "reach",
                    "window_type": "k",
                    "window_value": k,
                    "feature_set": fs_name,
                    "model": "OLS",
                    "metric": "mae",
                    "mean": mae_m,
                    "se": mae_se,
                    "N_used": len(dfk),
                }
            )

            # finite-after-imputation check
            pre = make_preprocessor(features, use_missing_ind_for_time_to=True)
            transformed = pre.fit_transform(dfk[features])
            finite_ok = bool(np.isfinite(transformed).all())
            if not finite_ok:
                k_exclusion_log.append(f"K={k}, feature_set={fs_name}: non-finite values remain after imputation")

        excluded_ids = [c.cascade_id for c in cascades if c.final_size < 1]
        if excluded_ids:
            for cid in excluded_ids:
                k_exclusion_log.append(f"K={k}: excluded {cid} reason=empty_observation")
        else:
            k_exclusion_log.append(f"K={k}: no cascades excluded for modeling (all parsed cascades included; <K handled with partial BFS window)")

    step5_df = pd.DataFrame(step5_rows)
    p_step5 = OUT_TABLES / "results_k_main.csv"
    step5_df.to_csv(p_step5, index=False)
    created_files.append(str(p_step5))

    cov_df = pd.DataFrame(cov_rows)
    p_cov = OUT_TABLES / "coverage_k.csv"
    cov_df.to_csv(p_cov, index=False)
    created_files.append(str(p_cov))

    fig_a1_df = step5_df[(step5_df["task"] == "veracity") & (step5_df["metric"] == "auc")]
    n_used_by_k = {int(k): int(step5_df[(step5_df["window_value"] == k)]["N_used"].iloc[0]) for k in K_WINDOWS}
    fig_a1 = plot_three_lines(
        fig_a1_df,
        x_col="window_value",
        y_col="mean",
        se_col="se",
        lines=["baseline", "structure_only", "full", "interaction_full"],
        title="Veracity (AUC) vs K-window",
        ylabel="ROC-AUC",
        chance_line=0.5,
        n_map=n_used_by_k,
    )
    created_files.extend(save_figure(fig_a1, "A1K_k_veracity_baseline_full"))

    fig_a2_df = step5_df[(step5_df["task"] == "reach") & (step5_df["metric"] == "r2")]
    fig_a2 = plot_three_lines(
        fig_a2_df,
        x_col="window_value",
        y_col="mean",
        se_col="se",
        lines=["baseline", "structure_only", "full", "interaction_full"],
        title="Reach (R^2) vs K-window",
        ylabel="R^2",
        chance_line=None,
        n_map=n_used_by_k,
    )
    created_files.extend(save_figure(fig_a2, "A2K_k_reach_baseline_full"))

    created_files.append(
        write_caption(
            "A1K",
            "Supplementary K-window veracity view with baseline, residualized shape-only, full dynamic, and interaction-full feature sets. Error bars are 95% CIs from SE.",
        )
    )
    created_files.append(
        write_caption(
            "A2K",
            "Supplementary K-window reach view with baseline, residualized shape-only, full dynamic, and interaction-full feature sets. Error bars are 95% CIs from SE.",
        )
    )

    # Tuned decision-tree check at K=60 using full interaction feature set.
    df60_tree = k_df[k_df["window_value"] == 60].sort_values(["dataset", "cascade_id"]).reset_index(drop=True)
    tree_features = full_features_k
    x_tree = df60_tree[tree_features]
    y_tree_cls = df60_tree["y_false"].to_numpy().astype(int)
    y_tree_reg = df60_tree["y_reach"].to_numpy().astype(float)

    cls_splits = get_cv_splits(df60_tree, "veracity")
    reg_splits = get_cv_splits(df60_tree, "reach")

    tree_auc_scores: List[float] = []
    for tr, te in cls_splits:
        pre = make_preprocessor(tree_features, use_missing_ind_for_time_to=True)
        pipe = Pipeline([("pre", pre), ("tree", DecisionTreeClassifier(**TUNED_TREE_CLS_PARAMS))])
        pipe.fit(x_tree.iloc[tr], y_tree_cls[tr])
        prob = pipe.predict_proba(x_tree.iloc[te])[:, 1]
        if len(np.unique(y_tree_cls[te])) >= 2:
            tree_auc_scores.append(float(roc_auc_score(y_tree_cls[te], prob)))

    tree_r2_scores: List[float] = []
    tree_mae_scores: List[float] = []
    for tr, te in reg_splits:
        pre = make_preprocessor(tree_features, use_missing_ind_for_time_to=True)
        pipe = Pipeline([("pre", pre), ("tree", DecisionTreeRegressor(**TUNED_TREE_REG_PARAMS))])
        pipe.fit(x_tree.iloc[tr], y_tree_reg[tr])
        pred = pipe.predict(x_tree.iloc[te])
        fold_r2 = float(r2_score(y_tree_reg[te], pred))
        if not np.isfinite(fold_r2):
            fold_r2 = -1.0
        fold_r2 = float(np.clip(fold_r2, -1.0, 1.0))
        tree_r2_scores.append(fold_r2)
        tree_mae_scores.append(float(mean_absolute_error(y_tree_reg[te], pred)))

    tree_eval_df = pd.DataFrame(
        [
            {
                "task": "veracity",
                "window_type": "k",
                "window_value": 60,
                "model": "decision_tree_tuned",
                "metric": "auc",
                "mean": mean_and_se(tree_auc_scores)[0],
                "se": mean_and_se(tree_auc_scores)[1],
                "N_used": len(df60_tree),
            },
            {
                "task": "reach",
                "window_type": "k",
                "window_value": 60,
                "model": "decision_tree_tuned",
                "metric": "r2",
                "mean": mean_and_se(tree_r2_scores)[0],
                "se": mean_and_se(tree_r2_scores)[1],
                "N_used": len(df60_tree),
            },
            {
                "task": "reach",
                "window_type": "k",
                "window_value": 60,
                "model": "decision_tree_tuned",
                "metric": "mae",
                "mean": mean_and_se(tree_mae_scores)[0],
                "se": mean_and_se(tree_mae_scores)[1],
                "N_used": len(df60_tree),
            },
        ]
    )
    p_tree_eval = OUT_TABLES / "tree_tuned_k60.csv"
    tree_eval_df.to_csv(p_tree_eval, index=False)
    created_files.append(str(p_tree_eval))

    # Final logs/config
    run_config = {
        "seed": SEED,
        "time_windows": TIME_WINDOWS,
        "k_windows": K_WINDOWS,
        "learning_curve_windows": T_CURVE,
        "baseline_time": BASELINE_TIME,
        "baseline_k": BASELINE_K,
        "structure_bundle": "Residualized shape bundle + Hawkes dynamics",
        "raw_structure_shape_features": RAW_STRUCTURE_SHAPE,
        "structure_features": STRUCTURE_BUNDLE_C,
        "dynamic_features": DYNAMIC_BUNDLE_H,
        "time_interaction_features": time_inter_cols,
        "k_interaction_features": k_inter_cols,
        "cv_veracity": {"type": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": 42},
        "cv_reach": {"type": "KFold", "n_splits": 5, "shuffle": True, "random_state": 42},
        "xgboost": xgb_available,
        "xgb_note": xgb_note,
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(ROOT),
    }
    p_run_config = OUT_LOGS / "run_config.json"
    p_run_config.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    created_files.append(str(p_run_config))

    # sanity check for absurd R^2 in K pipeline
    absurd_r2 = step5_df[(step5_df["task"] == "reach") & (step5_df["metric"] == "r2") & (step5_df["mean"].abs() > 5.0)]
    if not absurd_r2.empty:
        k_exclusion_log.append("Warning: absurd R^2 detected (>5 absolute). Review leakage/scale checks.")
    else:
        k_exclusion_log.append("R^2 sanity check passed for K-window models (no absurd values).")

    run_log_lines.append(xgb_note)
    run_log_lines.append(f"tuned_tree_cls_params={TUNED_TREE_CLS_PARAMS}")
    run_log_lines.append(f"tuned_tree_reg_params={TUNED_TREE_REG_PARAMS}")
    run_log_lines.append(f"Parsed cascades: {len(cascades)}")
    run_log_lines.append(f"Excluded cascades: {len(exclusions)}")
    run_log_lines.append("Time-window coverage (>=1 observed node):")
    for t in TIME_WINDOWS:
        run_log_lines.append(f"- T={t}: coverage={coverage_by_t[t]:.4f}, N_used={n_used_by_t[t]}")
    run_log_lines.append("K-window coverage (>=1 observed node):")
    for k in K_WINDOWS:
        run_log_lines.append(f"- K={k}: coverage={coverage_by_k[k]:.4f}, N_used={n_used_by_k[k]}")
    run_log_lines.append("K-window exclusions/checks:")
    run_log_lines.extend([f"- {x}" for x in k_exclusion_log])
    run_log_lines.append("Created files:")
    run_log_lines.extend([f"- {p}" for p in sorted(set(created_files))])

    p_run_log = OUT_LOGS / "run_log.txt"
    p_run_log.write_text("\n".join(run_log_lines) + "\n", encoding="utf-8")
    created_files.append(str(p_run_log))

    print("Generated outputs:")
    for p in sorted(set(created_files)):
        print(p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run thesis diffusion analysis pipeline.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("Data"),
        help="Path to data root. If it contains rumor_detection_acl2017/ , that subdir is used automatically.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("thesis_outputs"),
        help="Directory for generated outputs (tables/figures/logs/captions).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(data_dir=args.data_dir, out_dir=args.out_dir)
