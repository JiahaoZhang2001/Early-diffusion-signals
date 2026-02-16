#!/usr/bin/env python3
import ast
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
DATA_ROOT = ROOT / "rumor_detection_acl2017"
OUT_ROOT = ROOT / "thesis_outputs"
OUT_TABLES = OUT_ROOT / "tables"
OUT_FIGURES = OUT_ROOT / "figures"
OUT_CAPTIONS = OUT_ROOT / "captions"
OUT_LOGS = OUT_ROOT / "logs"

TIME_WINDOWS = [30, 60, 180]
K_WINDOWS = [10, 20, 50]
T_CURVE = [10, 20, 30, 45, 60, 90, 120, 180]
LABELS = {"true", "false", "unverified", "non-rumor"}

BASELINE_TIME = ["early_n_nodes", "early_growth_rate", "time_to_10", "time_to_20"]
BASELINE_K = ["early_n_nodes", "early_growth_rate_k"]
STRUCTURE_BUNDLE_C = ["leaf_fraction", "avg_root_distance", "structural_virality_proxy"]

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


def build_subgraph_adj(full_adj: Dict[str, List[str]], obs_set: set) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    for u in obs_set:
        for v in full_adj.get(u, []):
            if v in obs_set:
                out[u].append(v)
    return out


def compute_structure_features(obs_nodes: List[str], obs_adj: Dict[str, List[str]], root: str, rng: np.random.Generator) -> Dict[str, float]:
    obs_nodes = list(dict.fromkeys(obs_nodes))
    n = len(obs_nodes)
    if n == 0:
        return {
            "leaf_fraction": 1.0,
            "avg_root_distance": 0.0,
            "structural_virality_proxy": 0.0,
        }

    out_deg = {u: 0 for u in obs_nodes}
    obs_set = set(obs_nodes)
    for u in obs_nodes:
        out_deg[u] = sum(1 for v in obs_adj.get(u, []) if v in obs_set)

    _, depths = bfs_depths(obs_adj, root)
    avg_dist = float(np.mean(list(depths.values()))) if depths else 0.0
    avg_dist = float(np.clip(avg_dist, 0.0, float(n)))

    leaves = sum(1 for u in obs_nodes if out_deg.get(u, 0) == 0)
    leaf_fraction = float(leaves / n)
    leaf_fraction = float(np.clip(leaf_fraction, 0.0, 1.0))

    undirected: Dict[str, List[str]] = defaultdict(list)
    for u in obs_nodes:
        for v in obs_adj.get(u, []):
            if v in obs_set:
                undirected[u].append(v)
                undirected[v].append(u)
    virality = structural_virality_proxy(obs_nodes, undirected, rng)
    virality = float(np.clip(virality, 0.0, float(n)))

    return {
        "leaf_fraction": finite_or_nan(leaf_fraction),
        "avg_root_distance": finite_or_nan(avg_dist),
        "structural_virality_proxy": finite_or_nan(virality),
    }


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

            s_feats = compute_structure_features(obs_nodes, obs_adj, c.root, rng)
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
            s_feats = compute_structure_features(obs_nodes, obs_adj, c.root, rng)
            max_delay = max(c.delays[n] for n in obs_nodes)
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
                "early_growth_rate_k": len(obs_nodes) / (max_delay + 1e-6),
                **s_feats,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.replace([np.inf, -np.inf], np.nan)


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


def eval_veracity(df: pd.DataFrame, features: List[str], model_name: str, splits: List[Tuple[np.ndarray, np.ndarray]], xgb_available: bool) -> Tuple[float, float]:
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
    return float(np.mean(scores)), float(np.std(scores))


def eval_reach(df: pd.DataFrame, features: List[str], model_name: str, splits: List[Tuple[np.ndarray, np.ndarray]], xgb_available: bool) -> Tuple[float, float, float, float]:
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
    return float(np.mean(r2s)), float(np.std(r2s)), float(np.mean(maes)), float(np.std(maes))


def add_n_under_ticks(ax: plt.Axes, xvals: Sequence[int], n_map: Dict[int, int]) -> None:
    ax.set_xticks(list(xvals))
    labels = [f"{x}\nN={n_map.get(int(x), 0)}" for x in xvals]
    ax.set_xticklabels(labels)


def plot_three_lines(df: pd.DataFrame, x_col: str, y_col: str, std_col: str, lines: Sequence[str], title: str, ylabel: str, chance_line: Optional[float], n_map: Dict[int, int]) -> plt.Figure:
    colors = {"baseline": "#4C78A8", "structure_only": "#F58518", "full": "#54A24B"}
    fig, ax = plt.subplots(figsize=(8, 5))
    xvals = sorted(df[x_col].unique())
    for line in lines:
        d = df[df["feature_set"] == line].sort_values(x_col)
        x = d[x_col].to_numpy()
        y = d[y_col].to_numpy()
        s = d[std_col].to_numpy()
        ax.plot(x, y, marker="o", linewidth=2, label=line.replace("_", "-"), color=colors.get(line, None))
        ax.fill_between(x, y - s, y + s, alpha=0.2, color=colors.get(line, None))
    if chance_line is not None:
        ax.axhline(chance_line, color="gray", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Window")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    add_n_under_ticks(ax, xvals, n_map)
    ax.legend()
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


def main() -> None:
    ensure_dirs()
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

    # STEP 1 Time-window main figures and table (baseline models only)
    step1_rows: List[Dict[str, Any]] = []
    for t in TIME_WINDOWS:
        df_t = time_df[time_df["window_value"] == t].sort_values(["dataset", "cascade_id"]).reset_index(drop=True)

        feature_sets = {
            "baseline": BASELINE_TIME,
            "structure_only": STRUCTURE_BUNDLE_C,
            "full": BASELINE_TIME + STRUCTURE_BUNDLE_C,
        }

        for fs_name, features in feature_sets.items():
            auc_m, auc_s = eval_veracity(df_t, features, "logit", veracity_splits_by_t[t], xgb_available)
            step1_rows.append(
                {
                    "task": "veracity",
                    "window_type": "time",
                    "window_value": t,
                    "feature_set": fs_name,
                    "model": "logit",
                    "metric": "auc",
                    "mean": auc_m,
                    "std": auc_s,
                    "N_used": len(df_t),
                }
            )

            r2_m, r2_s, mae_m, mae_s = eval_reach(df_t, features, "ols", reach_splits_by_t[t], xgb_available)
            step1_rows.append(
                {
                    "task": "reach",
                    "window_type": "time",
                    "window_value": t,
                    "feature_set": fs_name,
                    "model": "OLS",
                    "metric": "r2",
                    "mean": r2_m,
                    "std": r2_s,
                    "N_used": len(df_t),
                }
            )
            step1_rows.append(
                {
                    "task": "reach",
                    "window_type": "time",
                    "window_value": t,
                    "feature_set": fs_name,
                    "model": "OLS",
                    "metric": "mae",
                    "mean": mae_m,
                    "std": mae_s,
                    "N_used": len(df_t),
                }
            )

    step1_df = pd.DataFrame(step1_rows)
    p_step1 = OUT_TABLES / "results_time_main.csv"
    step1_df.to_csv(p_step1, index=False)
    created_files.append(str(p_step1))

    fig1_df = step1_df[(step1_df["task"] == "veracity") & (step1_df["metric"] == "auc")]
    fig1 = plot_three_lines(
        fig1_df,
        x_col="window_value",
        y_col="mean",
        std_col="std",
        lines=["baseline", "structure_only", "full"],
        title="Veracity (AUC) vs Time Window",
        ylabel="ROC-AUC",
        chance_line=0.5,
        n_map=n_used_by_t,
    )
    created_files.extend(save_figure(fig1, "F1T_time_veracity_baseline_full"))

    fig2_df = step1_df[(step1_df["task"] == "reach") & (step1_df["metric"] == "r2")]
    fig2 = plot_three_lines(
        fig2_df,
        x_col="window_value",
        y_col="mean",
        std_col="std",
        lines=["baseline", "structure_only", "full"],
        title="Reach (R^2) vs Time Window",
        ylabel="R^2",
        chance_line=None,
        n_map=n_used_by_t,
    )
    created_files.extend(save_figure(fig2, "F2T_time_reach_baseline_full"))

    created_files.append(
        write_caption(
            "F1T",
            "Time-window veracity results (Twitter15/16). ROC-AUC from 5-fold stratified CV using logistic regression for baseline, structure-only (Bundle C), and full features; shaded regions are ±1 SD. Dashed line marks chance level (0.5), and N per window is shown below ticks.",
        )
    )
    created_files.append(
        write_caption(
            "F2T",
            "Time-window reach results (Twitter15/16). R^2 from 5-fold CV using OLS for baseline, structure-only (Bundle C), and full features; shaded regions are ±1 SD. Target is log(1 + final_size), and N per window is shown below ticks.",
        )
    )

    # STEP 2 Full-only model family comparison
    step2_rows: List[Dict[str, Any]] = []
    for t in TIME_WINDOWS:
        df_t = time_df[time_df["window_value"] == t].sort_values(["dataset", "cascade_id"]).reset_index(drop=True)
        full_features = BASELINE_TIME + STRUCTURE_BUNDLE_C

        for m in ["logit", "rf", "xgb"]:
            auc_m, auc_s = eval_veracity(df_t, full_features, m, veracity_splits_by_t[t], xgb_available)
            step2_rows.append(
                {
                    "task": "veracity",
                    "window_minutes": t,
                    "model": m,
                    "metric": "auc",
                    "mean": auc_m,
                    "std": auc_s,
                    "N_used": len(df_t),
                }
            )

        for m in ["ols", "rfreg", "xgbreg"]:
            r2_m, r2_s, mae_m, mae_s = eval_reach(df_t, full_features, m, reach_splits_by_t[t], xgb_available)
            step2_rows.append(
                {
                    "task": "reach",
                    "window_minutes": t,
                    "model": m,
                    "metric": "r2",
                    "mean": r2_m,
                    "std": r2_s,
                    "N_used": len(df_t),
                }
            )
            step2_rows.append(
                {
                    "task": "reach",
                    "window_minutes": t,
                    "model": m,
                    "metric": "mae",
                    "mean": mae_m,
                    "std": mae_s,
                    "N_used": len(df_t),
                }
            )

    step2_df = pd.DataFrame(step2_rows)
    p_step2 = OUT_TABLES / "model_family_comparison_time_full.csv"
    step2_df.to_csv(p_step2, index=False)
    created_files.append(str(p_step2))

    fig3, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    colors = {"logit": "#4C78A8", "rf": "#F58518", "xgb": "#54A24B", "ols": "#4C78A8", "rfreg": "#F58518", "xgbreg": "#54A24B"}

    for m in ["logit", "rf", "xgb"]:
        d = step2_df[(step2_df["task"] == "veracity") & (step2_df["metric"] == "auc") & (step2_df["model"] == m)].sort_values("window_minutes")
        x = d["window_minutes"].to_numpy()
        y = d["mean"].to_numpy()
        s = d["std"].to_numpy()
        axes[0].plot(x, y, marker="o", linewidth=2, label=m, color=colors[m])
        axes[0].fill_between(x, y - s, y + s, alpha=0.2, color=colors[m])

    for m in ["ols", "rfreg", "xgbreg"]:
        d = step2_df[(step2_df["task"] == "reach") & (step2_df["metric"] == "r2") & (step2_df["model"] == m)].sort_values("window_minutes")
        x = d["window_minutes"].to_numpy()
        y = d["mean"].to_numpy()
        s = d["std"].to_numpy()
        axes[1].plot(x, y, marker="o", linewidth=2, label=m, color=colors[m])
        axes[1].fill_between(x, y - s, y + s, alpha=0.2, color=colors[m])

    axes[0].set_title("A: Veracity AUC (Full Features)")
    axes[1].set_title("B: Reach R^2 (Full Features)")
    axes[0].set_ylabel("AUC")
    axes[1].set_ylabel("R^2")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
        add_n_under_ticks(ax, TIME_WINDOWS, n_used_by_t)
        ax.set_xlabel("Time window (minutes)")
        ax.legend()
    fig3.tight_layout()
    created_files.extend(save_figure(fig3, "F3_model_family_full_only"))

    created_files.append(
        write_caption(
            "F3",
            "Full-feature model-family comparison over time windows. Panel A shows veracity ROC-AUC for logistic regression, random forest, and XGB (or fallback HistGradientBoosting); Panel B shows reach R^2 for OLS, random forest regressor, and XGB regressor (or fallback). Shaded regions are ±1 SD across fixed 5-fold splits.",
        )
    )

    # STEP 3 Delta gain by model (full - baseline)
    step3_rows: List[Dict[str, Any]] = []
    for t in TIME_WINDOWS:
        df_t = time_df[time_df["window_value"] == t].sort_values(["dataset", "cascade_id"]).reset_index(drop=True)

        for m in ["logit", "rf", "xgb"]:
            b_mean, _ = eval_veracity(df_t, BASELINE_TIME, m, veracity_splits_by_t[t], xgb_available)
            f_mean = float(
                step2_df[
                    (step2_df["task"] == "veracity")
                    & (step2_df["window_minutes"] == t)
                    & (step2_df["model"] == m)
                    & (step2_df["metric"] == "auc")
                ]["mean"].iloc[0]
            )
            step3_rows.append(
                {
                    "task": "veracity",
                    "window_minutes": t,
                    "model": m,
                    "metric": "auc",
                    "baseline_mean": b_mean,
                    "full_mean": f_mean,
                    "delta": f_mean - b_mean,
                }
            )

        for m in ["ols", "rfreg", "xgbreg"]:
            b_r2, _, _, _ = eval_reach(df_t, BASELINE_TIME, m, reach_splits_by_t[t], xgb_available)
            f_r2 = float(
                step2_df[
                    (step2_df["task"] == "reach")
                    & (step2_df["window_minutes"] == t)
                    & (step2_df["model"] == m)
                    & (step2_df["metric"] == "r2")
                ]["mean"].iloc[0]
            )
            step3_rows.append(
                {
                    "task": "reach",
                    "window_minutes": t,
                    "model": m,
                    "metric": "r2",
                    "baseline_mean": b_r2,
                    "full_mean": f_r2,
                    "delta": f_r2 - b_r2,
                }
            )

    step3_df = pd.DataFrame(step3_rows)
    p_step3 = OUT_TABLES / "delta_gain_by_model_time.csv"
    step3_df.to_csv(p_step3, index=False)
    created_files.append(str(p_step3))

    fig4, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    width = 0.22

    v_df = step3_df[step3_df["task"] == "veracity"].copy()
    models_v = ["logit", "rf", "xgb"]
    x = np.arange(len(TIME_WINDOWS))
    for i, m in enumerate(models_v):
        d = v_df[v_df["model"] == m].sort_values("window_minutes")
        axes[0].bar(x + (i - 1) * width, d["delta"].to_numpy(), width=width, label=m)
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(t) for t in TIME_WINDOWS])
    axes[0].set_title("A: ΔAUC (Full - Baseline)")
    axes[0].set_xlabel("Time window (minutes)")
    axes[0].set_ylabel("ΔAUC")
    axes[0].legend()

    r_df = step3_df[step3_df["task"] == "reach"].copy()
    models_r = ["ols", "rfreg", "xgbreg"]
    for i, m in enumerate(models_r):
        d = r_df[r_df["model"] == m].sort_values("window_minutes")
        axes[1].bar(x + (i - 1) * width, d["delta"].to_numpy(), width=width, label=m)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(t) for t in TIME_WINDOWS])
    axes[1].set_title("B: ΔR^2 (Full - Baseline)")
    axes[1].set_xlabel("Time window (minutes)")
    axes[1].set_ylabel("ΔR^2")
    axes[1].legend()

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
    fig4.tight_layout()
    created_files.extend(save_figure(fig4, "F4_delta_gain_by_model"))

    created_files.append(
        write_caption(
            "F4",
            "Performance gain from adding structure features (Bundle C): delta = metric(full) - metric(baseline) by model and time window. Panel A reports ΔAUC for veracity; Panel B reports ΔR^2 for reach. Horizontal line marks zero improvement.",
        )
    )

    # STEP 4A Permutation test at T=60 (full features)
    df60 = time_df[time_df["window_value"] == 60].sort_values(["dataset", "cascade_id"]).reset_index(drop=True)
    full_features = BASELINE_TIME + STRUCTURE_BUNDLE_C

    ver_model_perm = "logit"
    observed_auc, null_auc, p_auc = permutation_test(
        df60,
        full_features,
        task="veracity",
        model_name=ver_model_perm,
        splits=veracity_splits_by_t[60],
        xgb_available=xgb_available,
        n_permutations=200,
        rng=np.random.default_rng(SEED + 101),
    )

    observed_r2, null_r2, p_r2 = permutation_test(
        df60,
        full_features,
        task="reach",
        model_name="ols",
        splits=reach_splits_by_t[60],
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
                "observed_score": observed_r2,
                "null_mean": float(np.mean(null_r2)),
                "null_std": float(np.std(null_r2)),
                "p_value": p_r2,
                "n_permutations": 200,
            },
        ]
    )
    p_perm_summary = OUT_TABLES / "permutation_test_60min.csv"
    perm_summary.to_csv(p_perm_summary, index=False)
    created_files.append(str(p_perm_summary))

    perm_null_rows = []
    for i, s in enumerate(null_auc):
        perm_null_rows.append({"task": "veracity", "metric": "auc", "perm_idx": i, "null_score": float(s)})
    for i, s in enumerate(null_r2):
        perm_null_rows.append({"task": "reach", "metric": "r2", "perm_idx": i, "null_score": float(s)})
    perm_null_df = pd.DataFrame(perm_null_rows)
    p_perm_null = OUT_TABLES / "permutation_null_scores_60min.csv"
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
            "Permutation test at T=60 with full features. Histograms show null score distributions from 200 target shuffles (same CV splits). Vertical red lines mark observed scores; p-values are one-sided fractions of null >= observed.",
        )
    )

    # STEP 4B Learning curve (full features + baseline models)
    lc_rows: List[Dict[str, Any]] = []
    time_curve_df = make_time_feature_rows(cascades, T_CURVE, np.random.default_rng(SEED + 404))
    n_used_curve: Dict[int, int] = {}
    for t in T_CURVE:
        dft = time_curve_df[time_curve_df["window_value"] == t].sort_values(["dataset", "cascade_id"]).reset_index(drop=True)
        n_used_curve[t] = len(dft)
        v_splits = get_cv_splits(dft, "veracity")
        r_splits = get_cv_splits(dft, "reach")

        auc_m, auc_s = eval_veracity(dft, full_features, "logit", v_splits, xgb_available)
        lc_rows.append({"task": "veracity", "window_minutes": t, "model": "logit", "metric": "auc", "mean": auc_m, "std": auc_s, "N_used": len(dft)})

        r2_m, r2_s, _, _ = eval_reach(dft, full_features, "ols", r_splits, xgb_available)
        lc_rows.append({"task": "reach", "window_minutes": t, "model": "ols", "metric": "r2", "mean": r2_m, "std": r2_s, "N_used": len(dft)})

    lc_df = pd.DataFrame(lc_rows)
    p_lc = OUT_TABLES / "learning_curve_time.csv"
    lc_df.to_csv(p_lc, index=False)
    created_files.append(str(p_lc))

    fig6, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

    d_auc = lc_df[(lc_df["task"] == "veracity") & (lc_df["metric"] == "auc")].sort_values("window_minutes")
    x = d_auc["window_minutes"].to_numpy()
    y = d_auc["mean"].to_numpy()
    s = d_auc["std"].to_numpy()
    axes[0].plot(x, y, marker="o", linewidth=2, color="#4C78A8")
    axes[0].fill_between(x, y - s, y + s, alpha=0.2, color="#4C78A8")
    axes[0].set_title("A: Learning Curve (Veracity AUC)")
    axes[0].set_ylabel("AUC")

    d_r2 = lc_df[(lc_df["task"] == "reach") & (lc_df["metric"] == "r2")].sort_values("window_minutes")
    x2 = d_r2["window_minutes"].to_numpy()
    y2 = d_r2["mean"].to_numpy()
    s2 = d_r2["std"].to_numpy()
    axes[1].plot(x2, y2, marker="o", linewidth=2, color="#F58518")
    axes[1].fill_between(x2, y2 - s2, y2 + s2, alpha=0.2, color="#F58518")
    axes[1].set_title("B: Learning Curve (Reach R^2)")
    axes[1].set_ylabel("R^2")

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
        add_n_under_ticks(ax, T_CURVE, n_used_curve)
        ax.set_xlabel("Time window (minutes)")
    fig6.tight_layout()
    created_files.extend(save_figure(fig6, "F6_learning_curve_time"))

    created_files.append(
        write_caption(
            "F6",
            "Signal accumulation over time windows (full features). Veracity uses logistic regression (AUC), reach uses OLS (R^2), each with fixed 5-fold CV per window. Shaded regions denote ±1 SD and N is shown below ticks.",
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
            "full": BASELINE_K + STRUCTURE_BUNDLE_C,
        }

        for fs_name, features in feature_sets.items():
            auc_m, auc_s = eval_veracity(dfk, features, "logit", v_splits, xgb_available)
            step5_rows.append(
                {
                    "task": "veracity",
                    "window_type": "k",
                    "window_value": k,
                    "feature_set": fs_name,
                    "model": "logit",
                    "metric": "auc",
                    "mean": auc_m,
                    "std": auc_s,
                    "N_used": len(dfk),
                }
            )

            r2_m, r2_s, mae_m, mae_s = eval_reach(dfk, features, "ols", r_splits, xgb_available)
            step5_rows.append(
                {
                    "task": "reach",
                    "window_type": "k",
                    "window_value": k,
                    "feature_set": fs_name,
                    "model": "OLS",
                    "metric": "r2",
                    "mean": r2_m,
                    "std": r2_s,
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
                    "std": mae_s,
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
        std_col="std",
        lines=["baseline", "structure_only", "full"],
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
        std_col="std",
        lines=["baseline", "structure_only", "full"],
        title="Reach (R^2) vs K-window",
        ylabel="R^2",
        chance_line=None,
        n_map=n_used_by_k,
    )
    created_files.extend(save_figure(fig_a2, "A2K_k_reach_baseline_full"))

    created_files.append(
        write_caption(
            "A1K",
            "K-window veracity results (AUC) for baseline, structure-only (Bundle C), and full features using logistic regression with 5-fold stratified CV. Observed nodes are first K in BFS order from the full cascade tree.",
        )
    )
    created_files.append(
        write_caption(
            "A2K",
            "K-window reach results (R^2) for baseline, structure-only (Bundle C), and full features using OLS with 5-fold CV. Target is computed from full final size as log(1 + final_size).",
        )
    )

    # Tuned decision-tree check at T=60 using full features.
    df60_tree = time_df[time_df["window_value"] == 60].sort_values(["dataset", "cascade_id"]).reset_index(drop=True)
    tree_features = BASELINE_TIME + STRUCTURE_BUNDLE_C
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
                "window_minutes": 60,
                "model": "decision_tree_tuned",
                "metric": "auc",
                "mean": float(np.mean(tree_auc_scores)),
                "std": float(np.std(tree_auc_scores)),
                "N_used": len(df60_tree),
            },
            {
                "task": "reach",
                "window_minutes": 60,
                "model": "decision_tree_tuned",
                "metric": "r2",
                "mean": float(np.mean(tree_r2_scores)),
                "std": float(np.std(tree_r2_scores)),
                "N_used": len(df60_tree),
            },
            {
                "task": "reach",
                "window_minutes": 60,
                "model": "decision_tree_tuned",
                "metric": "mae",
                "mean": float(np.mean(tree_mae_scores)),
                "std": float(np.std(tree_mae_scores)),
                "N_used": len(df60_tree),
            },
        ]
    )
    p_tree_eval = OUT_TABLES / "tree_tuned_60min.csv"
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
        "structure_bundle": "Bundle C",
        "structure_features": STRUCTURE_BUNDLE_C,
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


if __name__ == "__main__":
    main()
