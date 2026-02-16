#!/usr/bin/env python3
import ast
import json
import math
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, f1_score, r2_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Configuration
# -----------------------------
SEED = 42
TIME_WINDOWS = [30, 60, 180]  # minutes
K_WINDOWS = [10, 20, 50]
MIN_NODES_FOR_TIME_COVERAGE = 2
VIRALITY_APPROX_THRESHOLD = 250
VIRALITY_SAMPLE_SIZE = 200

EXPECTED_LABELS = {"true", "false", "unverified", "non-rumor"}

ROOT = Path(".")
DATA_ROOT = ROOT / "rumor_detection_acl2017"
TWITTER15 = DATA_ROOT / "twitter15"
TWITTER16 = DATA_ROOT / "twitter16"
RUMOUREVAL_TRAIN = ROOT / "rumoureval2019" / "training" / "rumoureval-2019-training-data" / "twitter-english"

OUT_ROOT = ROOT / "thesis_outputs"
OUT_DATA = OUT_ROOT / "data"
OUT_FIG = OUT_ROOT / "figures"
OUT_TAB = OUT_ROOT / "tables"
OUT_CAP = OUT_ROOT / "captions"
OUT_LOG = OUT_ROOT / "logs"


def ensure_output_dirs() -> None:
    for p in [OUT_ROOT, OUT_DATA, OUT_FIG, OUT_TAB, OUT_CAP, OUT_LOG]:
        p.mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 120,
        }
    )


def fail_if_missing_paths() -> None:
    required = [
        TWITTER15 / "label.txt",
        TWITTER16 / "label.txt",
        TWITTER15 / "tree",
        TWITTER16 / "tree",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Required path missing: {p}")


@dataclass
class Endpoint:
    uid: str
    tweet_id: str
    delay_min: float


@dataclass
class ParseResult:
    ok: bool
    dataset: str
    cascade_id: str
    label: str
    error: str = ""
    warnings: List[str] = None
    final_size: int = 0
    final_depth: int = 0
    reach_target: float = np.nan
    features_time: List[Dict[str, Any]] = None
    features_k: List[Dict[str, Any]] = None


# -----------------------------
# Parsing + Validation
# -----------------------------

def parse_tree_line(line: str) -> Tuple[Endpoint, Endpoint]:
    """Parse one tree line of format: ['uid','tweet','delay']->['uid','tweet','delay']"""
    if "->" not in line:
        raise ValueError("Missing '->' in tree line")
    left, right = line.split("->", 1)

    try:
        left_obj = ast.literal_eval(left.strip())
        right_obj = ast.literal_eval(right.strip())
    except Exception as exc:
        raise ValueError(f"ast parse failure: {exc}")

    if not (isinstance(left_obj, list) and len(left_obj) == 3):
        raise ValueError("Left endpoint must be list of len=3")
    if not (isinstance(right_obj, list) and len(right_obj) == 3):
        raise ValueError("Right endpoint must be list of len=3")

    try:
        parent = Endpoint(str(left_obj[0]), str(left_obj[1]), float(left_obj[2]))
        child = Endpoint(str(right_obj[0]), str(right_obj[1]), float(right_obj[2]))
    except Exception as exc:
        raise ValueError(f"Endpoint conversion error: {exc}")

    return parent, child


def run_parser_unit_tests() -> List[str]:
    logs = []
    # Sample 1
    s1 = "['ROOT', 'ROOT', '0.0']->['15754281', '724703995147751424', '0.0']"
    p1, c1 = parse_tree_line(s1)
    assert p1.tweet_id == "ROOT"
    assert c1.tweet_id == "724703995147751424"
    assert math.isclose(c1.delay_min, 0.0)

    # Sample 2
    s2 = "['15754281', '724703995147751424', '0.0']->['27264308', '724704170205278208', '0.7']"
    p2, c2 = parse_tree_line(s2)
    assert p2.uid == "15754281"
    assert c2.uid == "27264308"
    assert math.isclose(c2.delay_min, 0.7)

    # Sample 3 (twitter16 style repeated tweet id)
    s3 = "['428333', '553587303172833280', '0.0']->['454236505', '553587303172833280', '0.1']"
    p3, c3 = parse_tree_line(s3)
    assert p3.tweet_id == c3.tweet_id
    assert c3.delay_min > p3.delay_min

    logs.append("Parser unit tests passed: 3/3")
    return logs


def parse_label_file(path: Path, dataset: str) -> Dict[str, str]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or ":" not in s:
                continue
            label, cascade_id = s.split(":", 1)
            out[cascade_id.strip()] = label.strip().lower()
    return out


def node_key(ep: Endpoint) -> str:
    # Event-level node identity avoids collapse when tweet IDs repeat in retweets.
    return f"{ep.uid}|{ep.tweet_id}|{ep.delay_min:.6f}"


def is_acyclic_directed(adj: Dict[str, List[str]], nodes: List[str]) -> bool:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes}

    def dfs(u: str) -> bool:
        color[u] = GRAY
        for v in adj.get(u, []):
            if color[v] == GRAY:
                return False
            if color[v] == WHITE and not dfs(v):
                return False
        color[u] = BLACK
        return True

    for n in nodes:
        if color[n] == WHITE:
            if not dfs(n):
                return False
    return True


def bfs_from_root(adj: Dict[str, List[str]], root: str) -> Tuple[List[str], Dict[str, int]]:
    order = []
    depth = {}
    q = deque([root])
    depth[root] = 0
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj.get(u, []):
            if v not in depth:
                depth[v] = depth[u] + 1
                q.append(v)
    return order, depth


def bfs_tree_projection(adj: Dict[str, List[str]], root: str) -> Dict[str, List[str]]:
    """Project possibly cyclic directed graph into a rooted BFS tree."""
    tree_adj = defaultdict(list)
    visited = set([root])
    q = deque([root])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                tree_adj[u].append(v)
                q.append(v)
    return tree_adj


def entropy_of_outdegree(outdeg: Dict[str, int], nodes: List[str]) -> float:
    vals = [outdeg.get(n, 0) for n in nodes]
    counts = Counter(vals)
    total = len(vals)
    if total == 0:
        return np.nan
    probs = [c / total for c in counts.values()]
    return float(-sum(p * math.log2(p) for p in probs if p > 0))


def structural_virality_proxy(
    undirected_adj: Dict[str, List[str]], nodes: List[str], rng: np.random.Generator
) -> float:
    n = len(nodes)
    if n <= 1:
        return 0.0

    work_nodes = nodes
    if n > VIRALITY_APPROX_THRESHOLD:
        work_nodes = list(rng.choice(nodes, size=min(VIRALITY_SAMPLE_SIZE, n), replace=False))

    work_set = set(work_nodes)

    def bfs_dist(src: str) -> Dict[str, int]:
        d = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in undirected_adj.get(u, []):
                if v in work_set and v not in d:
                    d[v] = d[u] + 1
                    q.append(v)
        return d

    dists = []
    for i, u in enumerate(work_nodes):
        du = bfs_dist(u)
        for v in work_nodes[i + 1 :]:
            if v in du:
                dists.append(du[v])

    if not dists:
        return np.nan
    return float(np.mean(dists))


def compute_window_features(
    observed_nodes: List[str],
    observed_adj: Dict[str, List[str]],
    root_key: str,
    delays_map: Dict[str, float],
    window_minutes: float,
    is_time_window: bool,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    if root_key not in observed_nodes:
        observed_nodes = [root_key] + observed_nodes
    observed_nodes = list(dict.fromkeys(observed_nodes))

    bfs_nodes, depths = bfs_from_root(observed_adj, root_key)
    node_set = set(bfs_nodes)

    outdeg = {n: 0 for n in bfs_nodes}
    for u in bfs_nodes:
        for v in observed_adj.get(u, []):
            if v in node_set:
                outdeg[u] += 1

    n_nodes = len(bfs_nodes)
    early_depth = max(depths.values()) if depths else 0
    avg_root_distance = float(np.mean(list(depths.values()))) if depths else 0.0

    width_counts = Counter(depths.values()) if depths else Counter()
    early_width = max(width_counts.values()) if width_counts else 1

    internal = [outdeg[n] for n in bfs_nodes if outdeg[n] > 0]
    branching = float(np.mean(internal)) if internal else 0.0
    internal_node_share = float(len(internal) / n_nodes) if n_nodes else np.nan

    leaf_fraction = float(sum(1 for n in bfs_nodes if outdeg[n] == 0) / n_nodes) if n_nodes else np.nan
    outdeg_entropy = entropy_of_outdegree(outdeg, bfs_nodes)

    undirected_adj = defaultdict(list)
    for u in bfs_nodes:
        for v in observed_adj.get(u, []):
            if v in node_set:
                undirected_adj[u].append(v)
                undirected_adj[v].append(u)

    virality = structural_virality_proxy(undirected_adj, bfs_nodes, rng)

    feats = {
        "early_n_nodes": n_nodes,
        "early_depth": early_depth,
        "avg_root_distance": avg_root_distance,
        "early_width": early_width,
        "branching_factor": branching,
        "internal_node_share": internal_node_share,
        "leaf_fraction": leaf_fraction,
        "outdegree_entropy": outdeg_entropy,
        "structural_virality_proxy": virality,
    }

    if is_time_window:
        feats["early_growth_rate"] = n_nodes / window_minutes if window_minutes > 0 else np.nan
        # time-to-k based on observed chronological nodes within T
        observed_delays = sorted(delays_map[n] for n in bfs_nodes)
        feats["time_to_10"] = observed_delays[9] if len(observed_delays) >= 10 else np.nan
        feats["time_to_20"] = observed_delays[19] if len(observed_delays) >= 20 else np.nan

    return feats


def parse_cascade(
    dataset: str,
    cascade_id: str,
    label: str,
    tree_path: Path,
    rng: np.random.Generator,
) -> ParseResult:
    warnings = []
    try:
        lines = [x.strip() for x in tree_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    except Exception as exc:
        return ParseResult(False, dataset, cascade_id, label, f"File read error: {exc}", warnings=warnings)

    if not lines:
        return ParseResult(False, dataset, cascade_id, label, "Empty tree file", warnings=warnings)

    all_nodes: Dict[str, Endpoint] = {}
    adj = defaultdict(list)
    incoming = defaultdict(int)
    root_candidates = []

    for li, line in enumerate(lines, start=1):
        try:
            parent, child = parse_tree_line(line)
        except Exception as exc:
            return ParseResult(False, dataset, cascade_id, label, f"Line {li} parse error: {exc}", warnings=warnings)

        if parent.delay_min < 0 or child.delay_min < 0:
            return ParseResult(False, dataset, cascade_id, label, f"Negative delay at line {li}", warnings=warnings)

        if parent.tweet_id == "ROOT" and parent.uid == "ROOT":
            root_candidates.append(child)
            pkey = "__ROOT_SENTINEL__"
        else:
            pkey = node_key(parent)
            all_nodes[pkey] = parent

        ckey = node_key(child)
        all_nodes[ckey] = child

        adj[pkey].append(ckey)
        incoming[ckey] += 1

    if not root_candidates:
        return ParseResult(False, dataset, cascade_id, label, "No ROOT->source edge found", warnings=warnings)

    root_ep = root_candidates[0]
    if root_ep.tweet_id != cascade_id:
        return ParseResult(False, dataset, cascade_id, label, f"Root tweet_id mismatch: {root_ep.tweet_id} != {cascade_id}", warnings=warnings)
    root_key = node_key(root_ep)

    # Remove sentinel from graph, keep edges from root key.
    sentinel_children = adj.get("__ROOT_SENTINEL__", [])
    if root_key not in sentinel_children:
        warnings.append("First root candidate not in sentinel children list")
    del adj["__ROOT_SENTINEL__"]

    # Check delays non-negative globally
    if any(ep.delay_min < 0 for ep in all_nodes.values()):
        return ParseResult(False, dataset, cascade_id, label, "Negative delay found", warnings=warnings)

    node_list = list(all_nodes.keys())

    # Acyclicity check on original graph. If cyclic, project to BFS tree and keep cascade.
    is_acyclic = is_acyclic_directed(adj, node_list)
    work_adj = adj
    if not is_acyclic:
        warnings.append("Original graph has cycles; projected to rooted BFS tree")
        work_adj = bfs_tree_projection(adj, root_key)

    # Reachability and outcomes
    bfs_nodes, depths = bfs_from_root(work_adj, root_key)
    if root_key not in bfs_nodes:
        return ParseResult(False, dataset, cascade_id, label, "Root not reachable in BFS", warnings=warnings)

    reachable = set(bfs_nodes)
    all_set = set(node_list)
    if reachable != all_set:
        warnings.append(f"Disconnected nodes excluded from outcomes: {len(all_set - reachable)}")

    final_size = len(reachable)
    final_depth = max(depths.values()) if depths else 0
    reach_target = float(np.log1p(final_size))

    # Build delay map and edge list on reachable nodes only
    delays_map = {k: all_nodes[k].delay_min for k in reachable}
    sorted_by_delay = sorted(reachable, key=lambda n: delays_map[n])

    # time windows
    features_time = []
    for T in TIME_WINDOWS:
        observed_nodes = [n for n in reachable if delays_map[n] <= T]
        if root_key not in observed_nodes:
            observed_nodes.append(root_key)
        observed_set = set(observed_nodes)

        obs_adj = defaultdict(list)
        for u in observed_nodes:
            for v in work_adj.get(u, []):
                if v in observed_set:
                    obs_adj[u].append(v)

        feats = compute_window_features(
            observed_nodes=observed_nodes,
            observed_adj=obs_adj,
            root_key=root_key,
            delays_map=delays_map,
            window_minutes=T,
            is_time_window=True,
            rng=rng,
        )

        features_time.append(
            {
                "dataset": dataset,
                "cascade_id": cascade_id,
                "label": label,
                "y_false": 1 if label == "false" else 0,
                "final_size": final_size,
                "final_depth": final_depth,
                "reach_target": reach_target,
                "window_type": "time",
                "window_value": T,
                **feats,
            }
        )

    # k windows (BFS order on full tree)
    full_bfs = bfs_nodes
    features_k = []
    for K in K_WINDOWS:
        reached = len(full_bfs) >= K
        observed_nodes = full_bfs[:K] if reached else full_bfs[:]
        obs_set = set(observed_nodes)

        obs_adj = defaultdict(list)
        for u in observed_nodes:
            for v in work_adj.get(u, []):
                if v in obs_set:
                    obs_adj[u].append(v)

        feats = compute_window_features(
            observed_nodes=observed_nodes,
            observed_adj=obs_adj,
            root_key=root_key,
            delays_map=delays_map,
            window_minutes=float(K),
            is_time_window=False,
            rng=rng,
        )

        features_k.append(
            {
                "dataset": dataset,
                "cascade_id": cascade_id,
                "label": label,
                "y_false": 1 if label == "false" else 0,
                "final_size": final_size,
                "final_depth": final_depth,
                "reach_target": reach_target,
                "window_type": "k",
                "window_value": K,
                "not_reached_k": (not reached),
                **feats,
            }
        )

    return ParseResult(
        ok=True,
        dataset=dataset,
        cascade_id=cascade_id,
        label=label,
        warnings=warnings,
        final_size=final_size,
        final_depth=final_depth,
        reach_target=reach_target,
        features_time=features_time,
        features_k=features_k,
    )


# -----------------------------
# Modeling
# -----------------------------

def evaluate_classification_binary(df: pd.DataFrame, feature_cols: List[str], y_col: str) -> Tuple[float, float, int]:
    work = df.copy()
    X = work[feature_cols].values
    y = work[y_col].astype(int).values
    n = len(work)
    if n < 20 or len(np.unique(y)) < 2:
        return np.nan, np.nan, n

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []

    for train_idx, test_idx in skf.split(X, y):
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=3000,
                        random_state=SEED,
                    ),
                ),
            ]
        )
        model.fit(X[train_idx], y[train_idx])
        prob = model.predict_proba(X[test_idx])[:, 1]
        scores.append(roc_auc_score(y[test_idx], prob))

    return float(np.mean(scores)), float(np.std(scores)), n


def evaluate_classification_multiclass(df: pd.DataFrame, feature_cols: List[str], y_col: str) -> Tuple[float, float, int]:
    work = df.copy()
    X = work[feature_cols].values
    y = work[y_col].values
    n = len(work)
    if n < 20 or len(np.unique(y)) < 3:
        return np.nan, np.nan, n

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []

    for train_idx, test_idx in skf.split(X, y):
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=4000,
                        random_state=SEED,
                    ),
                ),
            ]
        )
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        scores.append(f1_score(y[test_idx], pred, average="macro"))

    return float(np.mean(scores)), float(np.std(scores)), n


def evaluate_regression(df: pd.DataFrame, feature_cols: List[str], y_col: str) -> Tuple[float, float, float, float, int]:
    work = df.copy()
    X = work[feature_cols].values
    y = work[y_col].values
    n = len(work)
    if n < 20:
        return np.nan, np.nan, np.nan, np.nan, n

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    r2s = []
    maes = []

    for train_idx, test_idx in kf.split(X):
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("reg", LinearRegression()),
            ]
        )
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        r2s.append(r2_score(y[test_idx], pred))
        maes.append(mean_absolute_error(y[test_idx], pred))

    return float(np.mean(r2s)), float(np.std(r2s)), float(np.mean(maes)), float(np.std(maes)), n


# -----------------------------
# Figure helpers
# -----------------------------

def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / f"{stem}.pdf", bbox_inches="tight")


def write_caption(stem: str, text: str) -> None:
    (OUT_CAP / f"{stem}.txt").write_text(text, encoding="utf-8")


def line_plot_with_band(
    df: pd.DataFrame,
    metric_col_mean: str,
    metric_col_std: str,
    title: str,
    ylabel: str,
    stem: str,
    n_used_map: Dict[int, int],
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    colors = {
        "baseline": "#4C78A8",
        "structure_only": "#F58518",
        "full": "#54A24B",
    }
    labels = {
        "baseline": "Baseline",
        "structure_only": "Structure-only",
        "full": "Full",
    }

    xvals = sorted(df["window_value"].unique())
    for feat_set in ["baseline", "structure_only", "full"]:
        sub = df[df["feature_set"] == feat_set].sort_values("window_value")
        y = sub[metric_col_mean].values
        s = sub[metric_col_std].values
        x = sub["window_value"].values
        ax.plot(x, y, marker="o", linewidth=2, color=colors[feat_set], label=labels[feat_set])
        ax.fill_between(x, y - s, y + s, color=colors[feat_set], alpha=0.15)

    full_sub = df[df["feature_set"] == "full"]
    benchmark = float(full_sub[metric_col_mean].mean())
    ax.axhline(benchmark, linestyle="--", color="#666666", linewidth=1.3)
    ax.text(
        xvals[-1] + (xvals[-1] - xvals[0]) * 0.06,
        benchmark,
        f"Overall Full={benchmark:.3f}",
        va="center",
        ha="left",
        fontsize=10,
        color="#444444",
    )

    ax.set_title(title)
    ax.set_xlabel("Time window (minutes)")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")

    xticklabels = [f"{x}\nN={n_used_map.get(int(x), 0)}" for x in xvals]
    ax.set_xticks(xvals)
    ax.set_xticklabels(xticklabels)

    save_fig(fig, stem)
    plt.close(fig)


def make_fig3_geometry_by_label(df_time: pd.DataFrame) -> None:
    sub = df_time[df_time["window_value"] == 60].copy()
    order = ["false", "true", "unverified", "non-rumor"]
    sub = sub[sub["label"].isin(order)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    f1 = "leaf_fraction"
    f2 = "structural_virality_proxy"

    data1 = [sub[sub["label"] == lab][f1].dropna().values for lab in order]
    data2 = [sub[sub["label"] == lab][f2].dropna().values for lab in order]

    axes[0].boxplot(data1, tick_labels=order, showfliers=False)
    axes[0].set_title("Leaf fraction at 60 min")
    axes[0].set_ylabel("Leaf fraction")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].boxplot(data2, tick_labels=order, showfliers=False)
    axes[1].set_title("Structural virality proxy at 60 min")
    axes[1].set_ylabel("Mean pairwise distance")
    axes[1].tick_params(axis="x", rotation=20)

    false_med_leaf = float(np.median(sub[sub["label"] == "false"][f1].dropna().values))
    not_false_med_leaf = float(np.median(sub[sub["label"] != "false"][f1].dropna().values))
    false_med_vir = float(np.median(sub[sub["label"] == "false"][f2].dropna().values))
    not_false_med_vir = float(np.median(sub[sub["label"] != "false"][f2].dropna().values))

    fig.suptitle("Early Geometry by Veracity Label (T=60 min)")
    fig.text(
        0.5,
        -0.01,
        f"Median diff (false - not-false): leaf_fraction={false_med_leaf - not_false_med_leaf:.3f}, "
        f"virality={false_med_vir - not_false_med_vir:.3f}",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_fig(fig, "F3_geometry_by_label_60min")
    plt.close(fig)


def make_fig4_decomposition(results_time: pd.DataFrame) -> None:
    ver = results_time[(results_time["task"] == "veracity_binary") & (results_time["metric"] == "auc")]
    reg = results_time[(results_time["task"] == "reach") & (results_time["metric"] == "r2")]

    windows = sorted(ver["window_value"].unique())
    feat_sets = ["baseline", "structure_only", "full"]
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    width = 0.22
    x = np.arange(len(windows))

    for i, fs in enumerate(feat_sets):
        v = ver[ver["feature_set"] == fs].sort_values("window_value")
        r = reg[reg["feature_set"] == fs].sort_values("window_value")
        axes[0].bar(x + (i - 1) * width, v["metric_mean"].values, width, label=fs, color=colors[i], alpha=0.9)
        axes[1].bar(x + (i - 1) * width, r["metric_mean"].values, width, label=fs, color=colors[i], alpha=0.9)

    axes[0].set_title("Panel A: Veracity AUC")
    axes[1].set_title("Panel B: Reach R²")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([str(w) for w in windows])
        ax.set_xlabel("Time window (minutes)")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("AUC")
    axes[1].set_ylabel("R²")
    axes[0].legend(loc="best")

    fig.suptitle("Contribution Decomposition: Volume vs Geometry")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    save_fig(fig, "F4_contribution_decomposition")
    plt.close(fig)


def make_fig5_external_coverage(tw_final_sizes: np.ndarray) -> bool:
    if not RUMOUREVAL_TRAIN.exists():
        return False

    rum_sizes = []
    # Limit scope strictly to RumourEval training twitter-english structure files.
    for event_dir in sorted(RUMOUREVAL_TRAIN.iterdir()):
        if not event_dir.is_dir():
            continue
        for thread_dir in sorted(event_dir.iterdir()):
            if not thread_dir.is_dir():
                continue
            sp = thread_dir / "structure.json"
            if not sp.exists():
                continue
            try:
                data = json.loads(sp.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or not data:
                    continue
                root = next(iter(data.keys()))

                edges = []

                def walk(node, parent=None):
                    if isinstance(node, dict):
                        for k, v in node.items():
                            if parent is not None:
                                edges.append((parent, k))
                            walk(v, k)
                    elif isinstance(node, list):
                        for item in node:
                            walk(item, parent)

                walk(data)
                adj = defaultdict(list)
                for u, v in edges:
                    adj[u].append(v)
                bfs_nodes, _ = bfs_from_root(adj, root)
                rum_sizes.append(len(set(bfs_nodes)))
            except Exception:
                continue

    if not rum_sizes:
        return False

    tw_cov = [float(np.mean(tw_final_sizes >= k)) for k in K_WINDOWS]
    re_cov = [float(np.mean(np.array(rum_sizes) >= k)) for k in K_WINDOWS]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(K_WINDOWS, tw_cov, marker="o", linewidth=2, label="Twitter15/16", color="#4C78A8")
    ax.plot(K_WINDOWS, re_cov, marker="o", linewidth=2, label="RumourEval2019", color="#F58518")
    ax.set_title("External Feasibility: Coverage by k")
    ax.set_xlabel("k-window")
    ax.set_ylabel("Coverage (fraction with final_size >= k)")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")

    # inset values
    table_txt = "k | TW15/16 | RE\n" + "\n".join(
        [f"{k} | {tw_cov[i]:.3f} | {re_cov[i]:.3f}" for i, k in enumerate(K_WINDOWS)]
    )
    ax.text(0.98, 0.05, table_txt, transform=ax.transAxes, va="bottom", ha="right", fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="#cccccc"))

    save_fig(fig, "F5_external_coverage")
    plt.close(fig)
    return True


# -----------------------------
# Main execution
# -----------------------------

def main() -> None:
    start = time.time()
    rng = np.random.default_rng(SEED)

    ensure_output_dirs()
    set_plot_style()
    fail_if_missing_paths()

    run_log_lines = []
    audit_lines = []
    run_log_lines.append(f"Run started: {datetime.now().isoformat(timespec='seconds')}")

    parser_test_logs = run_parser_unit_tests()
    audit_lines.extend(parser_test_logs)

    datasets = [
        ("twitter15", TWITTER15),
        ("twitter16", TWITTER16),
    ]

    labels_all = {}
    for dname, droot in datasets:
        labels_all[dname] = parse_label_file(droot / "label.txt", dname)

    label_rows = []
    for dname, mp in labels_all.items():
        for cid, lab in mp.items():
            label_rows.append({"dataset": dname, "cascade_id": cid, "label": lab})

    label_df = pd.DataFrame(label_rows)
    invalid_labels = sorted(set(label_df[~label_df["label"].isin(EXPECTED_LABELS)]["label"].unique()))
    if invalid_labels:
        audit_lines.append(f"Unexpected labels found: {invalid_labels}")

    parse_errors = []
    parse_warnings = []
    ok_records = []
    time_rows = []
    k_rows = []

    for dname, droot in datasets:
        tree_dir = droot / "tree"
        label_map = labels_all[dname]

        for tree_file in sorted(tree_dir.glob("*.txt")):
            cid = tree_file.stem
            lab = label_map.get(cid)
            if lab is None:
                parse_errors.append((dname, cid, "Missing label"))
                continue

            res = parse_cascade(dname, cid, lab, tree_file, rng)
            if not res.ok:
                parse_errors.append((dname, cid, res.error))
                continue

            ok_records.append(
                {
                    "dataset": dname,
                    "cascade_id": cid,
                    "label": lab,
                    "final_size": res.final_size,
                    "final_depth": res.final_depth,
                    "reach_target": res.reach_target,
                }
            )
            for w in res.warnings or []:
                parse_warnings.append((dname, cid, w))
            time_rows.extend(res.features_time)
            k_rows.extend(res.features_k)

    casc_df = pd.DataFrame(ok_records)
    time_df = pd.DataFrame(time_rows)
    k_df = pd.DataFrame(k_rows)

    # Save feature tables
    time_path = OUT_DATA / "twitter1516_features_time.csv"
    k_path = OUT_DATA / "twitter1516_features_k.csv"
    time_df.to_csv(time_path, index=False)
    k_df.to_csv(k_path, index=False)

    # Dataset summary table
    summary_rows = []
    for dname, _ in datasets:
        labels = labels_all[dname]
        total_labeled = len(labels)
        ok_n = int((casc_df["dataset"] == dname).sum()) if not casc_df.empty else 0
        err_n = sum(1 for d, _, _ in parse_errors if d == dname)

        row = {
            "dataset": dname,
            "total_labeled": total_labeled,
            "parsed_ok": ok_n,
            "excluded": err_n,
            "label_true": sum(1 for x in labels.values() if x == "true"),
            "label_false": sum(1 for x in labels.values() if x == "false"),
            "label_unverified": sum(1 for x in labels.values() if x == "unverified"),
            "label_non_rumor": sum(1 for x in labels.values() if x == "non-rumor"),
        }
        summary_rows.append(row)

    dataset_summary_df = pd.DataFrame(summary_rows)
    dataset_summary_df.to_csv(OUT_TAB / "dataset_summary.csv", index=False)

    # Audit log
    audit_lines.append(f"Total labeled cascades: {len(label_df)}")
    audit_lines.append(f"Parsed cascades (ok): {len(casc_df)}")
    audit_lines.append(f"Parsing/validation exclusions: {len(parse_errors)}")
    audit_lines.append(f"Warnings count: {len(parse_warnings)}")
    cyc_warn_n = sum(1 for _, _, w in parse_warnings if "cycles" in w.lower())
    audit_lines.append(f"Cascades with original cycles (kept via BFS-tree projection): {cyc_warn_n}")

    if parse_errors:
        audit_lines.append("\nExcluded cascades:")
        for d, cid, err in parse_errors[:2000]:
            audit_lines.append(f"- {d}:{cid} => {err}")

    if parse_warnings:
        audit_lines.append("\nCascade warnings:")
        for d, cid, warn in parse_warnings[:2000]:
            audit_lines.append(f"- {d}:{cid} => {warn}")

    (OUT_LOG / "data_audit.txt").write_text("\n".join(audit_lines), encoding="utf-8")

    # Coverage table
    cov_rows = []
    for T in TIME_WINDOWS:
        sub = time_df[time_df["window_value"] == T]
        coverage = float((sub["early_n_nodes"] >= MIN_NODES_FOR_TIME_COVERAGE).mean()) if len(sub) else np.nan
        cov_rows.append(
            {
                "window_type": "time",
                "window_value": T,
                "coverage": coverage,
                "n_total": len(sub),
            }
        )

    for K in K_WINDOWS:
        coverage = float((casc_df["final_size"] >= K).mean()) if len(casc_df) else np.nan
        sub = k_df[k_df["window_value"] == K]
        cov_rows.append(
            {
                "window_type": "k",
                "window_value": K,
                "coverage": coverage,
                "n_total": len(sub),
            }
        )

    coverage_df = pd.DataFrame(cov_rows)

    # Modeling
    baseline_time = ["early_n_nodes", "early_growth_rate", "time_to_10", "time_to_20"]
    structure_feats = [
        "early_depth",
        "early_width",
        "branching_factor",
        "leaf_fraction",
        "outdegree_entropy",
        "structural_virality_proxy",
    ]
    full_time = baseline_time + structure_feats

    baseline_k = ["early_n_nodes"]
    full_k = baseline_k + structure_feats

    results_time = []
    results_k = []

    def add_result(rows, window_type, window_value, task, model_family, feature_set, metric, mean, std, n):
        rows.append(
            {
                "window_type": window_type,
                "window_value": window_value,
                "task": task,
                "model_family": model_family,
                "feature_set": feature_set,
                "metric": metric,
                "metric_mean": mean,
                "metric_std": std,
                "N_used": n,
            }
        )

    # Time windows
    for T in TIME_WINDOWS:
        sub = time_df[time_df["window_value"] == T].copy()
        sub = sub[sub["early_n_nodes"] >= MIN_NODES_FOR_TIME_COVERAGE]

        # Veracity binary
        for feat_set, cols in [
            ("baseline", baseline_time),
            ("structure_only", structure_feats),
            ("full", full_time),
        ]:
            mean, std, n = evaluate_classification_binary(sub, cols, "y_false")
            add_result(results_time, "time", T, "veracity_binary", "logistic", feat_set, "auc", mean, std, n)

        # Veracity multiclass
        for feat_set, cols in [
            ("baseline", baseline_time),
            ("structure_only", structure_feats),
            ("full", full_time),
        ]:
            mean, std, n = evaluate_classification_multiclass(sub, cols, "label")
            add_result(results_time, "time", T, "veracity_multiclass", "multinomial_logistic", feat_set, "macro_f1", mean, std, n)

        # Reach regression
        for feat_set, cols in [
            ("baseline", baseline_time),
            ("structure_only", structure_feats),
            ("full", full_time),
        ]:
            r2m, r2s, maem, maes, n = evaluate_regression(sub, cols, "reach_target")
            add_result(results_time, "time", T, "reach", "linear_regression", feat_set, "r2", r2m, r2s, n)
            add_result(results_time, "time", T, "reach", "linear_regression", feat_set, "mae", maem, maes, n)

    # K windows
    for K in K_WINDOWS:
        sub = k_df[k_df["window_value"] == K].copy()

        for feat_set, cols in [
            ("baseline", baseline_k),
            ("structure_only", structure_feats),
            ("full", full_k),
        ]:
            mean, std, n = evaluate_classification_binary(sub, cols, "y_false")
            add_result(results_k, "k", K, "veracity_binary", "logistic", feat_set, "auc", mean, std, n)

        for feat_set, cols in [
            ("baseline", baseline_k),
            ("structure_only", structure_feats),
            ("full", full_k),
        ]:
            mean, std, n = evaluate_classification_multiclass(sub, cols, "label")
            add_result(results_k, "k", K, "veracity_multiclass", "multinomial_logistic", feat_set, "macro_f1", mean, std, n)

        for feat_set, cols in [
            ("baseline", baseline_k),
            ("structure_only", structure_feats),
            ("full", full_k),
        ]:
            r2m, r2s, maem, maes, n = evaluate_regression(sub, cols, "reach_target")
            add_result(results_k, "k", K, "reach", "linear_regression", feat_set, "r2", r2m, r2s, n)
            add_result(results_k, "k", K, "reach", "linear_regression", feat_set, "mae", maem, maes, n)

    results_time_df = pd.DataFrame(results_time)
    results_k_df = pd.DataFrame(results_k)

    results_time_df.to_csv(OUT_TAB / "results_time.csv", index=False)
    results_k_df.to_csv(OUT_TAB / "results_k.csv", index=False)

    # Add N_used into coverage table
    cov_extra = []
    for T in TIME_WINDOWS:
        vsub = results_time_df[
            (results_time_df["window_type"] == "time")
            & (results_time_df["window_value"] == T)
            & (results_time_df["task"] == "veracity_binary")
            & (results_time_df["feature_set"] == "full")
            & (results_time_df["metric"] == "auc")
        ]
        rsub = results_time_df[
            (results_time_df["window_type"] == "time")
            & (results_time_df["window_value"] == T)
            & (results_time_df["task"] == "reach")
            & (results_time_df["feature_set"] == "full")
            & (results_time_df["metric"] == "r2")
        ]
        cov_extra.append(("time", T, int(vsub["N_used"].iloc[0]) if len(vsub) else 0, int(rsub["N_used"].iloc[0]) if len(rsub) else 0))

    for K in K_WINDOWS:
        vsub = results_k_df[
            (results_k_df["window_type"] == "k")
            & (results_k_df["window_value"] == K)
            & (results_k_df["task"] == "veracity_binary")
            & (results_k_df["feature_set"] == "full")
            & (results_k_df["metric"] == "auc")
        ]
        rsub = results_k_df[
            (results_k_df["window_type"] == "k")
            & (results_k_df["window_value"] == K)
            & (results_k_df["task"] == "reach")
            & (results_k_df["feature_set"] == "full")
            & (results_k_df["metric"] == "r2")
        ]
        cov_extra.append(("k", K, int(vsub["N_used"].iloc[0]) if len(vsub) else 0, int(rsub["N_used"].iloc[0]) if len(rsub) else 0))

    cov_extra_df = pd.DataFrame(cov_extra, columns=["window_type", "window_value", "N_used_veracity", "N_used_reach"])
    t2_df = coverage_df.merge(cov_extra_df, on=["window_type", "window_value"], how="left")
    t2_df.to_csv(OUT_TAB / "T2_coverage.csv", index=False)
    (OUT_TAB / "T2_coverage.tex").write_text(t2_df.to_latex(index=False, float_format=lambda x: f"{x:.3f}"), encoding="utf-8")

    # T1 best time-window results baseline vs full
    t1_rows = []
    for task, metric in [("veracity_binary", "auc"), ("reach", "r2")]:
        for feat in ["baseline", "full"]:
            sub = results_time_df[
                (results_time_df["task"] == task)
                & (results_time_df["metric"] == metric)
                & (results_time_df["feature_set"] == feat)
            ]
            if len(sub) == 0:
                continue
            idx = sub["metric_mean"].idxmax()
            best = sub.loc[idx]
            t1_rows.append(
                {
                    "task": task,
                    "feature_set": feat,
                    "best_window": int(best["window_value"]),
                    "metric": metric,
                    "metric_mean": float(best["metric_mean"]),
                    "metric_std": float(best["metric_std"]),
                    "N_used": int(best["N_used"]),
                }
            )
    t1_df = pd.DataFrame(t1_rows)
    t1_df.to_csv(OUT_TAB / "T1_best_results_time.csv", index=False)
    (OUT_TAB / "T1_best_results_time.tex").write_text(t1_df.to_latex(index=False, float_format=lambda x: f"{x:.3f}"), encoding="utf-8")

    # Figures 1 and 2
    f1_df = results_time_df[
        (results_time_df["task"] == "veracity_binary")
        & (results_time_df["metric"] == "auc")
        & (results_time_df["feature_set"].isin(["baseline", "structure_only", "full"]))
    ]
    n_map_f1 = {
        int(r["window_value"]): int(r["N_used"])
        for _, r in f1_df[f1_df["feature_set"] == "full"].iterrows()
    }
    line_plot_with_band(
        f1_df,
        metric_col_mean="metric_mean",
        metric_col_std="metric_std",
        title="F1: Time-window Timeliness vs Veracity Performance",
        ylabel="AUC (5-fold CV)",
        stem="F1_time_veracity_auc",
        n_used_map=n_map_f1,
    )

    f2_df = results_time_df[
        (results_time_df["task"] == "reach")
        & (results_time_df["metric"] == "r2")
        & (results_time_df["feature_set"].isin(["baseline", "structure_only", "full"]))
    ]
    n_map_f2 = {
        int(r["window_value"]): int(r["N_used"])
        for _, r in f2_df[f2_df["feature_set"] == "full"].iterrows()
    }
    line_plot_with_band(
        f2_df,
        metric_col_mean="metric_mean",
        metric_col_std="metric_std",
        title="F2: Time-window Timeliness vs Reach Performance",
        ylabel="R² (5-fold CV)",
        stem="F2_time_reach_r2",
        n_used_map=n_map_f2,
    )

    make_fig3_geometry_by_label(time_df)
    make_fig4_decomposition(results_time_df)

    made_f5 = make_fig5_external_coverage(casc_df["final_size"].values)

    # Captions
    write_caption(
        "F1_time_veracity_auc",
        "Figure F1 tests RQ(2): whether early geometry improves veracity prediction under early-warning time limits. "
        "Each point is mean AUC from 5-fold Stratified CV at T={30,60,180} minutes, with shaded +/-1 SD. "
        "Baseline uses early volume/timing only; Full adds early tree geometry (depth, width, branching, leaves, entropy, virality). "
        "The dashed line marks average Full performance across windows. "
        "N_used per window is shown on the x-axis and should be considered when interpreting uncertainty.",
    )

    write_caption(
        "F2_time_reach_r2",
        "Figure F2 tests RQ(1): whether early geometry improves eventual reach prediction (target log(1+final_size)). "
        "Points show mean R² from 5-fold CV with +/-1 SD shading across time windows. "
        "Comparing Baseline vs Structure-only vs Full isolates the incremental value of geometry beyond early volume/timing. "
        "The dashed benchmark reports the overall Full average across windows. "
        "Coverage and N_used constrain which early-warning settings are practically informative.",
    )

    write_caption(
        "F3_geometry_by_label_60min",
        "Figure F3 provides mechanistic evidence at T=60 minutes by comparing early geometry across veracity labels. "
        "The two panels show leaf_fraction and structural_virality_proxy distributions with median lines. "
        "Visible shifts in these geometry features between false and non-false cascades support the claim that shape differs early. "
        "This complements predictive metrics by showing interpretable structural differences rather than model output alone.",
    )

    write_caption(
        "F4_contribution_decomposition",
        "Figure F4 decomposes performance by feature family for each time window. "
        "Panel A reports veracity AUC; Panel B reports reach R². "
        "Baseline captures early volume/timing, Structure-only uses geometry only, and Full combines both. "
        "If Full consistently exceeds Baseline, geometry adds incremental predictive signal under early-warning constraints.",
    )

    if made_f5:
        write_caption(
            "F5_external_coverage",
            "Figure F5 compares k-window feasibility between Twitter15/16 and RumourEval2019. "
            "Coverage is the fraction of cascades reaching at least k nodes (k=10,20,50). "
            "This robustness check evaluates whether early-window constraints are dataset-specific. "
            "Lower coverage at larger k indicates practical limits for early detection and informs fair cross-dataset interpretation.",
        )

    # Config + run logs
    config = {
        "seed": SEED,
        "time_windows": TIME_WINDOWS,
        "k_windows": K_WINDOWS,
        "min_nodes_for_time_coverage": MIN_NODES_FOR_TIME_COVERAGE,
        "virality_approx_threshold": VIRALITY_APPROX_THRESHOLD,
        "virality_sample_size": VIRALITY_SAMPLE_SIZE,
        "models": {
            "veracity_binary": "LogisticRegression(class_weight='balanced', solver='liblinear')",
            "veracity_multiclass": "Multinomial LogisticRegression(lbfgs)",
            "reach": "LinearRegression",
        },
        "feature_sets": {
            "time_baseline": baseline_time,
            "k_baseline": baseline_k,
            "structure": structure_feats,
        },
        "paths": {
            "twitter15": str(TWITTER15),
            "twitter16": str(TWITTER16),
            "rumoureval_training": str(RUMOUREVAL_TRAIN),
        },
    }
    (OUT_LOG / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    runtime = time.time() - start
    run_log_lines.append(f"Runtime seconds: {runtime:.2f}")
    run_log_lines.append(f"Cascades parsed OK: {len(casc_df)}")
    run_log_lines.append(f"Exclusions: {len(parse_errors)}")

    coverage_warns = []
    for _, row in t2_df.iterrows():
        if row["coverage"] < 0.2:
            coverage_warns.append(f"Low coverage warning: {row['window_type']}={int(row['window_value'])}, coverage={row['coverage']:.3f}")
    run_log_lines.extend(coverage_warns)

    (OUT_LOG / "run_log.txt").write_text("\n".join(run_log_lines), encoding="utf-8")

    # Headline numbers
    auc60_base = results_time_df[
        (results_time_df["task"] == "veracity_binary")
        & (results_time_df["metric"] == "auc")
        & (results_time_df["window_value"] == 60)
        & (results_time_df["feature_set"] == "baseline")
    ]["metric_mean"].iloc[0]
    auc60_full = results_time_df[
        (results_time_df["task"] == "veracity_binary")
        & (results_time_df["metric"] == "auc")
        & (results_time_df["window_value"] == 60)
        & (results_time_df["feature_set"] == "full")
    ]["metric_mean"].iloc[0]

    r260_base = results_time_df[
        (results_time_df["task"] == "reach")
        & (results_time_df["metric"] == "r2")
        & (results_time_df["window_value"] == 60)
        & (results_time_df["feature_set"] == "baseline")
    ]["metric_mean"].iloc[0]
    r260_full = results_time_df[
        (results_time_df["task"] == "reach")
        & (results_time_df["metric"] == "r2")
        & (results_time_df["window_value"] == 60)
        & (results_time_df["feature_set"] == "full")
    ]["metric_mean"].iloc[0]

    # Print required summary
    created_files = sorted([str(p) for p in OUT_ROOT.rglob("*") if p.is_file()])
    print("Created files:")
    for p in created_files:
        print(p)

    print("\nHeadline numbers:")
    print(f"Full AUC @60min: {auc60_full:.4f}; delta vs baseline: {auc60_full - auc60_base:+.4f}")
    print(f"Full R2 @60min: {r260_full:.4f}; delta vs baseline: {r260_full - r260_base:+.4f}")

    if coverage_warns:
        print("\nCoverage warnings:")
        for w in coverage_warns:
            print(w)


if __name__ == "__main__":
    main()
