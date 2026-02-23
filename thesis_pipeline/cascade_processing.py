import ast
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import SAMPLE_VIRALITY_N


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
