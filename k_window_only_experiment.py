import os
import json
import re
import argparse
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    RepeatedStratifiedKFold,
    RepeatedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

# ---------------------------
# Config
# ---------------------------
RANDOM_STATE = 42
K_LIST = [10, 20, 50]

ROOT_DIR = "rumor_detection_acl2017"
DATASETS = {
    "twitter15": {
        "label_path": os.path.join(ROOT_DIR, "twitter15", "label.txt"),
        "tree_dir": os.path.join(ROOT_DIR, "twitter15", "tree"),
    },
    "twitter16": {
        "label_path": os.path.join(ROOT_DIR, "twitter16", "label.txt"),
        "tree_dir": os.path.join(ROOT_DIR, "twitter16", "tree"),
    },
}

OUT_DIR = "thesis_outputs"
TABLE_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(OUT_DIR, "figures")
CAP_DIR = os.path.join(OUT_DIR, "captions")
LOG_DIR = os.path.join(OUT_DIR, "logs")
for d in [OUT_DIR, TABLE_DIR, FIG_DIR, CAP_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------
# Parsing
# ---------------------------
LABEL_MAP = {
    "true": "true",
    "false": "false",
    "unverified": "unverified",
    "non-rumor": "non-rumor",
    "non-rumour": "non-rumor",
    "nonrumor": "non-rumor",
}

# Edge line example:
# ['ROOT', 'ROOT', '0.0']->['39364684', '265953285247209472', '0.0']
EDGE_RE = re.compile(
    r"\['[^']*',\s*'(?P<pid>[^']+)',\s*'(?P<pt>[^']+)'\]"
    r"\s*->\s*"
    r"\['[^']*',\s*'(?P<cid>[^']+)',\s*'(?P<ct>[^']+)'\]"
)


def parse_labels(label_path: str) -> dict:
    labels = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            label = None
            cid = None

            if ":" in line:
                left, right = line.split(":", 1)
                left_norm = LABEL_MAP.get(left.strip().lower())
                right_norm = LABEL_MAP.get(right.strip().lower())
                if left_norm is not None:
                    label = left_norm
                    cid = right.strip()
                elif right_norm is not None:
                    label = right_norm
                    cid = left.strip()

            if label is None:
                parts = re.split(r"\s+", line)
                for tok in parts:
                    mapped = LABEL_MAP.get(tok.lower())
                    if mapped is not None:
                        label = mapped
                        break
                if label is not None:
                    # Prefer long numeric-like token as cascade id
                    for tok in parts:
                        if tok.isdigit() and len(tok) >= 8:
                            cid = tok
                            break
                    if cid is None and parts:
                        cid = parts[0]

            if label is not None and cid is not None:
                labels[cid] = label

    return labels


def parse_tree_file(path: str):
    edges = []
    nodes = set()
    indeg = defaultdict(int)

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m = EDGE_RE.search(line)
            if m:
                p = m.group("pid")
                c = m.group("cid")
                ct = float(m.group("ct"))
                edges.append((p, c, ct))
                nodes.add(p)
                nodes.add(c)
                indeg[c] += 1
                indeg.setdefault(p, 0)
                continue

            # Fallback generic format(s)
            if "->" in line:
                left, *rest = re.split(r"\s+", line)
                if not rest:
                    continue
                t = float(rest[0])
                p, c = left.split("->", 1)
            else:
                parts = re.split(r"\s+", line)
                if len(parts) < 3:
                    continue
                p, c, t = parts[0], parts[1], float(parts[2])

            edges.append((p, c, t))
            nodes.add(p)
            nodes.add(c)
            indeg[c] += 1
            indeg.setdefault(p, 0)

    if not nodes:
        raise ValueError("Empty tree")

    roots = [n for n in nodes if indeg.get(n, 0) == 0]
    root = roots[0] if roots else next(iter(nodes))
    return edges, nodes, root


# ---------------------------
# K-window extraction (BFS)
# ---------------------------
def bfs_first_k_nodes(edges, root, k):
    children = defaultdict(list)
    for p, c, _ in edges:
        children[p].append(c)
    for p in children:
        children[p] = sorted(children[p])

    q = deque([root])
    visited = []
    seen = {root}

    while q and len(visited) < k:
        u = q.popleft()
        visited.append(u)
        for v in children.get(u, []):
            if v not in seen:
                seen.add(v)
                q.append(v)

    return visited


def induced_subgraph(edges, kept_nodes):
    kept = set(kept_nodes)
    return [(p, c, t) for (p, c, t) in edges if p in kept and c in kept]


def node_depths(sub_edges, root):
    children = defaultdict(list)
    for p, c, _ in sub_edges:
        children[p].append(c)
    for p in children:
        children[p] = sorted(children[p])

    depth = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in children.get(u, []):
            if v not in depth:
                depth[v] = depth[u] + 1
                q.append(v)
    return depth


# ---------------------------
# Bundle C (structure-only)
# ---------------------------
def bundle_c_features(sub_edges, kept_nodes, root, sample_size=150, rng=None):
    kept = list(kept_nodes)
    n = len(kept)
    if n <= 1:
        return {
            "leaf_fraction": 0.0,
            "avg_root_distance": 0.0,
            "sv_proxy": 0.0,
        }

    und = defaultdict(list)
    outdeg = defaultdict(int)
    for p, c, _ in sub_edges:
        und[p].append(c)
        und[c].append(p)
        outdeg[p] += 1
        outdeg.setdefault(c, 0)

    leaves = [v for v in kept if outdeg.get(v, 0) == 0 and v != root]
    leaf_fraction = len(leaves) / max(1, (n - 1))

    depths = node_depths(sub_edges, root)
    dvals = [depths.get(v, np.nan) for v in kept if v != root]
    avg_root_distance = float(np.nanmean(dvals)) if dvals else 0.0

    if rng is None:
        rng = np.random.RandomState(RANDOM_STATE)
    sample = kept if n <= sample_size else list(rng.choice(kept, size=sample_size, replace=False))

    def bfs_dist(s):
        dist = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in und.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    dists = {s: bfs_dist(s) for s in sample}

    pair_ds = []
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            a, b = sample[i], sample[j]
            dij = dists[a].get(b, np.nan)
            if not np.isnan(dij):
                pair_ds.append(dij)

    sv_proxy = float(np.mean(pair_ds)) if pair_ds else 0.0

    return {
        "leaf_fraction": float(np.clip(leaf_fraction, 0.0, 1.0)),
        "avg_root_distance": float(np.clip(avg_root_distance, 0.0, 1e6)),
        "sv_proxy": float(np.clip(sv_proxy, 0.0, 1e6)),
    }


# ---------------------------
# Baseline features for K-window
# ---------------------------
def baseline_k_features(sub_edges, kept_nodes):
    n = len(kept_nodes)
    max_delay = max((t for _, _, t in sub_edges), default=0.0)
    growth = n / (max_delay + 1e-6)
    return {
        "early_n_nodes": float(n),
        "early_growth_rate_k": float(np.clip(growth, 0.0, 1e9)),
    }


def compute_time_to_k(kept_nodes, sub_edges):
    # Prefer node-level delays if present in edges; fallback is NaN handled by imputer.
    node_delay = {}
    for p, c, t in sub_edges:
        if c not in node_delay:
            node_delay[c] = float(t)
        else:
            node_delay[c] = min(node_delay[c], float(t))
    # Root may be absent from child delays; treat it as t=0 by convention.
    if kept_nodes:
        node_delay.setdefault(kept_nodes[0], 0.0)
    delays = [node_delay.get(n, np.nan) for n in kept_nodes]
    finite_delays = [d for d in delays if np.isfinite(d)]
    if not finite_delays:
        return np.nan
    return float(np.max(finite_delays))


# ---------------------------
# Build dataset table
# ---------------------------
def build_k_window_table():
    rng = np.random.RandomState(RANDOM_STATE)
    audit_rows = []
    rows_by_k = {k: [] for k in K_LIST}
    exclusion_log = []

    for dname, spec in DATASETS.items():
        labels = parse_labels(spec["label_path"])
        tree_dir = spec["tree_dir"]

        for fname in sorted(os.listdir(tree_dir)):
            if not fname.endswith(".txt"):
                continue
            cid = os.path.splitext(fname)[0]
            tree_path = os.path.join(tree_dir, fname)

            parse_ok = True
            label = labels.get(cid)
            try:
                edges, nodes, root = parse_tree_file(tree_path)
            except Exception as e:
                parse_ok = False
                edges, nodes, root = [], set(), None
                exclusion_log.append(f"PARSE_FAIL\t{dname}\t{cid}\t{repr(e)}")

            final_size = len(nodes) if parse_ok else np.nan

            if label is None:
                parse_ok = False
                exclusion_log.append(f"LABEL_MISSING\t{dname}\t{cid}")

            audit_rows.append({
                "cascade_id": cid,
                "dataset": dname,
                "label": label if label is not None else "",
                "final_size": final_size,
                "parse_ok": bool(parse_ok),
            })

            if not parse_ok:
                continue

            y_false = 1 if label == "false" else 0
            y_reach = float(np.log1p(final_size))

            for k in K_LIST:
                kept_nodes = bfs_first_k_nodes(edges, root, k)
                sub_edges = induced_subgraph(edges, kept_nodes)

                coverage = 1 if final_size >= k else 0
                feats_base = baseline_k_features(sub_edges, kept_nodes)
                feats_struct = bundle_c_features(sub_edges, kept_nodes, root, sample_size=150, rng=rng)
                feats_full = {**feats_base, **feats_struct}

                rows_by_k[k].append({
                    "cascade_id": cid,
                    "dataset": dname,
                    "label": label,
                    "y_false": y_false,
                    "y_reach": y_reach,
                    "final_size": final_size,
                    "K": k,
                    "coverage_reachK": coverage,
                    "time_to_K": compute_time_to_k(kept_nodes, sub_edges),
                    **{f"base__{name}": val for name, val in feats_base.items()},
                    **{f"struct__{name}": val for name, val in feats_struct.items()},
                    **{f"full__{name}": val for name, val in feats_full.items()},
                })

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(os.path.join(TABLE_DIR, "dataset_audit.csv"), index=False)

    counts = audit_df[audit_df["parse_ok"]].groupby(["dataset", "label"]).size().reset_index(name="n")
    with open(os.path.join(LOG_DIR, "data_audit.txt"), "w", encoding="utf-8") as f:
        f.write("Counts by dataset x label (parse_ok only)\n")
        f.write(counts.to_string(index=False) + "\n\n")
        f.write("Exclusions\n")
        for line in exclusion_log:
            f.write(line + "\n")

    return rows_by_k


# ---------------------------
# CV + evaluation
# ---------------------------
def build_valid_k_context(rows_by_k):
    coverage_rows = []
    k_context = {}

    for k, rows in rows_by_k.items():
        df_all = pd.DataFrame(rows)
        if df_all.empty:
            continue

        n_total = int(len(df_all))
        n_gek = int((df_all["final_size"] >= k).sum())
        n_gtk = int((df_all["final_size"] > k).sum())
        coverage_gek = float(n_gek / n_total) if n_total else np.nan
        coverage_gtk = float(n_gtk / n_total) if n_total else np.nan

        df = df_all[df_all["final_size"] > k].copy()
        n_used = int(len(df))

        # Precompute fixed splits once per K for fair model comparisons.
        splits_cls = None
        if n_used > 0:
            y_false = df["y_false"].astype(int).values
            if len(np.unique(y_false)) == 2:
                class_counts = np.bincount(y_false, minlength=2)
                if np.min(class_counts) >= 5:
                    cv_cls = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                    splits_cls = list(cv_cls.split(np.zeros(n_used), y_false))

        splits_reg = None
        if n_used >= 5:
            cv_reg = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            splits_reg = list(cv_reg.split(np.zeros(n_used)))

        coverage_rows.append({
            "K": k,
            "N_total": n_total,
            "N_geK": n_gek,
            "N_gtK": n_gtk,
            "coverage_geK": coverage_gek,
            "coverage_gtK": coverage_gtk,
        })

        if n_used == 0:
            continue

        k_context[k] = {
            "df": df,
            "N_total": n_total,
            "N_used": n_used,
            "coverage_geK": coverage_gek,
            "coverage_gtK": coverage_gtk,
            "splits_cls": splits_cls,
            "splits_reg": splits_reg,
        }

    cov_df = pd.DataFrame(coverage_rows).sort_values("K")
    cov_df.to_csv(os.path.join(TABLE_DIR, "coverage_k_only_fixed.csv"), index=False)
    return k_context, cov_df


def run_cv_for_k(k_context):
    results = []

    for k, ctx in sorted(k_context.items()):
        df = ctx["df"]
        n_total = ctx["N_total"]
        coverage_gek = ctx["coverage_geK"]
        coverage_gtk = ctx["coverage_gtK"]
        splits_cls = ctx["splits_cls"]
        splits_reg = ctx["splits_reg"]

        X_base = df[[c for c in df.columns if c.startswith("base__")]].copy()
        X_struct = df[[c for c in df.columns if c.startswith("struct__")]].copy()
        X_full = df[[c for c in df.columns if c.startswith("full__")]].copy()

        y_false = df["y_false"].astype(int).values
        y_reach = df["y_reach"].astype(float).values

        logit = Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=5000, random_state=RANDOM_STATE)),
        ])
        ols = Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("sc", StandardScaler()),
            ("reg", LinearRegression()),
        ])

        def eval_one(X, feature_set, task):
            if task == "veracity":
                if splits_cls is None:
                    return
                out = cross_validate(
                    logit,
                    X.values,
                    y_false,
                    cv=splits_cls,
                    scoring={"auc": "roc_auc"},
                    return_train_score=False,
                )
                results.append({
                    "task": "veracity",
                    "window_type": "k",
                    "window_value": k,
                    "feature_set": feature_set,
                    "model": "logit",
                    "metric": "roc_auc",
                    "mean": float(np.mean(out["test_auc"])),
                    "std": float(np.std(out["test_auc"])),
                    "N_total": n_total,
                    "N_used": int(len(X)),
                    "coverage_geK": coverage_gek,
                    "coverage_gtK": coverage_gtk,
                })
            else:
                if splits_reg is None:
                    return
                out = cross_validate(
                    ols,
                    X.values,
                    y_reach,
                    cv=splits_reg,
                    scoring={"r2": "r2", "neg_mae": "neg_mean_absolute_error"},
                    return_train_score=False,
                )
                results.append({
                    "task": "reach",
                    "window_type": "k",
                    "window_value": k,
                    "feature_set": feature_set,
                    "model": "OLS",
                    "metric": "r2",
                    "mean": float(np.mean(out["test_r2"])),
                    "std": float(np.std(out["test_r2"])),
                    "N_total": n_total,
                    "N_used": int(len(X)),
                    "coverage_geK": coverage_gek,
                    "coverage_gtK": coverage_gtk,
                })
                results.append({
                    "task": "reach",
                    "window_type": "k",
                    "window_value": k,
                    "feature_set": feature_set,
                    "model": "OLS",
                    "metric": "mae",
                    "mean": float(-np.mean(out["test_neg_mae"])),
                    "std": float(np.std(-out["test_neg_mae"])),
                    "N_total": n_total,
                    "N_used": int(len(X)),
                    "coverage_geK": coverage_gek,
                    "coverage_gtK": coverage_gtk,
                })

        eval_one(X_base, "baseline", "veracity")
        eval_one(X_struct, "structure_only", "veracity")
        eval_one(X_full, "full", "veracity")

        eval_one(X_base, "baseline", "reach")
        eval_one(X_struct, "structure_only", "reach")
        eval_one(X_full, "full", "reach")

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(TABLE_DIR, "results_k_only_fixed.csv"), index=False)

    return res_df


def get_model_family_registry():
    info = {
        "xgboost_available": False,
        "clf_name": "GB",
        "reg_name": "GBREG",
        "fallback_note": "xgboost not used; using GradientBoosting fallback (HGB unavailable in current runtime).",
    }

    enable_xgb = os.environ.get("ENABLE_XGBOOST", "0") == "1"
    if enable_xgb:
        try:
            from xgboost import XGBClassifier, XGBRegressor  # type: ignore

            info["xgboost_available"] = True
            info["clf_name"] = "XGB"
            info["reg_name"] = "XGBREG"
            info["fallback_note"] = "xgboost available (ENABLE_XGBOOST=1); using XGB models."
            clf_boost = XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=1,
            )
            reg_boost = XGBRegressor(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
        except Exception as e:
            info["fallback_note"] = f"xgboost unavailable at runtime ({repr(e)}); using GradientBoosting fallback (HGB unavailable in current runtime)."
            clf_boost = GradientBoostingClassifier(random_state=RANDOM_STATE)
            reg_boost = GradientBoostingRegressor(random_state=RANDOM_STATE)
    else:
        info["fallback_note"] = "ENABLE_XGBOOST!=1 in this run; using GradientBoosting fallback (HGB unavailable in current runtime)."
        clf_boost = GradientBoostingClassifier(random_state=RANDOM_STATE)
        reg_boost = GradientBoostingRegressor(random_state=RANDOM_STATE)

    models = {
        "veracity": {
            "LOGIT": Pipeline([
                ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                )),
            ]),
            "RF": Pipeline([
                ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                ("clf", RandomForestClassifier(
                    n_estimators=500,
                    random_state=RANDOM_STATE,
                    class_weight="balanced_subsample",
                    n_jobs=1,
                    min_samples_leaf=2,
                )),
            ]),
            info["clf_name"]: Pipeline([
                ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                ("clf", clf_boost),
            ]),
        },
        "reach": {
            "OLS": Pipeline([
                ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                ("sc", StandardScaler()),
                ("reg", LinearRegression()),
            ]),
            "RFREG": Pipeline([
                ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                ("reg", RandomForestRegressor(
                    n_estimators=500,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    min_samples_leaf=2,
                )),
            ]),
            info["reg_name"]: Pipeline([
                ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                ("reg", reg_boost),
            ]),
        },
    }
    return models, info


def run_model_family_comparison_full(k_context, model_registry):
    rows = []
    for k, ctx in sorted(k_context.items()):
        df = ctx["df"]
        X_full = df[[c for c in df.columns if c.startswith("full__")]].copy().values
        y_false = df["y_false"].astype(int).values
        y_reach = df["y_reach"].astype(float).values

        for mname, model in model_registry["veracity"].items():
            if ctx["splits_cls"] is None:
                continue
            out = cross_validate(
                model,
                X_full,
                y_false,
                cv=ctx["splits_cls"],
                scoring={"auc": "roc_auc"},
                return_train_score=False,
            )
            rows.append({
                "task": "veracity",
                "K": k,
                "model": mname,
                "metric": "roc_auc",
                "mean": float(np.mean(out["test_auc"])),
                "std": float(np.std(out["test_auc"])),
                "N_used": int(ctx["N_used"]),
                "coverage_gtK": float(ctx["coverage_gtK"]),
            })

        for mname, model in model_registry["reach"].items():
            if ctx["splits_reg"] is None:
                continue
            out = cross_validate(
                model,
                X_full,
                y_reach,
                cv=ctx["splits_reg"],
                scoring={"r2": "r2", "neg_mae": "neg_mean_absolute_error"},
                return_train_score=False,
            )
            rows.append({
                "task": "reach",
                "K": k,
                "model": mname,
                "metric": "r2",
                "mean": float(np.mean(out["test_r2"])),
                "std": float(np.std(out["test_r2"])),
                "N_used": int(ctx["N_used"]),
                "coverage_gtK": float(ctx["coverage_gtK"]),
            })
            rows.append({
                "task": "reach",
                "K": k,
                "model": mname,
                "metric": "mae",
                "mean": float(-np.mean(out["test_neg_mae"])),
                "std": float(np.std(-out["test_neg_mae"])),
                "N_used": int(ctx["N_used"]),
                "coverage_gtK": float(ctx["coverage_gtK"]),
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(TABLE_DIR, "model_family_comparison_k_full_fixed.csv"), index=False)
    out_df.to_csv(os.path.join(TABLE_DIR, "model_family_comparison_k_full_fixed_with_mae.csv"), index=False)
    return out_df


def run_delta_gain_by_model(k_context, model_registry):
    rows = []
    for k, ctx in sorted(k_context.items()):
        df = ctx["df"]
        X_base = df[[c for c in df.columns if c.startswith("base__")]].copy().values
        X_full = df[[c for c in df.columns if c.startswith("full__")]].copy().values
        y_false = df["y_false"].astype(int).values
        y_reach = df["y_reach"].astype(float).values

        for mname, model in model_registry["veracity"].items():
            if ctx["splits_cls"] is None:
                continue
            out_base = cross_validate(
                model, X_base, y_false, cv=ctx["splits_cls"], scoring={"auc": "roc_auc"}, return_train_score=False
            )
            out_full = cross_validate(
                model, X_full, y_false, cv=ctx["splits_cls"], scoring={"auc": "roc_auc"}, return_train_score=False
            )
            base_mean = float(np.mean(out_base["test_auc"]))
            full_mean = float(np.mean(out_full["test_auc"]))
            rows.append({
                "task": "veracity",
                "K": k,
                "model": mname,
                "metric": "roc_auc",
                "baseline_mean": base_mean,
                "full_mean": full_mean,
                "delta": full_mean - base_mean,
                "N_used": int(ctx["N_used"]),
            })

        for mname, model in model_registry["reach"].items():
            if ctx["splits_reg"] is None:
                continue
            out_base = cross_validate(
                model, X_base, y_reach, cv=ctx["splits_reg"], scoring={"r2": "r2"}, return_train_score=False
            )
            out_full = cross_validate(
                model, X_full, y_reach, cv=ctx["splits_reg"], scoring={"r2": "r2"}, return_train_score=False
            )
            base_mean = float(np.mean(out_base["test_r2"]))
            full_mean = float(np.mean(out_full["test_r2"]))
            rows.append({
                "task": "reach",
                "K": k,
                "model": mname,
                "metric": "r2",
                "baseline_mean": base_mean,
                "full_mean": full_mean,
                "delta": full_mean - base_mean,
                "N_used": int(ctx["N_used"]),
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(TABLE_DIR, "delta_gain_by_model_k_fixed.csv"), index=False)
    return out_df


# ---------------------------
# Plotting
# ---------------------------
def plot_k_results(res_df, cov_df):
    import matplotlib.pyplot as plt

    def get_series(task, metric, feature_set):
        d = res_df[(res_df.task == task) & (res_df.metric == metric) & (res_df.feature_set == feature_set)]
        d = d.sort_values("window_value")
        return d["window_value"].values, d["mean"].values, d["std"].values

    plt.figure()
    for fs in ["baseline", "structure_only", "full"]:
        x, m, s = get_series("veracity", "roc_auc", fs)
        plt.plot(x, m, marker="o", label=fs)
        plt.fill_between(x, m - s, m + s, alpha=0.18)
    plt.axhline(0.5, linestyle="--")
    plt.xlabel("Early k-window (first K nodes)")
    plt.ylabel("AUC (5-fold CV mean ± SD)")
    plt.title("Veracity prediction (k-window, fixed valid set: final_size > K)")
    cov = cov_df.set_index("K")["coverage_gtK"].to_dict()
    n_used_map = {}
    for kval in sorted(cov_df["K"].unique()):
        d = res_df[
            (res_df["task"] == "veracity")
            & (res_df["metric"] == "roc_auc")
            & (res_df["feature_set"] == "baseline")
            & (res_df["window_value"] == kval)
        ]
        if not d.empty:
            n_used_map[kval] = int(d.iloc[0]["N_used"])
        else:
            n_used_map[kval] = int(cov_df.loc[cov_df["K"] == kval, "N_gtK"].iloc[0])
    y0, y1 = plt.ylim()
    span = y1 - y0
    for kval in sorted(cov.keys()):
        plt.text(kval, y0 - 0.02 * span, f"N={n_used_map[kval]}\ncov>{kval}={cov[kval]:.2f}", ha="center", va="top")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "A1K_fixed_k_veracity_baseline_structure_full.png"), dpi=300)
    plt.savefig(os.path.join(FIG_DIR, "A1K_fixed_k_veracity_baseline_structure_full.pdf"))
    plt.close()

    plt.figure()
    for fs in ["baseline", "structure_only", "full"]:
        x, m, s = get_series("reach", "r2", fs)
        plt.plot(x, m, marker="o", label=fs)
        plt.fill_between(x, m - s, m + s, alpha=0.18)
    plt.xlabel("Early k-window (first K nodes)")
    plt.ylabel(r"$R^2$ (5-fold CV mean ± SD)")
    plt.title("Reach prediction (k-window, fixed valid set: final_size > K)")
    y0, y1 = plt.ylim()
    span = y1 - y0
    cov = cov_df.set_index("K")["coverage_gtK"].to_dict()
    n_used_map = {}
    for kval in sorted(cov_df["K"].unique()):
        d = res_df[
            (res_df["task"] == "reach")
            & (res_df["metric"] == "r2")
            & (res_df["feature_set"] == "baseline")
            & (res_df["window_value"] == kval)
        ]
        if not d.empty:
            n_used_map[kval] = int(d.iloc[0]["N_used"])
        else:
            n_used_map[kval] = int(cov_df.loc[cov_df["K"] == kval, "N_gtK"].iloc[0])
    for kval in sorted(cov.keys()):
        plt.text(kval, y0 - 0.02 * span, f"N={n_used_map[kval]}\ncov>{kval}={cov[kval]:.2f}", ha="center", va="top")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "A2K_fixed_k_reach_baseline_structure_full.png"), dpi=300)
    plt.savefig(os.path.join(FIG_DIR, "A2K_fixed_k_reach_baseline_structure_full.pdf"))
    plt.close()

    with open(os.path.join(CAP_DIR, "A1K_fixed.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Veracity prediction (k-window). ROC-AUC (mean±SD) from 5-fold Stratified CV. "
            "Baseline uses early_n_nodes and early_growth_rate_k; structure-only uses Bundle C; full combines both. "
            "At each K, evaluation uses only cascades with final_size > K; under-tick annotations report N_used and coverage_gtK.\n"
        )
    with open(os.path.join(CAP_DIR, "A2K_fixed.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Reach prediction (k-window). Out-of-sample R^2 (mean±SD) from 5-fold CV, target log(1+final_size). "
            "Baseline/structure-only/full as in A1K. At each K, evaluation uses only cascades with final_size > K; under-tick annotations report N_used and coverage_gtK.\n"
        )


def plot_model_family_full_only(model_family_df, cov_df, model_info):
    import matplotlib.pyplot as plt

    clf_models = ["LOGIT", "RF", model_info["clf_name"]]
    reg_models = ["OLS", "RFREG", model_info["reg_name"]]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True)

    ax = axes[0]
    for m in clf_models:
        d = model_family_df[
            (model_family_df["task"] == "veracity")
            & (model_family_df["metric"] == "roc_auc")
            & (model_family_df["model"] == m)
        ].sort_values("K")
        x = d["K"].values
        y = d["mean"].values
        s = d["std"].values
        ax.plot(x, y, marker="o", label=m)
        ax.fill_between(x, y - s, y + s, alpha=0.18)
    ax.axhline(0.5, linestyle="--", color="black", linewidth=1.0)
    ax.set_title("A) Veracity (full features)")
    ax.set_xlabel("K")
    ax.set_ylabel("ROC-AUC")
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    for _, row in cov_df.sort_values("K").iterrows():
        ax.text(
            row["K"],
            y0 - 0.02 * span,
            f"N={int(row['N_gtK'])}\ncov>{int(row['K'])}={row['coverage_gtK']:.2f}",
            ha="center",
            va="top",
            fontsize=8,
        )
    ax.legend()

    ax = axes[1]
    for m in reg_models:
        d = model_family_df[
            (model_family_df["task"] == "reach")
            & (model_family_df["metric"] == "r2")
            & (model_family_df["model"] == m)
        ].sort_values("K")
        x = d["K"].values
        y = d["mean"].values
        s = d["std"].values
        ax.plot(x, y, marker="o", label=m)
        ax.fill_between(x, y - s, y + s, alpha=0.18)
    ax.set_title("B) Reach (full features)")
    ax.set_xlabel("K")
    ax.set_ylabel(r"$R^2$")
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    for _, row in cov_df.sort_values("K").iterrows():
        ax.text(
            row["K"],
            y0 - 0.02 * span,
            f"N={int(row['N_gtK'])}\ncov>{int(row['K'])}={row['coverage_gtK']:.2f}",
            ha="center",
            va="top",
            fontsize=8,
        )
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "A3K_fixed_model_family_full_only.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "A3K_fixed_model_family_full_only.pdf"))
    plt.close(fig)


def plot_delta_gain_by_model(delta_df, model_info):
    import matplotlib.pyplot as plt

    clf_models = ["LOGIT", "RF", model_info["clf_name"]]
    reg_models = ["OLS", "RFREG", model_info["reg_name"]]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True)

    ax = axes[0]
    for m in clf_models:
        d = delta_df[
            (delta_df["task"] == "veracity")
            & (delta_df["metric"] == "roc_auc")
            & (delta_df["model"] == m)
        ].sort_values("K")
        ax.plot(d["K"].values, d["delta"].values, marker="o", label=m)
    ax.axhline(0.0, linestyle="--", color="black", linewidth=1.0)
    ax.set_title("A) ΔAUC (full - baseline)")
    ax.set_xlabel("K")
    ax.set_ylabel("ΔAUC")
    ax.legend()

    ax = axes[1]
    for m in reg_models:
        d = delta_df[
            (delta_df["task"] == "reach")
            & (delta_df["metric"] == "r2")
            & (delta_df["model"] == m)
        ].sort_values("K")
        ax.plot(d["K"].values, d["delta"].values, marker="o", label=m)
    ax.axhline(0.0, linestyle="--", color="black", linewidth=1.0)
    ax.set_title("B) Δ$R^2$ (full - baseline)")
    ax.set_xlabel("K")
    ax.set_ylabel(r"Δ$R^2$")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "A4K_fixed_delta_gain_by_model.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "A4K_fixed_delta_gain_by_model.pdf"))
    plt.close(fig)


def plot_reach_mae_full_only(model_family_df, cov_df, model_info):
    import matplotlib.pyplot as plt

    reg_models = ["OLS", "RFREG", model_info["reg_name"]]
    plt.figure(figsize=(6.2, 4.8))
    for m in reg_models:
        d = model_family_df[
            (model_family_df["task"] == "reach")
            & (model_family_df["metric"] == "mae")
            & (model_family_df["model"] == m)
        ].sort_values("K")
        x = d["K"].values
        y = d["mean"].values
        s = d["std"].values
        plt.plot(x, y, marker="o", label=m)
        plt.fill_between(x, y - s, y + s, alpha=0.18)

    plt.xlabel("K")
    plt.ylabel("MAE")
    plt.title("Reach (MAE), full features (k-window, valid set: final_size > K)")
    y0, y1 = plt.ylim()
    span = y1 - y0
    for _, row in cov_df.sort_values("K").iterrows():
        plt.text(
            row["K"],
            y0 - 0.02 * span,
            f"N={int(row['N_gtK'])}\ncov>{int(row['K'])}={row['coverage_gtK']:.2f}",
            ha="center",
            va="top",
            fontsize=8,
        )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "A5K_fixed_reach_mae_full_only.png"), dpi=300)
    plt.savefig(os.path.join(FIG_DIR, "A5K_fixed_reach_mae_full_only.pdf"))
    plt.close()


def run_extended_time_to_k(k_context, model_registry, model_info):
    results = []
    delta_rows = []

    gb_clf_name = "GB" if "GB" in model_registry["veracity"] else model_info["clf_name"]
    gb_reg_name = "GBREG" if "GBREG" in model_registry["reach"] else model_info["reg_name"]
    models = {
        "veracity": {
            "LOGIT": model_registry["veracity"]["LOGIT"],
            gb_clf_name: model_registry["veracity"][gb_clf_name],
        },
        "reach": {
            "OLS": model_registry["reach"]["OLS"],
            gb_reg_name: model_registry["reach"][gb_reg_name],
        },
    }

    for k, ctx in sorted(k_context.items()):
        df = ctx["df"]
        time_col = df[["time_to_K"]].copy()

        X_base = df[[c for c in df.columns if c.startswith("base__")]].copy()
        X_struct = df[[c for c in df.columns if c.startswith("struct__")]].copy()
        X_full = df[[c for c in df.columns if c.startswith("full__")]].copy()

        X_sets = {
            "baseline_ext": pd.concat([X_base, time_col], axis=1),
            "structure_only_ext": X_struct,
            "full_ext": pd.concat([X_full, time_col], axis=1),
            "full_original": X_full,
        }

        y_false = df["y_false"].astype(int).values
        y_reach = df["y_reach"].astype(float).values

        for mname, model in models["veracity"].items():
            if ctx["splits_cls"] is None:
                continue
            for fset in ["baseline_ext", "structure_only_ext", "full_ext", "full_original"]:
                out = cross_validate(
                    model,
                    X_sets[fset].values,
                    y_false,
                    cv=ctx["splits_cls"],
                    scoring={"auc": "roc_auc"},
                    return_train_score=False,
                )
                mean_auc = float(np.mean(out["test_auc"]))
                std_auc = float(np.std(out["test_auc"]))
                if fset != "full_original":
                    results.append({
                        "task": "veracity",
                        "K": k,
                        "feature_set": fset,
                        "model": mname,
                        "metric": "roc_auc",
                        "mean": mean_auc,
                        "std": std_auc,
                        "N_used": int(ctx["N_used"]),
                        "coverage_gtK": float(ctx["coverage_gtK"]),
                    })
                if fset == "full_original":
                    base_val = mean_auc
                elif fset == "full_ext":
                    ext_val = mean_auc
            delta_rows.append({
                "task": "veracity",
                "K": k,
                "model": mname,
                "metric": "roc_auc",
                "full_original": base_val,
                "full_ext": ext_val,
                "delta_time_to_K": ext_val - base_val,
                "N_used": int(ctx["N_used"]),
            })

        for mname, model in models["reach"].items():
            if ctx["splits_reg"] is None:
                continue
            metric_store = {"full_original": {}, "full_ext": {}}
            for fset in ["baseline_ext", "structure_only_ext", "full_ext", "full_original"]:
                out = cross_validate(
                    model,
                    X_sets[fset].values,
                    y_reach,
                    cv=ctx["splits_reg"],
                    scoring={"r2": "r2", "mae": "neg_mean_absolute_error"},
                    return_train_score=False,
                )
                mean_r2 = float(np.mean(out["test_r2"]))
                std_r2 = float(np.std(out["test_r2"]))
                mean_mae = float(np.mean(-out["test_mae"]))
                std_mae = float(np.std(-out["test_mae"]))
                if fset != "full_original":
                    results.append({
                        "task": "reach",
                        "K": k,
                        "feature_set": fset,
                        "model": mname,
                        "metric": "r2",
                        "mean": mean_r2,
                        "std": std_r2,
                        "N_used": int(ctx["N_used"]),
                        "coverage_gtK": float(ctx["coverage_gtK"]),
                    })
                    results.append({
                        "task": "reach",
                        "K": k,
                        "feature_set": fset,
                        "model": mname,
                        "metric": "mae",
                        "mean": mean_mae,
                        "std": std_mae,
                        "N_used": int(ctx["N_used"]),
                        "coverage_gtK": float(ctx["coverage_gtK"]),
                    })
                if fset in metric_store:
                    metric_store[fset]["r2"] = mean_r2
                    metric_store[fset]["mae"] = mean_mae

            for metric in ["r2", "mae"]:
                base_val = metric_store["full_original"][metric]
                ext_val = metric_store["full_ext"][metric]
                delta_rows.append({
                    "task": "reach",
                    "K": k,
                    "model": mname,
                    "metric": metric,
                    "full_original": base_val,
                    "full_ext": ext_val,
                    "delta_time_to_K": ext_val - base_val,
                    "N_used": int(ctx["N_used"]),
                })

    res_df = pd.DataFrame(results)
    res_df = res_df[
        ["task", "K", "feature_set", "model", "metric", "mean", "std", "N_used", "coverage_gtK"]
    ].sort_values(["task", "metric", "model", "feature_set", "K"])
    res_path = os.path.join(TABLE_DIR, "results_k_only_fixed_extended_time_to_K.csv")
    res_df.to_csv(res_path, index=False)

    delta_df = pd.DataFrame(delta_rows)
    delta_df = delta_df[
        ["task", "K", "model", "metric", "full_original", "full_ext", "delta_time_to_K", "N_used"]
    ].sort_values(["task", "metric", "model", "K"])
    delta_path = os.path.join(TABLE_DIR, "delta_gain_time_to_K_k.csv")
    delta_df.to_csv(delta_path, index=False)
    return res_df, delta_df, res_path, delta_path


def plot_extended_time_to_k_effect(res_df, cov_df, model_registry, model_info):
    import matplotlib.pyplot as plt

    gb_clf_name = "GB" if "GB" in model_registry["veracity"] else model_info["clf_name"]
    gb_reg_name = "GBREG" if "GBREG" in model_registry["reach"] else model_info["reg_name"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True)

    ax = axes[0]
    for fs in ["baseline_ext", "structure_only_ext", "full_ext"]:
        d = res_df[
            (res_df["task"] == "veracity")
            & (res_df["metric"] == "roc_auc")
            & (res_df["model"] == gb_clf_name)
            & (res_df["feature_set"] == fs)
        ].sort_values("K")
        ax.plot(d["K"].values, d["mean"].values, marker="o", label=fs)
        ax.fill_between(d["K"].values, d["mean"].values - d["std"].values, d["mean"].values + d["std"].values, alpha=0.18)
    ax.axhline(0.5, linestyle="--", color="black", linewidth=1.0)
    ax.set_title(f"A) Veracity AUC ({gb_clf_name})")
    ax.set_xlabel("K")
    ax.set_ylabel("ROC-AUC")
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    for _, row in cov_df.sort_values("K").iterrows():
        ax.text(
            row["K"],
            y0 - 0.02 * span,
            f"N={int(row['N_gtK'])}\ncov>{int(row['K'])}={row['coverage_gtK']:.2f}",
            ha="center",
            va="top",
            fontsize=8,
        )
    ax.legend()

    ax = axes[1]
    for fs in ["baseline_ext", "structure_only_ext", "full_ext"]:
        d = res_df[
            (res_df["task"] == "reach")
            & (res_df["metric"] == "r2")
            & (res_df["model"] == gb_reg_name)
            & (res_df["feature_set"] == fs)
        ].sort_values("K")
        ax.plot(d["K"].values, d["mean"].values, marker="o", label=fs)
        ax.fill_between(d["K"].values, d["mean"].values - d["std"].values, d["mean"].values + d["std"].values, alpha=0.18)
    ax.set_title(f"B) Reach $R^2$ ({gb_reg_name})")
    ax.set_xlabel("K")
    ax.set_ylabel(r"$R^2$")
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    for _, row in cov_df.sort_values("K").iterrows():
        ax.text(
            row["K"],
            y0 - 0.02 * span,
            f"N={int(row['N_gtK'])}\ncov>{int(row['K'])}={row['coverage_gtK']:.2f}",
            ha="center",
            va="top",
            fontsize=8,
        )
    ax.legend()

    fig.tight_layout()
    out_png = os.path.join(FIG_DIR, "A7K_extended_time_to_K_effect.png")
    out_pdf = os.path.join(FIG_DIR, "A7K_extended_time_to_K_effect.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_png, out_pdf


def write_extended_time_to_k_artifacts(delta_df, model_info, created_paths):
    out_caption = os.path.join(CAP_DIR, "A7K_extended.txt")
    with open(out_caption, "w", encoding="utf-8") as f:
        f.write(
            "Adding time_to_K restores timing heterogeneity that the pure k-window node set suppresses: two cascades can share the first K nodes but reach them at different rates. "
            "Extended baseline/full feature sets therefore test whether this recovered timing signal improves veracity and reach prediction under the valid set final_size > K.\n"
        )

    out_log = os.path.join(LOG_DIR, "run_log_k_extended_time_to_K.txt")
    with open(out_log, "w", encoding="utf-8") as f:
        f.write("K-window extended run with time_to_K\n")
        f.write(f"xgboost_available={model_info['xgboost_available']}\n")
        f.write(f"model_note={model_info['fallback_note']}\n\n")
        f.write("Delta summary full_ext - full_original\n")
        for metric in ["roc_auc", "r2", "mae"]:
            d = delta_df[delta_df["metric"] == metric]
            if d.empty:
                continue
            mean_delta = float(np.mean(d["delta_time_to_K"]))
            if metric in {"roc_auc", "r2"}:
                improve = int((d["delta_time_to_K"] > 0).sum())
            else:
                improve = int((d["delta_time_to_K"] < 0).sum())
            f.write(
                f"{metric}: mean_delta={mean_delta:.6f}, improved_cells={improve}/{len(d)}\n"
            )
        f.write("\nCreated files\n")
        for p in created_paths + [out_caption, out_log]:
            f.write(p + "\n")
    return out_caption, out_log


def run_repeated_cv_k50_only(rows_by_k):
    import matplotlib.pyplot as plt

    k_context, cov_df = build_valid_k_context(rows_by_k)
    k = 50
    if k not in k_context:
        raise RuntimeError("K=50 context not available after validity filtering.")

    ctx = k_context[k]
    df = ctx["df"]
    X_full = df[[c for c in df.columns if c.startswith("full__")]].copy().values
    y_false = df["y_false"].astype(int).values
    y_reach = df["y_reach"].astype(float).values

    model_registry, model_info = get_model_family_registry()
    clf_model = model_registry["veracity"].get("GB", model_registry["veracity"][model_info["clf_name"]])
    reg_model = model_registry["reach"].get("GBREG", model_registry["reach"][model_info["reg_name"]])
    clf_name = "GB" if "GB" in model_registry["veracity"] else model_info["clf_name"]
    reg_name = "GBREG" if "GBREG" in model_registry["reach"] else model_info["reg_name"]

    cv_cls = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    cv_reg = RepeatedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)

    out_cls = cross_validate(
        clf_model,
        X_full,
        y_false,
        cv=cv_cls,
        scoring={"auc": "roc_auc"},
        return_train_score=False,
    )
    auc_scores = out_cls["test_auc"]

    out_reg = cross_validate(
        reg_model,
        X_full,
        y_reach,
        cv=cv_reg,
        scoring={"r2": "r2", "mae": "neg_mean_absolute_error"},
        return_train_score=False,
    )
    r2_scores = out_reg["test_r2"]
    mae_scores = -out_reg["test_mae"]

    rows = [
        {
            "task": "veracity",
            "K": 50,
            "model": clf_name,
            "cv_type": "RepeatedStratifiedKFold",
            "n_splits": 5,
            "n_repeats": 10,
            "metric": "roc_auc",
            "mean": float(np.mean(auc_scores)),
            "std": float(np.std(auc_scores)),
            "N_used": int(ctx["N_used"]),
            "coverage_gtK": float(ctx["coverage_gtK"]),
        },
        {
            "task": "reach",
            "K": 50,
            "model": reg_name,
            "cv_type": "RepeatedKFold",
            "n_splits": 5,
            "n_repeats": 10,
            "metric": "r2",
            "mean": float(np.mean(r2_scores)),
            "std": float(np.std(r2_scores)),
            "N_used": int(ctx["N_used"]),
            "coverage_gtK": float(ctx["coverage_gtK"]),
        },
        {
            "task": "reach",
            "K": 50,
            "model": reg_name,
            "cv_type": "RepeatedKFold",
            "n_splits": 5,
            "n_repeats": 10,
            "metric": "mae",
            "mean": float(np.mean(mae_scores)),
            "std": float(np.std(mae_scores)),
            "N_used": int(ctx["N_used"]),
            "coverage_gtK": float(ctx["coverage_gtK"]),
        },
    ]

    out_table = os.path.join(TABLE_DIR, "repeated_cv_k50_full_fixed.csv")
    pd.DataFrame(rows).to_csv(out_table, index=False)

    # A6 figure: score distributions across repeated CV splits.
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    panel_specs = [
        ("AUC", auc_scores, "tab:blue"),
        (r"$R^2$", r2_scores, "tab:green"),
        ("MAE", mae_scores, "tab:red"),
    ]
    for ax, (label, vals, color) in zip(axes, panel_specs):
        ax.hist(vals, bins=12, color=color, alpha=0.75, edgecolor="black", linewidth=0.6)
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        ax.axvline(mu, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(label)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.text(0.98, 0.95, f"mean={mu:.3f}\nstd={sd:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    fig.suptitle("Repeated CV at K=50 (full features, valid set: final_size > 50)", fontsize=11)
    fig.tight_layout()
    out_fig_png = os.path.join(FIG_DIR, "A6K_repeated_cv_k50_distributions.png")
    out_fig_pdf = os.path.join(FIG_DIR, "A6K_repeated_cv_k50_distributions.pdf")
    fig.savefig(out_fig_png, dpi=300)
    fig.savefig(out_fig_pdf)
    plt.close(fig)

    out_caption = os.path.join(CAP_DIR, "A6K_fixed.txt")
    with open(out_caption, "w", encoding="utf-8") as f:
        f.write(
            "Repeated 5x10 cross-validation at K=50 (full features, valid set final_size > 50) is used because N_used is smaller at large K, "
            "so single 5-fold estimates can have higher variance; distributions summarize ROC-AUC, R^2, and MAE (on y_reach=log(1+final_size)).\n"
        )

    out_log = os.path.join(LOG_DIR, "run_log_repeated_cv_k50_fixed.txt")
    created = [out_table, out_fig_png, out_fig_pdf, out_caption, out_log]
    with open(out_log, "w", encoding="utf-8") as f:
        f.write("Repeated CV K=50 full-feature stability check\n")
        f.write(f"xgboost_available={model_info['xgboost_available']}\n")
        f.write(f"model_note={model_info['fallback_note']}\n")
        f.write(f"N_used={int(ctx['N_used'])}\n")
        f.write(f"coverage_gtK={float(ctx['coverage_gtK']):.6f}\n\n")
        f.write("Created files\n")
        for p in created:
            f.write(p + "\n")

    print("Done. Created outputs:")
    for p in created:
        print(p)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeated-k50-only", action="store_true")
    parser.add_argument("--extended-time-to-k-only", action="store_true")
    args = parser.parse_args()

    rows_by_k = build_k_window_table()
    if args.repeated_k50_only:
        run_repeated_cv_k50_only(rows_by_k)
        raise SystemExit(0)
    if args.extended_time_to_k_only:
        k_context, cov_df = build_valid_k_context(rows_by_k)
        model_registry, model_info = get_model_family_registry()
        res_ext, delta_ext, res_path, delta_path = run_extended_time_to_k(k_context, model_registry, model_info)
        fig_png, fig_pdf = plot_extended_time_to_k_effect(res_ext, cov_df, model_registry, model_info)
        cap_path, log_path = write_extended_time_to_k_artifacts(
            delta_ext, model_info, [res_path, delta_path, fig_png, fig_pdf]
        )
        print("Done. Created outputs:")
        for p in [res_path, delta_path, fig_png, fig_pdf, cap_path, log_path]:
            print(p)
        raise SystemExit(0)

    k_context, cov_df = build_valid_k_context(rows_by_k)
    res_df = run_cv_for_k(k_context)
    plot_k_results(res_df, cov_df)
    model_registry, model_info = get_model_family_registry()
    model_family_df = run_model_family_comparison_full(k_context, model_registry)
    delta_df = run_delta_gain_by_model(k_context, model_registry)
    plot_model_family_full_only(model_family_df, cov_df, model_info)
    plot_delta_gain_by_model(delta_df, model_info)
    plot_reach_mae_full_only(model_family_df, cov_df, model_info)

    created_paths = [
        os.path.join(TABLE_DIR, "dataset_audit.csv"),
        os.path.join(LOG_DIR, "data_audit.txt"),
        os.path.join(TABLE_DIR, "results_k_only_fixed.csv"),
        os.path.join(TABLE_DIR, "coverage_k_only_fixed.csv"),
        os.path.join(TABLE_DIR, "model_family_comparison_k_full_fixed.csv"),
        os.path.join(TABLE_DIR, "model_family_comparison_k_full_fixed_with_mae.csv"),
        os.path.join(TABLE_DIR, "delta_gain_by_model_k_fixed.csv"),
        os.path.join(FIG_DIR, "A1K_fixed_k_veracity_baseline_structure_full.png"),
        os.path.join(FIG_DIR, "A1K_fixed_k_veracity_baseline_structure_full.pdf"),
        os.path.join(FIG_DIR, "A2K_fixed_k_reach_baseline_structure_full.png"),
        os.path.join(FIG_DIR, "A2K_fixed_k_reach_baseline_structure_full.pdf"),
        os.path.join(FIG_DIR, "A3K_fixed_model_family_full_only.png"),
        os.path.join(FIG_DIR, "A3K_fixed_model_family_full_only.pdf"),
        os.path.join(FIG_DIR, "A4K_fixed_delta_gain_by_model.png"),
        os.path.join(FIG_DIR, "A4K_fixed_delta_gain_by_model.pdf"),
        os.path.join(FIG_DIR, "A5K_fixed_reach_mae_full_only.png"),
        os.path.join(FIG_DIR, "A5K_fixed_reach_mae_full_only.pdf"),
        os.path.join(CAP_DIR, "A1K_fixed.txt"),
        os.path.join(CAP_DIR, "A2K_fixed.txt"),
        os.path.join(CAP_DIR, "A5K_fixed.txt"),
        os.path.join(LOG_DIR, "run_config_k_only_fixed.json"),
        os.path.join(LOG_DIR, "run_log_k_only_fixed.txt"),
        os.path.join(LOG_DIR, "run_log_k_model_family_fixed.txt"),
        os.path.join(LOG_DIR, "run_log_k_mae_fixed.txt"),
    ]

    run_cfg = {
        "random_state": RANDOM_STATE,
        "K_LIST": K_LIST,
        "root_dir": ROOT_DIR,
        "datasets": DATASETS,
        "outputs": {"tables": TABLE_DIR, "figures": FIG_DIR, "captions": CAP_DIR, "logs": LOG_DIR},
        "xgboost_available": bool(model_info["xgboost_available"]),
        "xgb_or_fallback_classifier": model_info["clf_name"],
        "xgb_or_fallback_regressor": model_info["reg_name"],
        "notes": "K-window only; BFS-first-K nodes; strict valid set final_size>K; baseline + nonlinear model families.",
    }
    with open(os.path.join(LOG_DIR, "run_config_k_only_fixed.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    with open(os.path.join(LOG_DIR, "run_log_k_only_fixed.txt"), "w", encoding="utf-8") as f:
        f.write("Created outputs:\n")
        for p in created_paths:
            f.write(p + "\n")

    with open(os.path.join(LOG_DIR, "run_log_k_model_family_fixed.txt"), "w", encoding="utf-8") as f:
        f.write("Model family run summary\n")
        f.write(f"xgboost_available={model_info['xgboost_available']}\n")
        f.write(f"model_note={model_info['fallback_note']}\n\n")
        f.write("Per-K valid-set coverage and N_used (final_size > K)\n")
        for _, row in cov_df.sort_values("K").iterrows():
            f.write(
                f"K={int(row['K'])}\tN_total={int(row['N_total'])}\tN_geK={int(row['N_geK'])}\t"
                f"N_gtK={int(row['N_gtK'])}\tcoverage_geK={row['coverage_geK']:.6f}\tcoverage_gtK={row['coverage_gtK']:.6f}\n"
            )
        f.write("\nCreated files\n")
        new_only = [
            os.path.join(TABLE_DIR, "model_family_comparison_k_full_fixed.csv"),
            os.path.join(TABLE_DIR, "model_family_comparison_k_full_fixed_with_mae.csv"),
            os.path.join(TABLE_DIR, "delta_gain_by_model_k_fixed.csv"),
            os.path.join(FIG_DIR, "A3K_fixed_model_family_full_only.png"),
            os.path.join(FIG_DIR, "A3K_fixed_model_family_full_only.pdf"),
            os.path.join(FIG_DIR, "A4K_fixed_delta_gain_by_model.png"),
            os.path.join(FIG_DIR, "A4K_fixed_delta_gain_by_model.pdf"),
            os.path.join(LOG_DIR, "run_log_k_model_family_fixed.txt"),
        ]
        for p in new_only:
            f.write(p + "\n")

    with open(os.path.join(CAP_DIR, "A5K_fixed.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Reach MAE (k-window, full features) across model families on the valid set (final_size > K). "
            "MAE is computed on y_reach = log(1+final_size), so lower values indicate better predictive accuracy.\n"
        )

    with open(os.path.join(LOG_DIR, "run_log_k_mae_fixed.txt"), "w", encoding="utf-8") as f:
        f.write("K-window MAE full-model-family summary\n")
        f.write(f"xgboost_available={model_info['xgboost_available']}\n")
        f.write(f"model_note={model_info['fallback_note']}\n\n")
        f.write("Per-K valid-set coverage and N_used (final_size > K)\n")
        for _, row in cov_df.sort_values("K").iterrows():
            f.write(
                f"K={int(row['K'])}\tN_gtK={int(row['N_gtK'])}\tcoverage_gtK={row['coverage_gtK']:.6f}\n"
            )
        f.write("\nCreated files\n")
        for p in [
            os.path.join(TABLE_DIR, "model_family_comparison_k_full_fixed_with_mae.csv"),
            os.path.join(FIG_DIR, "A5K_fixed_reach_mae_full_only.png"),
            os.path.join(FIG_DIR, "A5K_fixed_reach_mae_full_only.pdf"),
            os.path.join(CAP_DIR, "A5K_fixed.txt"),
            os.path.join(LOG_DIR, "run_log_k_mae_fixed.txt"),
        ]:
            f.write(p + "\n")

    print("Done. Created outputs:")
    for p in [
        os.path.join(TABLE_DIR, "model_family_comparison_k_full_fixed_with_mae.csv"),
        os.path.join(FIG_DIR, "A5K_fixed_reach_mae_full_only.png"),
        os.path.join(FIG_DIR, "A5K_fixed_reach_mae_full_only.pdf"),
        os.path.join(CAP_DIR, "A5K_fixed.txt"),
        os.path.join(LOG_DIR, "run_log_k_mae_fixed.txt"),
    ]:
        print(p)
