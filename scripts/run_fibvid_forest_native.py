#!/usr/bin/env python3
from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FIBVID_ROOT = ROOT / "merry555-FibVID-14b95c3"
OUT_DIR = ROOT / "new data" / "fibvid_forest_native"
RANDOM_STATE = 42
WINDOWS = {"W30": 30, "W60": 60, "W180": 180}
LABEL_MAP = {0: 0, 2: 0, 1: 1, 3: 1}


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    cp = pd.read_csv(
        FIBVID_ROOT / "claim_propagation" / "claim_propagation.csv",
        usecols=[
            "claim_number",
            "tweet_id",
            "parent_id",
            "create_date",
            "depth",
            "tweet_user",
            "like_count",
            "retweet_count",
        ],
        low_memory=False,
    )
    nc = pd.read_csv(
        FIBVID_ROOT / "news_claim" / "news_claim.csv",
        usecols=["claim_num", "group"],
        low_memory=False,
    )

    cp["create_date"] = pd.to_datetime(cp["create_date"], utc=True, errors="coerce")
    cp = cp.dropna(subset=["create_date"]).copy()
    cp["claim_number"] = pd.to_numeric(cp["claim_number"], errors="coerce").astype("Int64")
    cp = cp.dropna(subset=["claim_number"]).copy()
    cp["claim_number"] = cp["claim_number"].astype(int)
    cp["tweet_id"] = pd.to_numeric(cp["tweet_id"], errors="coerce").fillna(-1).astype(np.int64)
    cp["parent_id"] = pd.to_numeric(cp["parent_id"], errors="coerce").fillna(-1).astype(np.int64)
    cp["depth"] = pd.to_numeric(cp["depth"], errors="coerce").fillna(0.0)
    cp["tweet_user"] = pd.to_numeric(cp["tweet_user"], errors="coerce")
    cp["like_count"] = pd.to_numeric(cp["like_count"], errors="coerce").fillna(0.0)
    cp["retweet_count"] = pd.to_numeric(cp["retweet_count"], errors="coerce").fillna(0.0)

    nc["claim_num"] = pd.to_numeric(nc["claim_num"], errors="coerce").astype("Int64")
    nc = nc.dropna(subset=["claim_num"]).copy()
    nc["claim_num"] = nc["claim_num"].astype(int)
    claim_meta = nc[["claim_num", "group"]].drop_duplicates(subset=["claim_num"]).copy()
    claim_meta = claim_meta.loc[claim_meta["group"].isin(LABEL_MAP.keys())].copy()
    claim_meta["y"] = claim_meta["group"].map(LABEL_MAP).astype(int)
    claim_meta.rename(columns={"claim_num": "claim_number"}, inplace=True)
    return cp, claim_meta


def normalized_entropy(values: list[int]) -> float:
    arr = np.asarray(values, dtype=float)
    total = arr.sum()
    if total <= 0:
        return 0.0
    p = arr / total
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    h = -float(np.sum(p * np.log(p)))
    return float(h / np.log(float(p.size)))


def forest_features_for_window(cp: pd.DataFrame, window_minutes: int, root_min_map: pd.Series) -> pd.DataFrame:
    cp_local = cp.copy()
    cp_local["root_time"] = cp_local["claim_number"].map(root_min_map)
    cp_local["t_min"] = ((cp_local["create_date"] - cp_local["root_time"]).dt.total_seconds() / 60.0).clip(lower=0.0)
    early = cp_local.loc[cp_local["t_min"] <= window_minutes].copy()

    rows = []
    for claim_id, g in early.groupby("claim_number", sort=False):
        g = g.sort_values(["create_date", "tweet_id"], kind="mergesort").reset_index(drop=True)
        n_nodes = int(len(g))
        if n_nodes == 0:
            continue

        node_time = dict(zip(g["tweet_id"], g["create_date"]))
        node_idx = {tid: i for i, tid in enumerate(g["tweet_id"].tolist())}
        children = defaultdict(list)
        roots = []
        root_of = {}

        for row in g.itertuples(index=False):
            tid = int(row.tweet_id)
            pid = int(row.parent_id)
            if pid <= 0 or pid not in node_time or node_time[pid] > row.create_date or node_idx[pid] >= node_idx[tid]:
                roots.append(tid)
            else:
                children[pid].append(tid)

        if not roots:
            roots = [int(g.iloc[0]["tweet_id"])]

        depth_map = {}
        width_counter = Counter()
        comp_sizes = []
        branch_vals = []
        for root in roots:
            if root in depth_map:
                continue
            stack = [(root, 0)]
            comp_nodes = []
            while stack:
                u, d = stack.pop()
                if u in depth_map:
                    continue
                depth_map[u] = d
                width_counter[d] += 1
                comp_nodes.append(u)
                root_of[u] = root
                ch = children.get(u, [])
                branch_vals.append(len(ch))
                for v in ch:
                    stack.append((v, d + 1))
            comp_sizes.append(len(comp_nodes))

        # Guard against any isolated nodes missed due to broken links.
        for tid in g["tweet_id"].tolist():
            if tid not in depth_map:
                depth_map[tid] = 0
                width_counter[0] += 1
                roots.append(tid)
                comp_sizes.append(1)
                branch_vals.append(0)
                root_of[tid] = tid

        root_counts = Counter(root_of.values())
        largest_comp = max(comp_sizes) if comp_sizes else 1
        max_depth = max(depth_map.values()) if depth_map else 0
        max_width = max(width_counter.values()) if width_counter else 1
        branch_nonleaf = [x for x in branch_vals if x > 0]
        time_span = float(min(window_minutes, g["t_min"].max())) if n_nodes > 0 else 0.0

        rows.append(
            {
                "claim_number": int(claim_id),
                "n_nodes": n_nodes,
                "n_unique_users": int(g["tweet_user"].nunique(dropna=True)),
                "n_roots": int(len(root_counts)),
                "largest_component_share": float(largest_comp / max(n_nodes, 1)),
                "component_size_entropy": normalized_entropy(comp_sizes),
                "root_concentration": float(max(root_counts.values()) / max(n_nodes, 1)),
                "max_depth_forest": float(max_depth),
                "mean_depth_forest": float(np.mean(list(depth_map.values()))) if depth_map else 0.0,
                "max_width_forest": float(max_width),
                "branching_mean_nonleaf": float(np.mean(branch_nonleaf)) if branch_nonleaf else 0.0,
                "branching_all_nodes": float(np.mean(branch_vals)) if branch_vals else 0.0,
                "sum_retweets": float(g["retweet_count"].sum()),
                "sum_likes": float(g["like_count"].sum()),
                "mean_retweets": float(g["retweet_count"].mean()),
                "mean_likes": float(g["like_count"].mean()),
                "time_span_min": time_span,
                "node_rate_per_min": float(n_nodes / max(1.0, time_span)),
            }
        )
    return pd.DataFrame(rows)


def preprocess_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy().fillna(0.0)
    for c in ["sum_retweets", "sum_likes", "mean_retweets", "mean_likes"]:
        if c in X.columns:
            X[c] = np.log1p(np.clip(pd.to_numeric(X[c], errors="coerce").fillna(0.0), a_min=0, a_max=None))
    return X.fillna(0.0)


def metrics_cv_cls(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    aucs, f1s, accs = [], [], []
    for train_idx, test_idx in skf.split(X, y):
        clf = clone(model)
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = clf.predict(X.iloc[test_idx])
        y_score = clf.predict_proba(X.iloc[test_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[test_idx], y_score))
        tp = int(((y_pred == 1) & (y.iloc[test_idx] == 1)).sum())
        fp = int(((y_pred == 1) & (y.iloc[test_idx] == 0)).sum())
        fn = int(((y_pred == 0) & (y.iloc[test_idx] == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1s.append(0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall))
        accs.append(float((y_pred == y.iloc[test_idx]).mean()))
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs, ddof=1)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s, ddof=1)),
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs, ddof=1)),
    }


def spearman_rank_corr(y_true: pd.Series, y_pred: np.ndarray) -> float:
    rt = pd.Series(np.asarray(y_true)).rank(method="average")
    rp = pd.Series(np.asarray(y_pred)).rank(method="average")
    corr = rt.corr(rp, method="pearson")
    return float(corr) if pd.notna(corr) else 0.0


def metrics_cv_reg(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    r2s, maes, spears = [], [], []
    for train_idx, test_idx in kf.split(X):
        reg = clone(model)
        reg.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = reg.predict(X.iloc[test_idx])
        r2s.append(r2_score(y.iloc[test_idx], pred))
        maes.append(mean_absolute_error(y.iloc[test_idx], pred))
        spears.append(spearman_rank_corr(y.iloc[test_idx], pred))
    return {
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s, ddof=1)),
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes, ddof=1)),
        "spearman_mean": float(np.mean(spears)),
        "spearman_std": float(np.std(spears, ddof=1)),
    }


def main() -> None:
    ensure_dirs()
    cp, claim_meta = load_data()

    totals = cp.groupby("claim_number", as_index=False).agg(
        N_total=("tweet_id", "size"),
        sum_retweets_total=("retweet_count", "sum"),
    )
    totals["y_size"] = np.log1p(totals["N_total"])
    totals["y_rt"] = np.log1p(np.clip(totals["sum_retweets_total"], a_min=0, a_max=None))
    root_min_all = cp.groupby("claim_number")["create_date"].min()

    baseline_cols = ["n_nodes", "node_rate_per_min", "n_roots", "largest_component_share"]
    full_feature_cols = [
        "n_nodes",
        "n_unique_users",
        "n_roots",
        "largest_component_share",
        "component_size_entropy",
        "root_concentration",
        "max_depth_forest",
        "mean_depth_forest",
        "max_width_forest",
        "branching_mean_nonleaf",
        "branching_all_nodes",
        "sum_retweets",
        "sum_likes",
        "mean_retweets",
        "mean_likes",
        "time_span_min",
        "node_rate_per_min",
    ]

    cls_models = {
        "BaselineLogistic(forest)": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))]
        ),
        "LogisticRegression(forest_full)": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))]
        ),
        "RandomForest(forest_full)": RandomForestClassifier(
            n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced_subsample"
        ),
    }
    reg_models = {
        "BaselineRidge(forest)": Pipeline(
            [("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0, random_state=RANDOM_STATE))]
        ),
        "Ridge(forest_full)": Pipeline(
            [("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0, random_state=RANDOM_STATE))]
        ),
        "RandomForestRegressor(forest_full)": RandomForestRegressor(
            n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    coverage_rows = []
    ver_rows = []
    spread_rows = []

    for wname, wmin in WINDOWS.items():
        feats = forest_features_for_window(cp, wmin, root_min_all)
        merged = claim_meta[["claim_number", "y"]].merge(feats, on="claim_number", how="left").merge(
            totals[["claim_number", "y_size", "y_rt"]], on="claim_number", how="left"
        ).fillna(0.0)
        merged["window_name"] = wname
        merged["window_minutes"] = wmin
        merged["is_valid_for_model"] = (merged["n_nodes"] >= 2).astype(int)

        valid = merged.loc[merged["n_nodes"] >= 2].copy()
        coverage_rows.append(
            {
                "window_name": wname,
                "window_minutes": wmin,
                "n_claims_total": int(len(merged)),
                "n_claims_used": int(len(valid)),
                "used_ratio": float(len(valid) / max(1, len(merged))),
                "n_nodes_min": float(merged["n_nodes"].min()),
                "n_nodes_median": float(merged["n_nodes"].median()),
                "n_nodes_p90": float(merged["n_nodes"].quantile(0.9)),
                "n_roots_median": float(merged["n_roots"].median()),
                "largest_component_share_median": float(merged["largest_component_share"].median()),
            }
        )

        if len(valid) < 30 or valid["y"].nunique() < 2:
            continue

        X_base_cls = preprocess_features(valid, baseline_cols)
        X_full_cls = preprocess_features(valid, full_feature_cols)
        y_cls = valid["y"].astype(int)
        ver_rows.append(
            {"window_name": wname, "window_minutes": wmin, "model": "BaselineLogistic(forest)", **metrics_cv_cls(cls_models["BaselineLogistic(forest)"], X_base_cls, y_cls), "n_claims_used": int(len(valid)), "n_features": len(baseline_cols)}
        )
        ver_rows.append(
            {"window_name": wname, "window_minutes": wmin, "model": "LogisticRegression(forest_full)", **metrics_cv_cls(cls_models["LogisticRegression(forest_full)"], X_full_cls, y_cls), "n_claims_used": int(len(valid)), "n_features": len(full_feature_cols)}
        )
        ver_rows.append(
            {"window_name": wname, "window_minutes": wmin, "model": "RandomForest(forest_full)", **metrics_cv_cls(cls_models["RandomForest(forest_full)"], X_full_cls, y_cls), "n_claims_used": int(len(valid)), "n_features": len(full_feature_cols)}
        )

        for outcome_name, ycol in [("final_size_log1p", "y_size"), ("final_retweets_log1p", "y_rt")]:
            y_reg = valid[ycol].astype(float)
            spread_rows.append(
                {"window_name": wname, "window_minutes": wmin, "outcome": outcome_name, "model": "BaselineRidge(forest)", **metrics_cv_reg(reg_models["BaselineRidge(forest)"], X_base_cls, y_reg), "n_claims_used": int(len(valid)), "n_features": len(baseline_cols)}
            )
            spread_rows.append(
                {"window_name": wname, "window_minutes": wmin, "outcome": outcome_name, "model": "Ridge(forest_full)", **metrics_cv_reg(reg_models["Ridge(forest_full)"], X_full_cls, y_reg), "n_claims_used": int(len(valid)), "n_features": len(full_feature_cols)}
            )
            spread_rows.append(
                {"window_name": wname, "window_minutes": wmin, "outcome": outcome_name, "model": "RandomForestRegressor(forest_full)", **metrics_cv_reg(reg_models["RandomForestRegressor(forest_full)"], X_full_cls, y_reg), "n_claims_used": int(len(valid)), "n_features": len(full_feature_cols)}
            )

    coverage_df = pd.DataFrame(coverage_rows)
    ver_df = pd.DataFrame(ver_rows)
    spread_df = pd.DataFrame(spread_rows)
    coverage_df.to_csv(OUT_DIR / "forest_coverage.csv", index=False)
    ver_df.to_csv(OUT_DIR / "forest_veracity_results.csv", index=False)
    spread_df.to_csv(OUT_DIR / "forest_spread_results.csv", index=False)

    best_ver = ver_df.sort_values("auc_mean", ascending=False).iloc[0]
    best_size = spread_df[spread_df["outcome"] == "final_size_log1p"].sort_values("r2_mean", ascending=False).iloc[0]
    lines = [
        "# FibVID Forest-Native Analysis",
        "",
        "This analysis keeps FibVID as a multi-root forest rather than forcing each claim into a single-root tree.",
        "",
        "## Best Results",
        f"- Veracity best AUC: {best_ver['auc_mean']:.3f} at {best_ver['window_name']} with {best_ver['model']}",
        f"- Final size best R^2: {best_size['r2_mean']:.3f} at {best_size['window_name']} with {best_size['model']}",
        "",
    ]
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    # Simple figure for quick inspection.
    if not ver_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model in ver_df["model"].unique():
            d = ver_df[ver_df["model"] == model].sort_values("window_minutes")
            ax.plot(d["window_minutes"], d["auc_mean"], marker="o", label=model)
        ax.set_xlabel("Window (minutes)")
        ax.set_ylabel("AUC")
        ax.set_title("FibVID Forest-Native Veracity")
        ax.legend(frameon=False)
        ax.grid(True, axis="y", alpha=0.2, linestyle=":")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "forest_veracity.png", dpi=180)
        plt.close(fig)

    print(f"Wrote forest-native FibVID outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
