#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thesis_pipeline.runner import main as run_main_pipeline

FIBVID_ROOT = ROOT / "merry555-FibVID-14b95c3"
OUT_ROOT = ROOT / "new data"
ADAPTED_ROOT = OUT_ROOT / "fibvid_acl2017_like"
ADAPTED_DATA_ROOT = ADAPTED_ROOT / "rumor_detection_acl2017"

LABEL_MAP = {
    0: "true",
    2: "true",
    1: "false",
    3: "false",
}


@dataclass
class ClaimConversion:
    claim_number: int
    cascade_id: str
    label_group: int
    label_text: str
    root_tweet_id: str
    root_time: str
    n_nodes_original: int
    n_nodes_written: int
    n_rewired_to_root: int
    n_parent_zero: int
    n_parent_backward: int
    n_parent_missing: int


def ensure_dirs() -> None:
    for rel in [
        ADAPTED_DATA_ROOT / "twitter15" / "tree",
        ADAPTED_DATA_ROOT / "twitter16" / "tree",
        OUT_ROOT / "logs",
    ]:
        rel.mkdir(parents=True, exist_ok=True)


def load_fibvid_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    cp = pd.read_csv(
        FIBVID_ROOT / "claim_propagation" / "claim_propagation.csv",
        usecols=["tweet_user", "tweet_id", "parent_id", "create_date", "claim_number", "group"],
        low_memory=False,
    )
    nc = pd.read_csv(
        FIBVID_ROOT / "news_claim" / "news_claim.csv",
        usecols=["claim_num", "group"],
        low_memory=False,
    )

    cp["create_date"] = pd.to_datetime(cp["create_date"], utc=True, errors="coerce")
    cp = cp.dropna(subset=["create_date", "claim_number", "tweet_id"]).copy()
    cp["claim_number"] = pd.to_numeric(cp["claim_number"], errors="coerce").astype("Int64")
    cp = cp.dropna(subset=["claim_number"]).copy()
    cp["claim_number"] = cp["claim_number"].astype(int)
    cp["tweet_id"] = cp["tweet_id"].astype(str)
    cp["parent_id"] = cp["parent_id"].fillna("0").astype(str)
    cp["tweet_user"] = cp["tweet_user"].fillna("NA_USER").astype(str)

    nc["claim_num"] = pd.to_numeric(nc["claim_num"], errors="coerce").astype("Int64")
    nc = nc.dropna(subset=["claim_num"]).copy()
    nc["claim_num"] = nc["claim_num"].astype(int)

    labels = nc[["claim_num", "group"]].drop_duplicates(subset=["claim_num"]).copy()
    labels = labels[labels["group"].isin(LABEL_MAP)].copy()
    labels["label_text"] = labels["group"].map(LABEL_MAP)
    return cp, labels


def pick_root(group: pd.DataFrame) -> pd.Series:
    roots = group[group["parent_id"] == "0"].copy()
    if roots.empty:
        roots = group.copy()
    roots = roots.sort_values(["create_date", "tweet_id"], kind="mergesort").reset_index(drop=True)
    return roots.iloc[0]


def write_tree_file(path: Path, edges: list[tuple[list[Any], list[Any]]]) -> None:
    lines = [f"{repr(parent)} -> {repr(child)}" for parent, child in edges]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_fibvid_to_acl2017(cp: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    labels_by_claim = labels.set_index("claim_num")[["group", "label_text"]]
    manifest_rows: list[ClaimConversion] = []
    label_lines: list[str] = []
    tree_dir = ADAPTED_DATA_ROOT / "twitter15" / "tree"

    for claim_number, group in cp.groupby("claim_number", sort=True):
        if claim_number not in labels_by_claim.index:
            continue

        group = group.sort_values(["create_date", "tweet_id"], kind="mergesort").reset_index(drop=True)
        group["order_idx"] = range(len(group))
        root_row = pick_root(group)
        root_tweet_id = str(root_row["tweet_id"])
        root_time = root_row["create_date"]
        root_order = int(root_row["order_idx"])

        by_tweet = (
            group[["tweet_id", "create_date", "order_idx", "tweet_user"]]
            .drop_duplicates(subset=["tweet_id"])
            .set_index("tweet_id")
        )

        edges: list[tuple[list[Any], list[Any]]] = []
        root_delay = 0.0
        edges.append((["ROOT", "ROOT", 0.0], [str(root_row["tweet_user"]), root_tweet_id, root_delay]))

        rewired = 0
        parent_zero = 0
        parent_backward = 0
        parent_missing = 0

        for row in group.itertuples(index=False):
            tweet_id = str(row.tweet_id)
            if tweet_id == root_tweet_id and int(row.order_idx) == root_order:
                continue

            child_delay = max((row.create_date - root_time).total_seconds() / 60.0, 0.0)
            parent_id = str(row.parent_id)
            use_parent = root_tweet_id

            if parent_id == "0":
                parent_zero += 1
                rewired += 1
            elif parent_id not in by_tweet.index:
                parent_missing += 1
                rewired += 1
            else:
                parent_meta = by_tweet.loc[parent_id]
                parent_time = parent_meta["create_date"]
                parent_order = int(parent_meta["order_idx"])
                if (parent_time > row.create_date) or (parent_order >= int(row.order_idx)):
                    parent_backward += 1
                    rewired += 1
                else:
                    use_parent = parent_id

            if use_parent == tweet_id:
                use_parent = root_tweet_id
                rewired += 1

            parent_meta = by_tweet.loc[use_parent]
            parent_delay = max((parent_meta["create_date"] - root_time).total_seconds() / 60.0, 0.0)
            edges.append(
                (
                    [str(parent_meta["tweet_user"]), str(use_parent), round(parent_delay, 6)],
                    [str(row.tweet_user), tweet_id, round(child_delay, 6)],
                )
            )

        label_info = labels_by_claim.loc[claim_number]
        cascade_id = root_tweet_id
        write_tree_file(tree_dir / f"{cascade_id}.txt", edges)
        label_lines.append(f"{label_info['label_text']}:{cascade_id}")
        manifest_rows.append(
            ClaimConversion(
                claim_number=int(claim_number),
                cascade_id=cascade_id,
                label_group=int(label_info["group"]),
                label_text=str(label_info["label_text"]),
                root_tweet_id=root_tweet_id,
                root_time=root_time.isoformat(),
                n_nodes_original=int(len(group)),
                n_nodes_written=int(len(group)),
                n_rewired_to_root=int(rewired),
                n_parent_zero=int(parent_zero),
                n_parent_backward=int(parent_backward),
                n_parent_missing=int(parent_missing),
            )
        )

    (ADAPTED_DATA_ROOT / "twitter15" / "label.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    (ADAPTED_DATA_ROOT / "twitter16" / "label.txt").write_text("", encoding="utf-8")
    manifest = pd.DataFrame([row.__dict__ for row in manifest_rows]).sort_values("claim_number").reset_index(drop=True)
    manifest.to_csv(OUT_ROOT / "logs" / "fibvid_conversion_manifest.csv", index=False)
    return manifest


def summarize_against_old_results() -> None:
    old_dir = ROOT / "thesis_outputs" / "tables"
    new_dir = OUT_ROOT / "tables"
    if not old_dir.exists() or not new_dir.exists():
        return

    old_primary = pd.read_csv(old_dir / "results_k_primary.csv")
    new_primary = pd.read_csv(new_dir / "results_k_primary.csv")
    old_perm = pd.read_csv(old_dir / "permutation_test_k60.csv")
    new_perm = pd.read_csv(new_dir / "permutation_test_k60.csv")

    def best_row(df: pd.DataFrame, task: str, metric: str, feature_set: str) -> pd.Series:
        sub = df[(df["task"] == task) & (df["metric"] == metric) & (df["feature_set"] == feature_set)].copy()
        return sub.sort_values("mean", ascending=False).iloc[0]

    old_v = best_row(old_primary, "veracity", "auc", "interaction_full")
    new_v = best_row(new_primary, "veracity", "auc", "interaction_full")
    old_r = best_row(old_primary, "reach", "r2", "interaction_full")
    new_r = best_row(new_primary, "reach", "r2", "interaction_full")

    old_perm_v = old_perm[old_perm["task"] == "veracity"].iloc[0]
    new_perm_v = new_perm[new_perm["task"] == "veracity"].iloc[0]
    old_perm_r = old_perm[old_perm["task"] == "reach"].iloc[0]
    new_perm_r = new_perm[new_perm["task"] == "reach"].iloc[0]

    lines = [
        "# FibVID vs original main pipeline",
        "",
        "## Peak primary results (interaction_full)",
        "",
        "| task | original best window | original mean | FibVID best window | FibVID mean | delta |",
        "|---|---:|---:|---:|---:|---:|",
        f"| veracity AUC | {int(old_v['window_value'])} | {old_v['mean']:.3f} | {int(new_v['window_value'])} | {new_v['mean']:.3f} | {new_v['mean'] - old_v['mean']:+.3f} |",
        f"| reach R2 | {int(old_r['window_value'])} | {old_r['mean']:.3f} | {int(new_r['window_value'])} | {new_r['mean']:.3f} | {new_r['mean'] - old_r['mean']:+.3f} |",
        "",
        "## Permutation check at K=60",
        "",
        "| task | original observed | FibVID observed | original p | FibVID p |",
        "|---|---:|---:|---:|---:|",
        f"| veracity AUC | {old_perm_v['observed_score']:.3f} | {new_perm_v['observed_score']:.3f} | {old_perm_v['p_value']:.3f} | {new_perm_v['p_value']:.3f} |",
        f"| reach R2 | {old_perm_r['observed_score']:.3f} | {new_perm_r['observed_score']:.3f} | {old_perm_r['p_value']:.3f} | {new_perm_r['p_value']:.3f} |",
        "",
    ]
    (OUT_ROOT / "comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_run_metadata(manifest: pd.DataFrame) -> None:
    meta = {
        "fibvid_root": str(FIBVID_ROOT),
        "adapted_data_root": str(ADAPTED_ROOT),
        "output_root": str(OUT_ROOT),
        "n_claims_converted": int(len(manifest)),
        "n_false": int((manifest["label_text"] == "false").sum()),
        "n_true": int((manifest["label_text"] == "true").sum()),
        "rewired_claims": int((manifest["n_rewired_to_root"] > 0).sum()),
        "mean_rewired_nodes_per_claim": float(manifest["n_rewired_to_root"].mean()),
    }
    (OUT_ROOT / "logs" / "fibvid_run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    cp, labels = load_fibvid_tables()
    manifest = convert_fibvid_to_acl2017(cp, labels)
    write_run_metadata(manifest)
    run_main_pipeline(data_dir=ADAPTED_ROOT, out_dir=OUT_ROOT)
    summarize_against_old_results()
    print(f"FibVID main-pipeline outputs written to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
