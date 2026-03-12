#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "merry555-FibVID-14b95c3" / "robustness_fibvid_out"
OUT = ROOT / "new data" / "fibvid_native_raw"


FILES_TO_COPY = [
    "robustness_results_fixed.csv",
    "robustness_report_fixed.md",
    "spread_results_fixed.csv",
    "spread_report_fixed.md",
    "feature_table_fixed_preview.csv",
    "spread_feature_preview.csv",
    "spread_pred_vs_true_W180_size.png",
]


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def copy_outputs() -> None:
    for name in FILES_TO_COPY:
        src = SRC / name
        if src.exists():
            shutil.copy2(src, OUT / name)


def build_summary() -> None:
    ver = pd.read_csv(SRC / "robustness_results_fixed.csv")
    spr = pd.read_csv(SRC / "spread_results_fixed.csv")

    best_ver = ver.sort_values("auc_mean", ascending=False).iloc[0]
    best_size = spr[spr["outcome"] == "final_size_log1p"].sort_values("r2_mean", ascending=False).iloc[0]
    best_rt = spr[spr["outcome"] == "final_retweets_log1p"].sort_values("r2_mean", ascending=False).iloc[0]

    lines = [
        "# FibVID Native Raw Analysis Summary",
        "",
        "This directory summarizes the existing non-harmonized FibVID analysis, using claim-level propagation records directly rather than converting them into single-root cascade trees.",
        "",
        "## Setup",
        "- Early windows: W30 / W60 / W180 minutes",
        "- Veracity label mapping: fake={1,3}, real={0,2}",
        "- Modeling filter: claims with n_nodes >= 2 inside the early window",
        "- Features are native claim-level aggregates: n_nodes, unique users, depth summaries, n_roots, likes/retweets aggregates, node rate, and branching proxy",
        "",
        "## Best Native Results",
        f"- Veracity best AUC: {best_ver['auc_mean']:.3f} at {best_ver['window_name']} with {best_ver['model']}",
        f"- Final size best R^2: {best_size['r2_mean']:.3f} at {best_size['window_name']} with {best_size['model']}",
        f"- Final retweets best R^2: {best_rt['r2_mean']:.3f} at {best_rt['window_name']} with {best_rt['model']}",
        "",
        "## Interpretation",
        "- Native raw FibVID is weak for veracity: AUC stays close to chance and full feature bundles do not improve reliably over the simple baseline.",
        "- Native raw FibVID provides only modest spread signal: outcome R^2 is positive but far below the harmonized k-window analysis.",
        "- This makes the native analysis useful as a sensitivity check, but not a strong replacement for the harmonized cascade-based pipeline.",
        "",
        "## Files",
        "- `robustness_results_fixed.csv`: native veracity results",
        "- `spread_results_fixed.csv`: native spread/outcome regression results",
        "- `robustness_report_fixed.md`: native veracity report",
        "- `spread_report_fixed.md`: native spread report",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_out()
    copy_outputs()
    build_summary()
    print(f"Collected native FibVID outputs into: {OUT}")


if __name__ == "__main__":
    main()
