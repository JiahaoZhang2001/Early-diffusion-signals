#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thesis_pipeline.cascade_processing import Cascade, parse_cascade, parse_label_file
from thesis_pipeline.config import (
    BASELINE_TIME,
    DYNAMIC_BUNDLE_H,
    RAW_STRUCTURE_SHAPE,
    SEED,
    STRUCTURE_BUNDLE_C,
    TIME_WINDOWS,
)
from thesis_pipeline.feature_engineering import (
    add_structure_tempo_interactions,
    make_time_feature_rows,
    residualize_on_log_volume,
)
from thesis_pipeline.modeling import eval_reach, eval_veracity, get_cv_splits
from thesis_pipeline.plotting_utils import add_n_under_ticks, set_publication_style


DATA_ROOT = ROOT / "new data" / "fibvid_acl2017_like" / "rumor_detection_acl2017"
OUT_ROOT = ROOT / "new data"
OUT_TABLES = OUT_ROOT / "tables"
OUT_FIGURES = OUT_ROOT / "figures"


def load_cascades() -> list[Cascade]:
    datasets = [("twitter15", DATA_ROOT / "twitter15"), ("twitter16", DATA_ROOT / "twitter16")]
    cascades: list[Cascade] = []
    for dname, droot in datasets:
        labels = parse_label_file(droot / "label.txt")
        tree_dir = droot / "tree"
        for cid, label in labels.items():
            tree_path = tree_dir / f"{cid}.txt"
            if not tree_path.exists():
                continue
            cascade, _ = parse_cascade(dname, cid, label, tree_path)
            if cascade is not None:
                cascades.append(cascade)
    return cascades


def resolve_xgb_flag() -> tuple[bool, str]:
    use_xgb = os.environ.get("THESIS_USE_XGBOOST", "0") == "1"
    if not use_xgb:
        return False, "xgboost disabled by env; using GradientBoosting fallback for xgb/xgbreg labels"
    try:
        import xgboost  # noqa: F401

        return True, "xgboost available"
    except Exception:
        return False, "xgboost unavailable; using GradientBoosting fallback for xgb/xgbreg labels"


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_bundle_lines(
    df: pd.DataFrame,
    task: str,
    metric: str,
    title: str,
    ylabel: str,
    stem: str,
    n_map: dict[int, int],
) -> None:
    colors = {
        "baseline": "#1F4E79",
        "structure_only": "#B36A00",
        "full": "#2E7D32",
        "interaction_full": "#7A1F5C",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    lines = ["baseline", "structure_only", "full", "interaction_full"]
    d0 = df[(df["task"] == task) & (df["metric"] == metric)].copy()
    for fs in lines:
        d = d0[d0["feature_set"] == fs].sort_values("window_value")
        x = d["window_value"].to_numpy()
        y = d["mean"].to_numpy()
        se = d["se"].to_numpy()
        ax.errorbar(
            x,
            y,
            yerr=1.96 * se,
            marker="o",
            linewidth=1.8,
            capsize=3.0,
            color=colors[fs],
            label=fs.replace("_", "-"),
        )
    ax.set_title(title)
    ax.set_xlabel("Time window (minutes)")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.2, linestyle=":")
    add_n_under_ticks(ax, sorted(n_map.keys()), n_map)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, stem)


def plot_delta_full_baseline(df: pd.DataFrame, task: str, metric: str, ylabel: str, stem: str) -> pd.DataFrame:
    rows = []
    for t in sorted(df["window_value"].unique()):
        g = df[(df["window_value"] == t) & (df["task"] == task) & (df["metric"] == metric)]
        b = float(g[g["feature_set"] == "baseline"]["mean"].iloc[0])
        f = float(g[g["feature_set"] == "full"]["mean"].iloc[0])
        rows.append({"window_value": int(t), "baseline_mean": b, "full_mean": f, "delta": f - b})
    out = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.bar(out["window_value"].astype(str), out["delta"], color="#4C78A8")
    ax.set_xlabel("Time window (minutes)")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.2, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, stem)
    return out


def curve_stats(df: pd.DataFrame, task: str, metric: str) -> dict[str, float]:
    d = df[(df["task"] == task) & (df["metric"] == metric) & (df["feature_set"] == "full")].sort_values("window_value")
    b = df[(df["task"] == task) & (df["metric"] == metric) & (df["feature_set"] == "baseline")].sort_values("window_value")
    x = d["window_value"].to_numpy(dtype=float)
    y = d["mean"].to_numpy(dtype=float)
    se = d["se"].to_numpy(dtype=float)
    delta = y - b["mean"].to_numpy(dtype=float)
    best_idx = int(np.nanargmax(y))
    trend = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 3 else float("nan")
    return {
        "best_window": float(x[best_idx]),
        "best_score": float(y[best_idx]),
        "avg_ci95_width": float(np.nanmean(1.96 * se * 2.0)),
        "trend_corr": trend,
        "mean_delta_full_baseline": float(np.nanmean(delta)),
    }


def main() -> None:
    set_publication_style()
    rng = np.random.default_rng(SEED)
    xgb_available, xgb_note = resolve_xgb_flag()

    cascades = load_cascades()
    if not cascades:
        raise RuntimeError("No FibVID cascades loaded from harmonized directory.")

    time_df = make_time_feature_rows(cascades, TIME_WINDOWS, rng)
    time_df = residualize_on_log_volume(time_df, RAW_STRUCTURE_SHAPE)
    time_inter_cols = add_structure_tempo_interactions(time_df, STRUCTURE_BUNDLE_C, "early_growth_rate", "int_time")
    time_df = time_df.sort_values(["window_value", "dataset", "cascade_id"]).reset_index(drop=True)

    v_splits_by_t: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    r_splits_by_t: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    n_used_by_t: dict[int, int] = {}
    for t in TIME_WINDOWS:
        dft = time_df[time_df["window_value"] == t].sort_values(["dataset", "cascade_id"]).reset_index(drop=True)
        v_splits_by_t[t] = get_cv_splits(dft, "veracity")
        r_splits_by_t[t] = get_cv_splits(dft, "reach")
        n_used_by_t[t] = len(dft)

    feature_sets = {
        "baseline": BASELINE_TIME,
        "structure_only": STRUCTURE_BUNDLE_C,
        "full": BASELINE_TIME + STRUCTURE_BUNDLE_C + DYNAMIC_BUNDLE_H,
        "interaction_full": BASELINE_TIME + STRUCTURE_BUNDLE_C + DYNAMIC_BUNDLE_H + time_inter_cols,
    }

    rows: list[dict[str, Any]] = []
    for t in TIME_WINDOWS:
        dft = time_df[time_df["window_value"] == t].sort_values(["dataset", "cascade_id"]).reset_index(drop=True)
        for fs_name, features in feature_sets.items():
            auc_m, auc_se = eval_veracity(dft, features, "logit", v_splits_by_t[t], xgb_available)
            rows.append(
                {
                    "task": "veracity",
                    "window_type": "time",
                    "window_value": int(t),
                    "feature_set": fs_name,
                    "model": "logit",
                    "metric": "auc",
                    "mean": auc_m,
                    "se": auc_se,
                    "N_used": len(dft),
                }
            )
            r2_m, r2_se, mae_m, mae_se = eval_reach(dft, features, "ols", r_splits_by_t[t], xgb_available)
            rows.append(
                {
                    "task": "reach",
                    "window_type": "time",
                    "window_value": int(t),
                    "feature_set": fs_name,
                    "model": "ols",
                    "metric": "r2",
                    "mean": r2_m,
                    "se": r2_se,
                    "N_used": len(dft),
                }
            )
            rows.append(
                {
                    "task": "reach",
                    "window_type": "time",
                    "window_value": int(t),
                    "feature_set": fs_name,
                    "model": "ols",
                    "metric": "mae",
                    "mean": mae_m,
                    "se": mae_se,
                    "N_used": len(dft),
                }
            )

    step_time = pd.DataFrame(rows)
    step_time.to_csv(OUT_TABLES / "results_timewin_primary_fibvid.csv", index=False)

    plot_bundle_lines(
        step_time,
        task="veracity",
        metric="auc",
        title="FibVID Veracity (AUC) vs Time-window",
        ylabel="AUC",
        stem="Figure_time_1_veracity_timewin_fibvid",
        n_map=n_used_by_t,
    )
    plot_bundle_lines(
        step_time,
        task="reach",
        metric="r2",
        title="FibVID Reach (R^2) vs Time-window",
        ylabel="R^2",
        stem="Figure_time_2_reach_timewin_fibvid",
        n_map=n_used_by_t,
    )

    d_auc = plot_delta_full_baseline(
        step_time,
        "veracity",
        "auc",
        "ΔAUC (Full - Baseline)",
        "delta_auc_full_baseline_timewin_fibvid",
    )
    d_r2 = plot_delta_full_baseline(
        step_time,
        "reach",
        "r2",
        "ΔR^2 (Full - Baseline)",
        "delta_r2_full_baseline_timewin_fibvid",
    )
    d_auc.to_csv(OUT_TABLES / "delta_auc_full_baseline_timewin_fibvid.csv", index=False)
    d_r2.to_csv(OUT_TABLES / "delta_r2_full_baseline_timewin_fibvid.csv", index=False)

    reps = [30, 60, 180]
    rep_rows = []
    for t in reps:
        g_v = step_time[(step_time["window_value"] == t) & (step_time["task"] == "veracity") & (step_time["metric"] == "auc")]
        g_r = step_time[(step_time["window_value"] == t) & (step_time["task"] == "reach") & (step_time["metric"] == "r2")]
        rep_rows.append(
            {
                "T_minutes": t,
                "AUC_baseline": float(g_v[g_v["feature_set"] == "baseline"]["mean"].iloc[0]),
                "AUC_full": float(g_v[g_v["feature_set"] == "full"]["mean"].iloc[0]),
                "Delta_AUC": float(g_v[g_v["feature_set"] == "full"]["mean"].iloc[0] - g_v[g_v["feature_set"] == "baseline"]["mean"].iloc[0]),
                "R2_baseline": float(g_r[g_r["feature_set"] == "baseline"]["mean"].iloc[0]),
                "R2_full": float(g_r[g_r["feature_set"] == "full"]["mean"].iloc[0]),
                "Delta_R2": float(g_r[g_r["feature_set"] == "full"]["mean"].iloc[0] - g_r[g_r["feature_set"] == "baseline"]["mean"].iloc[0]),
            }
        )
    pd.DataFrame(rep_rows).to_csv(OUT_TABLES / "timewin_selected_baseline_vs_full_fibvid.csv", index=False)

    full_features_time = feature_sets["interaction_full"]
    model_rows = []
    for t in TIME_WINDOWS:
        dft = time_df[time_df["window_value"] == t].sort_values(["dataset", "cascade_id"]).reset_index(drop=True)
        for m in ["logit", "rf", "xgb"]:
            auc_m, auc_se = eval_veracity(dft, full_features_time, m, v_splits_by_t[t], xgb_available)
            model_rows.append({"task": "veracity", "window_minutes": int(t), "model": m, "metric": "auc", "mean": auc_m, "se": auc_se, "N_used": len(dft)})
        for m in ["ols", "rfreg", "xgbreg"]:
            r2_m, r2_se, mae_m, mae_se = eval_reach(dft, full_features_time, m, r_splits_by_t[t], xgb_available)
            model_rows.append({"task": "reach", "window_minutes": int(t), "model": m, "metric": "r2", "mean": r2_m, "se": r2_se, "N_used": len(dft)})
            model_rows.append({"task": "reach", "window_minutes": int(t), "model": m, "metric": "mae", "mean": mae_m, "se": mae_se, "N_used": len(dft)})
    model_df = pd.DataFrame(model_rows)
    model_df.to_csv(OUT_TABLES / "model_family_comparison_timewin_full_fibvid.csv", index=False)

    fig_m, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    colors = {"logit": "#1F4E79", "rf": "#B36A00", "xgb": "#2E7D32", "ols": "#1F4E79", "rfreg": "#B36A00", "xgbreg": "#2E7D32"}
    for m in ["logit", "rf", "xgb"]:
        d = model_df[(model_df["task"] == "veracity") & (model_df["metric"] == "auc") & (model_df["model"] == m)].sort_values("window_minutes")
        axes[0].errorbar(d["window_minutes"], d["mean"], yerr=1.96 * d["se"], marker="o", linewidth=1.8, capsize=3.0, color=colors[m], label=m)
    for m in ["ols", "rfreg", "xgbreg"]:
        d = model_df[(model_df["task"] == "reach") & (model_df["metric"] == "r2") & (model_df["model"] == m)].sort_values("window_minutes")
        axes[1].errorbar(d["window_minutes"], d["mean"], yerr=1.96 * d["se"], marker="o", linewidth=1.8, capsize=3.0, color=colors[m], label=m)
    axes[0].set_title("A: FibVID Veracity AUC (Full Bundle, Time-window)")
    axes[1].set_title("B: FibVID Reach R^2 (Full Bundle, Time-window)")
    axes[0].set_ylabel("AUC")
    axes[1].set_ylabel("R^2")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.2, linestyle=":")
        ax.set_xlabel("Time window (minutes)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
    fig_m.tight_layout()
    save_fig(fig_m, "model_family_full_only_timewin_fibvid")

    gain_rows = []
    for t in TIME_WINDOWS:
        dft = time_df[time_df["window_value"] == t].sort_values(["dataset", "cascade_id"]).reset_index(drop=True)
        for m in ["logit", "rf", "xgb"]:
            b, _ = eval_veracity(dft, BASELINE_TIME, m, v_splits_by_t[t], xgb_available)
            f = float(model_df[(model_df["task"] == "veracity") & (model_df["metric"] == "auc") & (model_df["model"] == m) & (model_df["window_minutes"] == t)]["mean"].iloc[0])
            gain_rows.append({"task": "veracity", "window_minutes": int(t), "model": m, "metric": "auc", "baseline_mean": b, "full_mean": f, "delta": f - b})
        for m in ["ols", "rfreg", "xgbreg"]:
            b, _, _, _ = eval_reach(dft, BASELINE_TIME, m, r_splits_by_t[t], xgb_available)
            f = float(model_df[(model_df["task"] == "reach") & (model_df["metric"] == "r2") & (model_df["model"] == m) & (model_df["window_minutes"] == t)]["mean"].iloc[0])
            gain_rows.append({"task": "reach", "window_minutes": int(t), "model": m, "metric": "r2", "baseline_mean": b, "full_mean": f, "delta": f - b})
    gain_df = pd.DataFrame(gain_rows)
    gain_df.to_csv(OUT_TABLES / "delta_gain_by_model_timewin_fibvid.csv", index=False)

    fig_g, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    x = np.arange(len(TIME_WINDOWS))
    width = 0.22
    vdf = gain_df[gain_df["task"] == "veracity"]
    for i, m in enumerate(["logit", "rf", "xgb"]):
        d = vdf[vdf["model"] == m].sort_values("window_minutes")
        axes[0].bar(x + (i - 1) * width, d["delta"].to_numpy(), width=width, label=m)
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(t) for t in TIME_WINDOWS])
    axes[0].set_title("A: ΔAUC (Full - Baseline, FibVID Time-window)")
    axes[0].set_xlabel("Time window (minutes)")
    axes[0].set_ylabel("ΔAUC")
    axes[0].legend(frameon=False)

    rdf = gain_df[gain_df["task"] == "reach"]
    for i, m in enumerate(["ols", "rfreg", "xgbreg"]):
        d = rdf[rdf["model"] == m].sort_values("window_minutes")
        axes[1].bar(x + (i - 1) * width, d["delta"].to_numpy(), width=width, label=m)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(t) for t in TIME_WINDOWS])
    axes[1].set_title("B: ΔR^2 (Full - Baseline, FibVID Time-window)")
    axes[1].set_xlabel("Time window (minutes)")
    axes[1].set_ylabel("ΔR^2")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.2, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig_g.tight_layout()
    save_fig(fig_g, "delta_gain_by_model_timewin_fibvid")

    kdf = pd.read_csv(OUT_TABLES / "results_k_primary.csv")
    tdf = step_time.copy()
    k_ver = curve_stats(kdf, "veracity", "auc")
    t_ver = curve_stats(tdf, "veracity", "auc")
    k_rea = curve_stats(kdf, "reach", "r2")
    t_rea = curve_stats(tdf, "reach", "r2")

    primary = "k-window"
    core_reason = "it still delivers the stronger reach peak while holding observed information quantity fixed."
    if (t_rea["best_score"] - k_rea["best_score"] > 0.03) and (t_rea["avg_ci95_width"] <= k_rea["avg_ci95_width"]):
        primary = "time-window"
        core_reason = "it yields materially stronger and similarly stable reach performance on FibVID."

    lines = [
        "# FibVID Time-window vs K-window Summary",
        "",
        f"- Same seed and splitter logic were used (`SEED={SEED}`); XGBoost setting: {xgb_note}.",
        f"- Veracity full AUC peak: k-window={k_ver['best_score']:.3f} at K={int(k_ver['best_window'])}; time-window={t_ver['best_score']:.3f} at T={int(t_ver['best_window'])}.",
        f"- Reach full R^2 peak: k-window={k_rea['best_score']:.3f} at K={int(k_rea['best_window'])}; time-window={t_rea['best_score']:.3f} at T={int(t_rea['best_window'])}.",
        f"- Mean full-minus-baseline gain: veracity k={k_ver['mean_delta_full_baseline']:+.3f} vs time={t_ver['mean_delta_full_baseline']:+.3f}; reach k={k_rea['mean_delta_full_baseline']:+.3f} vs time={t_rea['mean_delta_full_baseline']:+.3f}.",
        f"- Suggested primary on FibVID: {primary}.",
        f"- Reason: {core_reason}",
    ]
    (OUT_ROOT / "fibvid_timewin_vs_kwin_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"FibVID suggested primary = {primary} | {core_reason}")


if __name__ == "__main__":
    main()
