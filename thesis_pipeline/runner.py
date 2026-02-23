import json
import os
import platform
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import config
from .cascade_processing import Cascade, parse_cascade, parse_label_file
from .config import (
    BASELINE_K,
    BASELINE_TIME,
    DYNAMIC_BUNDLE_H,
    K_WINDOWS,
    LABELS,
    RAW_STRUCTURE_SHAPE,
    RNG,
    SEED,
    STRUCTURE_BUNDLE_C,
    T_CURVE,
    TIME_WINDOWS,
    TUNED_TREE_CLS_PARAMS,
    TUNED_TREE_REG_PARAMS,
)
from .feature_engineering import (
    add_structure_tempo_interactions,
    make_k_feature_rows,
    make_time_feature_rows,
    residualize_on_log_volume,
    volume_dependence_diagnostics,
)
from .io_utils import save_figure, write_caption
from .modeling import (
    eval_reach,
    eval_reach_scores,
    eval_veracity,
    eval_veracity_scores,
    get_cv_splits,
    make_preprocessor,
    mean_and_se,
    paired_t_test,
    permutation_test,
)
from .plotting_utils import add_n_under_ticks, plot_three_lines, set_publication_style

def main(data_dir: Path, out_dir: Path) -> None:
    config.configure_paths(data_dir, out_dir)
    config.ensure_dirs()
    set_publication_style()
    created_files: List[str] = []
    run_log_lines: List[str] = []
    exclusions: List[str] = []

    run_log_lines.append(f"Run start: {datetime.now().isoformat(timespec='seconds')}")

    # STEP 0 Parse + audit
    datasets = [("twitter15", config.DATA_ROOT / "twitter15"), ("twitter16", config.DATA_ROOT / "twitter16")]
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
    p_audit = config.OUT_TABLES / "dataset_audit.csv"
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
    p_audit_txt = config.OUT_LOGS / "data_audit.txt"
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
    p_vol_diag = config.OUT_TABLES / "volume_dependence_diagnostics.csv"
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
    p_step1 = config.OUT_TABLES / "results_k_primary.csv"
    step1_df.to_csv(p_step1, index=False)
    created_files.append(str(p_step1))
    p_step1_test = config.OUT_TABLES / "k_primary_paired_ttests.csv"
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
    p_step2 = config.OUT_TABLES / "model_family_comparison_k_full.csv"
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
    p_step3 = config.OUT_TABLES / "delta_gain_by_model_k.csv"
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
    p_perm_summary = config.OUT_TABLES / "permutation_test_k60.csv"
    perm_summary.to_csv(p_perm_summary, index=False)
    created_files.append(str(p_perm_summary))

    perm_null_rows = []
    for i, s in enumerate(null_auc):
        perm_null_rows.append({"task": "veracity", "metric": "auc", "perm_idx": i, "null_score": float(s)})
    for i, s in enumerate(null_r2):
        perm_null_rows.append({"task": "reach", "metric": "r2", "perm_idx": i, "null_score": float(s)})
    perm_null_df = pd.DataFrame(perm_null_rows)
    p_perm_null = config.OUT_TABLES / "permutation_null_scores_k60.csv"
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
    p_lc = config.OUT_TABLES / "learning_curve_time.csv"
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
    p_step5 = config.OUT_TABLES / "results_k_main.csv"
    step5_df.to_csv(p_step5, index=False)
    created_files.append(str(p_step5))

    cov_df = pd.DataFrame(cov_rows)
    p_cov = config.OUT_TABLES / "coverage_k.csv"
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
    p_tree_eval = config.OUT_TABLES / "tree_tuned_k60.csv"
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
        "cwd": str(config.ROOT),
    }
    p_run_config = config.OUT_LOGS / "run_config.json"
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

    p_run_log = config.OUT_LOGS / "run_log.txt"
    p_run_log.write_text("\n".join(run_log_lines) + "\n", encoding="utf-8")
    created_files.append(str(p_run_log))

    print("Generated outputs:")
    for p in sorted(set(created_files)):
        print(p)

