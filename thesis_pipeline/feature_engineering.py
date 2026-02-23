from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from .cascade_processing import build_subgraph_adj, compute_structure_features, Cascade

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

