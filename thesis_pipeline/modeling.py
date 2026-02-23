import math
from typing import List, Sequence, Tuple

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

from .config import SEED

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


def mean_and_se(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def eval_veracity_scores(
    df: pd.DataFrame,
    features: List[str],
    model_name: str,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    xgb_available: bool,
) -> np.ndarray:
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
    return np.asarray(scores, dtype=float)


def eval_reach_scores(
    df: pd.DataFrame,
    features: List[str],
    model_name: str,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    xgb_available: bool,
) -> Tuple[np.ndarray, np.ndarray]:
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
    return np.asarray(r2s, dtype=float), np.asarray(maes, dtype=float)


def eval_veracity(df: pd.DataFrame, features: List[str], model_name: str, splits: List[Tuple[np.ndarray, np.ndarray]], xgb_available: bool) -> Tuple[float, float]:
    scores = eval_veracity_scores(df, features, model_name, splits, xgb_available)
    return mean_and_se(scores)


def eval_reach(df: pd.DataFrame, features: List[str], model_name: str, splits: List[Tuple[np.ndarray, np.ndarray]], xgb_available: bool) -> Tuple[float, float, float, float]:
    r2s, maes = eval_reach_scores(df, features, model_name, splits, xgb_available)
    r2_mean, r2_se = mean_and_se(r2s)
    mae_mean, mae_se = mean_and_se(maes)
    return r2_mean, r2_se, mae_mean, mae_se


def paired_t_test(x0: Sequence[float], x1: Sequence[float]) -> Tuple[float, float, float, float]:
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    n = min(x0.size, x1.size)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    d = (x1[:n] - x0[:n]).astype(float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    delta = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    if sd <= 1e-12:
        t_stat = 0.0 if abs(delta) <= 1e-12 else float(np.sign(delta) * 1e6)
        p_val = 1.0 if abs(delta) <= 1e-12 else 0.0
        return delta, t_stat, p_val, float("nan")
    t_stat = delta / (sd / np.sqrt(d.size))
    try:
        from scipy.stats import t as t_dist

        p_val = float(2.0 * t_dist.sf(abs(t_stat), df=d.size - 1))
    except Exception:
        p_val = float(math.erfc(abs(t_stat) / math.sqrt(2.0)))
    effect = delta / sd
    return delta, float(t_stat), p_val, float(effect)
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
