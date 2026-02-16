#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TIME_WINDOWS = [30, 60, 180]
MIN_NODES = 2

ROOT = Path('.')
OUT = ROOT / 'thesis_outputs'
DATA_PATH = OUT / 'data' / 'twitter1516_features_time.csv'
RESULTS_TIME = OUT / 'tables' / 'results_time.csv'
SEARCH_TABLE = OUT / 'tables' / 'structure_search_time.csv'
LOG_PATH = OUT / 'logs' / 'structure_bundle_search.txt'
FIG6_PNG = OUT / 'figures' / 'F6_structure_bundle_comparison.png'
FIG6_PDF = OUT / 'figures' / 'F6_structure_bundle_comparison.pdf'


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-9
    df['norm_depth'] = df['early_depth'] / np.log1p(df['early_n_nodes'].clip(lower=1))
    df['norm_width'] = df['early_width'] / df['early_n_nodes'].clip(lower=1)
    df['depth_to_width'] = df['early_depth'] / (1.0 + df['early_width'])
    if 'internal_node_share' not in df.columns:
        df['internal_node_share'] = 1.0 - df['leaf_fraction']
    if 'avg_root_distance' not in df.columns:
        # fallback: exact value should exist from make_thesis_analysis.py update.
        df['avg_root_distance'] = df['early_depth'] / 2.0
    df['interaction_n_norm_depth'] = df['early_n_nodes'] * df['norm_depth']
    df['interaction_growth_branch'] = df['early_growth_rate'] * df['branching_factor']
    return df


def get_feature_sets() -> Tuple[List[str], Dict[str, List[str]], List[str]]:
    baseline = ['early_n_nodes', 'early_growth_rate', 'time_to_10', 'time_to_20']
    current_structure = [
        'early_depth',
        'avg_root_distance',
        'early_width',
        'branching_factor',
        'internal_node_share',
        'leaf_fraction',
        'outdegree_entropy',
        'structural_virality_proxy',
    ]
    bundles = {
        'A': ['norm_depth', 'norm_width', 'leaf_fraction', 'avg_root_distance'],
        'B': ['depth_to_width', 'internal_node_share', 'outdegree_entropy', 'leaf_fraction'],
        'C': ['structural_virality_proxy', 'avg_root_distance', 'leaf_fraction'],
        'D': [
            'norm_depth', 'norm_width', 'leaf_fraction', 'avg_root_distance',
            'interaction_n_norm_depth', 'interaction_growth_branch'
        ],
    }
    return baseline, bundles, current_structure


def make_folds_binary(y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return [(tr, te) for tr, te in skf.split(np.zeros(len(y)), y)]


def make_folds_reg(n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    idx = np.arange(n)
    return [(tr, te) for tr, te in kf.split(idx)]


def fit_auc_with_folds(X: np.ndarray, y: np.ndarray, folds, model_type: str) -> np.ndarray:
    fold_scores = []
    for tr, te in folds:
        if model_type == 'logistic':
            model = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=3000, random_state=SEED)),
            ])
        elif model_type == 'rf':
            model = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('clf', RandomForestClassifier(n_estimators=500, class_weight='balanced_subsample', random_state=SEED, n_jobs=-1)),
            ])
        else:
            raise ValueError(model_type)

        model.fit(X[tr], y[tr])
        prob = model.predict_proba(X[te])[:, 1]
        fold_scores.append(roc_auc_score(y[te], prob))
    return np.array(fold_scores)


def fit_r2_with_folds(X: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    scores = []
    for tr, te in folds:
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('reg', LinearRegression()),
        ])
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        scores.append(r2_score(y[te], pred))
    return np.array(scores)


def save_f6(search_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    colors = {'A': '#4C78A8', 'B': '#F58518', 'C': '#54A24B', 'D': '#B279A2'}

    for ax, model in zip(axes, ['logistic', 'rf']):
        sub = search_df[search_df['model_family'] == model]
        for b in ['A', 'B', 'C', 'D']:
            s = sub[sub['bundle'] == b].sort_values('window_value')
            ax.errorbar(
                s['window_value'], s['delta_auc_mean'], yerr=s['delta_auc_std'],
                marker='o', linewidth=2, label=f'Bundle {b}', color=colors[b]
            )
        ax.axhline(0, linestyle='--', color='#666666', linewidth=1)
        ax.set_title(f'{model.upper()}')
        ax.set_xlabel('Time window (minutes)')
        ax.grid(True, axis='y', alpha=0.25)

    axes[0].set_ylabel('Delta AUC vs baseline')
    axes[1].legend(loc='best')
    fig.suptitle('F6: Structure Bundle Comparison (Delta AUC)')
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(FIG6_PNG, dpi=300, bbox_inches='tight')
    fig.savefig(FIG6_PDF, bbox_inches='tight')
    plt.close(fig)


def save_bestbundle_figs(best_bundle: str, full_auc: pd.DataFrame, reach_df: pd.DataFrame) -> None:
    # F1 best bundle (veracity AUC, logistic)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    colors = {'baseline': '#4C78A8', 'full': '#54A24B'}

    for fs, label in [('baseline', 'Baseline'), ('full', f'Full (Bundle {best_bundle})')]:
        sub = full_auc[full_auc['feature_set'] == fs].sort_values('window_value')
        ax.plot(sub['window_value'], sub['auc_mean'], marker='o', linewidth=2, color=colors[fs], label=label)
        ax.fill_between(sub['window_value'], sub['auc_mean'] - sub['auc_std'], sub['auc_mean'] + sub['auc_std'], color=colors[fs], alpha=0.15)

    benchmark = float(full_auc[full_auc['feature_set'] == 'full']['auc_mean'].mean())
    ax.axhline(benchmark, linestyle='--', color='#666666', linewidth=1.2)
    ax.text(182, benchmark, f'Overall Full={benchmark:.3f}', va='center', ha='left', fontsize=10)
    ax.set_title(f'F1 (Best Bundle {best_bundle}): Time-window Veracity AUC')
    ax.set_xlabel('Time window (minutes)')
    ax.set_ylabel('AUC (5-fold CV)')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(OUT / 'figures' / 'F1_time_veracity_auc_bestbundle.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / 'figures' / 'F1_time_veracity_auc_bestbundle.pdf', bbox_inches='tight')
    plt.close(fig)

    # F4 best bundle decomposition (AUC + R2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    windows = [30, 60, 180]
    x = np.arange(len(windows))
    width = 0.25

    # AUC panel
    auc_plot = full_auc.copy()
    auc_plot = auc_plot[auc_plot['feature_set'].isin(['baseline', 'structure_only', 'full'])]
    for i, fs in enumerate(['baseline', 'structure_only', 'full']):
        s = auc_plot[auc_plot['feature_set'] == fs].sort_values('window_value')
        axes[0].bar(x + (i - 1) * width, s['auc_mean'].values, width, label=fs)
    axes[0].set_title('Panel A: Veracity AUC')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(w) for w in windows])
    axes[0].set_xlabel('Time window (minutes)')
    axes[0].set_ylabel('AUC')
    axes[0].grid(True, axis='y', alpha=0.25)

    # R2 panel
    for i, fs in enumerate(['baseline', 'structure_only', 'full']):
        s = reach_df[reach_df['feature_set'] == fs].sort_values('window_value')
        axes[1].bar(x + (i - 1) * width, s['r2_mean'].values, width, label=fs)
    axes[1].set_title('Panel B: Reach R²')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(w) for w in windows])
    axes[1].set_xlabel('Time window (minutes)')
    axes[1].set_ylabel('R²')
    axes[1].grid(True, axis='y', alpha=0.25)
    axes[1].legend(loc='best')

    fig.suptitle(f'F4 (Best Bundle {best_bundle}): Contribution Decomposition')
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(OUT / 'figures' / 'F4_contribution_decomposition_bestbundle.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / 'figures' / 'F4_contribution_decomposition_bestbundle.pdf', bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'Missing features file: {DATA_PATH}')

    df = pd.read_csv(DATA_PATH)
    df = df[df['early_n_nodes'] >= MIN_NODES].copy()
    df = add_derived_features(df)

    baseline, bundles, current_structure = get_feature_sets()

    log_lines = []
    log_lines.append('Structure bundle search for veracity AUC (binary y_false)')
    log_lines.append(f'Input table: {DATA_PATH}')
    log_lines.append(f'Windows: {TIME_WINDOWS}')
    log_lines.append('Baseline features (unchanged): ' + ', '.join(baseline))
    log_lines.append('Current structure features (from prior run): ' + ', '.join(current_structure))
    for b, cols in bundles.items():
        log_lines.append(f'Bundle {b}: ' + ', '.join(cols))

    all_rows = []
    best_picker = []

    for T in TIME_WINDOWS:
        sub = df[df['window_value'] == T].copy()
        y = sub['y_false'].astype(int).values
        folds = make_folds_binary(y)

        X_base = sub[baseline].values
        base_scores = {
            'logistic': fit_auc_with_folds(X_base, y, folds, 'logistic'),
            'rf': fit_auc_with_folds(X_base, y, folds, 'rf'),
        }

        for model in ['logistic', 'rf']:
            s = base_scores[model]
            all_rows.append({
                'window_type': 'time',
                'window_value': T,
                'model_family': model,
                'bundle': 'baseline',
                'auc_mean': float(np.mean(s)),
                'auc_std': float(np.std(s)),
                'delta_auc_mean': 0.0,
                'delta_auc_std': 0.0,
                'N_used': len(sub),
            })

        for bname, bcols in bundles.items():
            full_cols = baseline + bcols
            X_full = sub[full_cols].values
            for model in ['logistic', 'rf']:
                fs = fit_auc_with_folds(X_full, y, folds, model)
                delta = fs - base_scores[model]
                row = {
                    'window_type': 'time',
                    'window_value': T,
                    'model_family': model,
                    'bundle': bname,
                    'auc_mean': float(np.mean(fs)),
                    'auc_std': float(np.std(fs)),
                    'delta_auc_mean': float(np.mean(delta)),
                    'delta_auc_std': float(np.std(delta)),
                    'N_used': len(sub),
                }
                all_rows.append(row)
                if model == 'logistic':
                    best_picker.append(row)

    search_df = pd.DataFrame(all_rows)
    search_df.to_csv(SEARCH_TABLE, index=False)

    save_f6(search_df[search_df['bundle'] != 'baseline'])

    improved = bool((search_df[(search_df['bundle'] != 'baseline')]['delta_auc_mean'] > 0).any())
    log_lines.append(f'Any improvement over baseline: {improved}')

    # consistency notes
    for b in ['A', 'B', 'C', 'D']:
        bsub = search_df[(search_df['bundle'] == b) & (search_df['model_family'] == 'logistic')].sort_values('window_value')
        signs = np.sign(bsub['delta_auc_mean'].values)
        consistent = bool(np.all(signs >= 0) or np.all(signs <= 0))
        log_lines.append(f'Bundle {b} logistic deltas by window: {list(np.round(bsub.delta_auc_mean.values,4))}; consistent_sign={consistent}')

    log_lines.append('Rationale: normalization can reduce scale collinearity with early_n_nodes; ratio features encode shape independent of size;')
    log_lines.append('virality-focused bundle targets global structure signal; interaction terms add controlled nonlinearity for volume-geometry coupling.')

    if improved:
        pick = pd.DataFrame(best_picker)
        # best by AUC@60, tie-break by AUC@30 (logistic)
        p60 = pick[pick['window_value'] == 60][['bundle', 'auc_mean']]
        p30 = pick[pick['window_value'] == 30][['bundle', 'auc_mean']].rename(columns={'auc_mean': 'auc30'})
        rank = p60.merge(p30, on='bundle', how='left').sort_values(['auc_mean', 'auc30'], ascending=False)
        best_bundle = rank.iloc[0]['bundle']
        log_lines.append(f'Best bundle selected (logistic): {best_bundle}')

        # Build F1/F4 bestbundle using logistic for veracity and linear regression for reach with same folds per window
        auc_rows = []
        r2_rows = []
        for T in TIME_WINDOWS:
            sub = df[df['window_value'] == T].copy()
            y_bin = sub['y_false'].astype(int).values
            folds_cls = make_folds_binary(y_bin)
            y_reg = sub['reach_target'].values
            folds_reg = make_folds_reg(len(sub))

            bcols = bundles[best_bundle]
            sets = {
                'baseline': baseline,
                'structure_only': bcols,
                'full': baseline + bcols,
            }

            for fs, cols in sets.items():
                auc_scores = fit_auc_with_folds(sub[cols].values, y_bin, folds_cls, 'logistic')
                auc_rows.append({'window_value': T, 'feature_set': fs, 'auc_mean': float(np.mean(auc_scores)), 'auc_std': float(np.std(auc_scores))})

                r2_scores = fit_r2_with_folds(sub[cols].values, y_reg, folds_reg)
                r2_rows.append({'window_value': T, 'feature_set': fs, 'r2_mean': float(np.mean(r2_scores)), 'r2_std': float(np.std(r2_scores))})

        auc_df = pd.DataFrame(auc_rows)
        r2_df = pd.DataFrame(r2_rows)
        save_bestbundle_figs(best_bundle, auc_df, r2_df)

    LOG_PATH.write_text('\n'.join(log_lines), encoding='utf-8')

    print(f'Saved: {SEARCH_TABLE}')
    print(f'Saved: {FIG6_PNG}')
    print(f'Saved: {FIG6_PDF}')
    print(f'Updated log: {LOG_PATH}')
    if improved:
        print('Saved: thesis_outputs/figures/F1_time_veracity_auc_bestbundle.png/.pdf')
        print('Saved: thesis_outputs/figures/F4_contribution_decomposition_bestbundle.png/.pdf')


if __name__ == '__main__':
    main()
