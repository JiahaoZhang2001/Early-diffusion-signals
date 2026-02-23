from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def add_n_under_ticks(ax: plt.Axes, xvals: Sequence[int], n_map: Dict[int, int]) -> None:
    ax.set_xticks(list(xvals))
    labels = [f"{x}\nN={n_map.get(int(x), 0)}" for x in xvals]
    ax.set_xticklabels(labels)


def set_publication_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 120,
        }
    )


def plot_three_lines(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    se_col: str,
    lines: Sequence[str],
    title: str,
    ylabel: str,
    chance_line: Optional[float],
    n_map: Dict[int, int],
) -> plt.Figure:
    colors = {
        "baseline": "#1F4E79",
        "structure_only": "#B36A00",
        "full": "#2E7D32",
        "interaction_full": "#7A1F5C",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    xvals = sorted(df[x_col].unique())
    for line in lines:
        d = df[df["feature_set"] == line].sort_values(x_col)
        x = d[x_col].to_numpy()
        y = d[y_col].to_numpy()
        se = d[se_col].to_numpy()
        ci95 = 1.96 * se
        ax.errorbar(
            x,
            y,
            yerr=ci95,
            marker="o",
            linewidth=1.8,
            capsize=3.5,
            markersize=4.5,
            label=line.replace("_", "-"),
            color=colors.get(line, None),
        )
    if chance_line is not None:
        ax.axhline(chance_line, color="#666666", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Window")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.2, linestyle=":")
    add_n_under_ticks(ax, xvals, n_map)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig

