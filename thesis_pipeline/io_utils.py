from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config


def save_figure(fig: plt.Figure, stem: str) -> List[str]:
    png = config.OUT_FIGURES / f"{stem}.png"
    pdf = config.OUT_FIGURES / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [str(png), str(pdf)]


def write_caption(stem: str, text: str) -> str:
    path = config.OUT_CAPTIONS / f"{stem}.txt"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return str(path)
