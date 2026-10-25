"""Estilo compartilhado dos gráficos.

Paleta categórica validada (slots em ordem fixa, nunca cicladas) e tinta de
gráfico para superfície clara. Cores entram por papel, não por gosto.
"""

from __future__ import annotations

import matplotlib as mpl

# Paleta categórica — ordem fixa
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

# Tinta e cromo
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Papéis semânticos
MODEL = MUTED  # curva teórica / referência
CRITICAL = "#d03b3b"  # marcação do ótimo


def apply() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Helvetica Neue", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": MUTED,
            "ytick.labelcolor": MUTED,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def label_end(ax, x, y, text, color, dx=1.06, fontsize=9):
    """Rótulo direto no fim da série — identidade sem depender só da cor."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(x * dx, y),
        color=color,
        fontsize=fontsize,
        fontweight="bold",
        va="center",
        ha="left",
        annotation_clip=False,
    )
