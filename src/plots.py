"""Plots: equity curve, parameter surface, PnL diagnostics."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def equity_curve(equity: pd.Series, path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity.index, equity.values, lw=0.8)
    ax.set_title(title)
    ax.set_ylabel("equity (USDT)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parameter_surface(df: pd.DataFrame, path, title: str) -> None:
    """Heatmap mean window return: spacing x range (best over other params)."""
    if df.empty:
        return
    piv = (df.groupby(["spacing", "range_pct"])["mean_return_pct"]
             .mean().unstack() * 100)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [f"{c:.1%}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{i:.2%}" for i in piv.index])
    ax.set_xlabel("range_pct")
    ax.set_ylabel("grid_spacing_pct")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title(title + " (mean return %/window)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def scatter(x: pd.Series, y: pd.Series, path, xlabel: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(np.asarray(x), np.asarray(y), s=14, alpha=0.7)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
