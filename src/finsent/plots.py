"""Graphiques du projet (matplotlib), avec une palette sobre et lisible en daltonisme."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POS = "#2a78d6"     # bleu  : sentiment positif / stratégie
NEG = "#eb6834"     # orange: sentiment négatif
NEUTRAL = "#8a8984" # gris  : neutre / référence
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def setup_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "font.size": 10,
        }
    )


def _save(fig, path):
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")


def confusion(cm: np.ndarray, labels, title: str, path=None):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.grid(False)
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.6 else INK, fontsize=9)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Vrai")
    ax.set_title(title)
    _save(fig, path)
    return fig


def score_distribution(scores: pd.Series, path=None):
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bins = np.linspace(-1, 1, 41)
    ax.hist(scores[scores < -0.05], bins=bins, color=NEG, label="négatif", rwidth=0.9)
    ax.hist(scores[scores.abs() <= 0.05], bins=bins, color=NEUTRAL, label="neutre", rwidth=0.9)
    ax.hist(scores[scores > 0.05], bins=bins, color=POS, label="positif", rwidth=0.9)
    ax.set_xlabel("Score de sentiment  P(positif) − P(négatif)")
    ax.set_ylabel("Nombre de titres")
    ax.set_title("Distribution du sentiment des titres")
    ax.legend()
    _save(fig, path)
    return fig


def quantile_bars(q: pd.DataFrame, path=None):
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    colors = [NEG, NEG, NEUTRAL, POS, POS][: len(q)] if len(q) == 5 else NEUTRAL
    bps = q["mean"] * 1e4
    ax.bar(q["bin"].astype(str), bps, yerr=q["sem"] * 1e4 * 1.96, color=colors,
           width=0.6, capsize=4, error_kw={"ecolor": INK_2, "lw": 1})
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_xlabel("Quintile de sentiment (1 = plus négatif, 5 = plus positif)")
    ax.set_ylabel("Rendement moyen t+1 (pb)")
    ax.set_title("Rendement du lendemain selon le sentiment (IC 95 %)")
    _save(fig, path)
    return fig


def cumulative(bt: pd.DataFrame, path=None):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(bt.index, bt["cum_gross"], color=POS, label="avant frais")
    ax.plot(bt.index, bt["cum_net"], color=NEG, label="après frais")
    ax.axhline(1, color=INK_2, lw=1)
    ax.set_ylabel("Valeur de 1 € investi")
    ax.set_title("Stratégie long-short sur le sentiment")
    ax.legend(loc="upper left")
    _save(fig, path)
    return fig


def event_study_plot(es: pd.DataFrame, path=None):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for name, color, label in [("positif", POS, "actualité très positive"),
                               ("negatif", NEG, "actualité très négative")]:
        car = es[f"car_{name}"] * 100
        sem = es[f"sem_{name}"] * 100
        n = int(es[f"n_{name}"].iloc[0])
        ax.plot(es.index, car, color=color, marker="o", ms=4, label=f"{label} (n={n})")
        ax.fill_between(es.index, car - 1.96 * sem, car + 1.96 * sem, color=color, alpha=0.15, lw=0)
    ax.axvline(0, color=INK_2, lw=1, ls="--")
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_xlabel("Jours de bourse autour de l'actualité (0 = jour de l'actualité)")
    ax.set_ylabel("Rendement anormal cumulé (%)")
    ax.set_title("Étude d'événement")
    ax.legend(loc="upper left")
    _save(fig, path)
    return fig
