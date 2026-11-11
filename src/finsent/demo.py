"""Données synthétiques, uniquement pour les tests et pour vérifier l'installation.

Les résultats obtenus avec ces données n'ont AUCUNE valeur : ne jamais les publier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import MARKET_TZ

_POS = ["{t} beats earnings estimates", "{t} shares surge after upgrade", "{t} raises full-year guidance",
        "{t} posts record profit"]
_NEG = ["{t} misses revenue estimates", "{t} downgraded to sell", "{t} warns of weak demand",
        "{t} shares plunge on lawsuit"]
_NEU = ["{t} to present at investor conference", "{t} announces board meeting date",
        "{t} files quarterly report"]


def synthetic_prices(tickers, start="2019-01-01", end="2020-12-31", seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    rets = rng.normal(0.0003, 0.015, size=(len(days), len(tickers)))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=days, columns=list(tickers))


def synthetic_news(tickers, start="2019-01-01", end="2020-12-31", n=3000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0, t1 = pd.Timestamp(start, tz=MARKET_TZ), pd.Timestamp(end, tz=MARKET_TZ)
    stamps = t0 + pd.to_timedelta(rng.uniform(0, (t1 - t0).total_seconds(), n), unit="s")
    rows = []
    for ts in stamps:
        t = rng.choice(list(tickers))
        pool = [_POS, _NEG, _NEU][rng.integers(3)]
        rows.append({"stock": t, "date": ts.strftime("%Y-%m-%d %H:%M:%S%z"),
                     "headline": pool[rng.integers(len(pool))].format(t=t)})
    return pd.DataFrame(rows)


def synthetic_phrasebank(n=600, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        lab = ["positive", "negative", "neutral"][rng.integers(3)]
        pool = {"positive": _POS, "negative": _NEG, "neutral": _NEU}[lab]
        rows.append({"text": pool[rng.integers(len(pool))].format(t=f"Company{rng.integers(50)}"), "label": lab})
    return pd.DataFrame(rows)
