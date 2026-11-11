"""Chargement des données : phrases annotées, titres d'actualité et cours de bourse.

Toutes les sources sont publiques et gratuites :
- Financial PhraseBank (Malo et al., 2014) via Hugging Face, pour évaluer le modèle ;
- un fichier CSV de titres d'actualité datés (par défaut le jeu Kaggle
  « Daily Financial News for 6000+ Stocks ») ;
- les cours de clôture ajustés via Yahoo Finance (bibliothèque yfinance).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

MARKET_TZ = "America/New_York"
CLOSE_HOUR = 16  # clôture du NYSE / Nasdaq, heure de New York

PHRASEBANK_LABELS = {0: "negative", 1: "neutral", 2: "positive"}


# ---------------------------------------------------------------------------
# 1. Financial PhraseBank
# ---------------------------------------------------------------------------
def load_phrasebank(config: str = "sentences_allagree") -> pd.DataFrame:
    """Renvoie un DataFrame (text, label) avec label dans {negative, neutral, positive}.

    `config` choisit le niveau d'accord entre annotateurs : sentences_allagree
    (100 %, ~2 260 phrases), sentences_75agree, sentences_66agree, sentences_50agree.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("takala/financial_phrasebank", config, trust_remote_code=True)
    except Exception:
        # Les versions récentes de `datasets` n'exécutent plus les scripts de
        # chargement : on lit alors la copie Parquet générée par Hugging Face.
        ds = load_dataset(
            "takala/financial_phrasebank",
            revision="refs/convert/parquet",
            data_dir=config,
        )
    df = ds["train"].to_pandas()
    df = df.rename(columns={"sentence": "text"})
    df["label"] = df["label"].map(PHRASEBANK_LABELS)
    return df[["text", "label"]]


# ---------------------------------------------------------------------------
# 2. Titres d'actualité
# ---------------------------------------------------------------------------
_TEXT_COLS = ("headline", "title", "text", "news")
_DATE_COLS = ("date", "datetime", "published", "time")
_TICKER_COLS = ("stock", "ticker", "symbol")
_TZ_SUFFIX = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")


def _pick(columns, candidates, what):
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    raise KeyError(f"Aucune colonne {what} trouvée parmi {list(columns)}")


def parse_news_dates(values: pd.Series) -> pd.Series:
    """Convertit des dates hétérogènes en horodatages à l'heure de New York.

    Les dates avec fuseau (« 2020-06-05 10:30:54-04:00 ») sont converties ;
    les dates sans fuseau (« 2020-06-05 ») sont supposées déjà à l'heure de New York.
    """
    s = values.astype(str).str.strip()
    has_tz = s.str.contains(_TZ_SUFFIX)
    out = pd.Series(pd.NaT, index=s.index, dtype=f"datetime64[ns, {MARKET_TZ}]")
    if has_tz.any():
        aware = pd.to_datetime(s[has_tz], utc=True, format="mixed", errors="coerce")
        out[has_tz] = aware.dt.tz_convert(MARKET_TZ)
    if (~has_tz).any():
        naive = pd.to_datetime(s[~has_tz], format="mixed", errors="coerce")
        out[~has_tz] = naive.dt.tz_localize(MARKET_TZ, ambiguous="NaT", nonexistent="NaT")
    return out


def load_headlines(
    path: str | Path,
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    top_n_tickers: int | None = None,
    max_per_ticker: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Lit un CSV de titres et renvoie (ticker, published, headline).

    Le fichier doit contenir une colonne de texte (headline/title), une colonne de
    date et une colonne de ticker (stock/ticker/symbol), quels que soient leurs noms exacts.
    """
    raw = pd.read_csv(path)
    text_col = _pick(raw.columns, _TEXT_COLS, "de texte")
    date_col = _pick(raw.columns, _DATE_COLS, "de date")
    tick_col = _pick(raw.columns, _TICKER_COLS, "de ticker")

    df = pd.DataFrame(
        {
            "ticker": raw[tick_col].astype(str).str.upper().str.strip(),
            "published": parse_news_dates(raw[date_col]),
            "headline": raw[text_col].astype(str).str.strip(),
        }
    ).dropna(subset=["published"])
    df = df[df["headline"].str.len() > 0]

    if start:
        df = df[df["published"] >= pd.Timestamp(start, tz=MARKET_TZ)]
    if end:
        df = df[df["published"] < pd.Timestamp(end, tz=MARKET_TZ) + pd.Timedelta(days=1)]
    if tickers:
        df = df[df["ticker"].isin([t.upper() for t in tickers])]
    if top_n_tickers:
        keep = df["ticker"].value_counts().head(top_n_tickers).index
        df = df[df["ticker"].isin(keep)]
    if max_per_ticker:
        df = (
            df.groupby("ticker", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max_per_ticker), random_state=seed))
        )
    df = df.drop_duplicates(subset=["ticker", "headline", "published"])
    return df.sort_values("published").reset_index(drop=True)


def assign_trading_day(published: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """Associe chaque titre au jour de bourse où il est connu AVANT la clôture.

    - publié avant 16h (New York) un jour de bourse  -> ce jour-là ;
    - publié après 16h, le week-end ou un jour férié -> le jour de bourse suivant.

    C'est la règle qui évite de « regarder dans le futur » : le signal du jour t
    n'utilise que des informations disponibles avant la clôture de t.
    """
    days = pd.DatetimeIndex(trading_days).tz_localize(None).normalize().sort_values()
    local = published.dt.tz_convert(MARKET_TZ)
    effective = local.dt.tz_localize(None).dt.normalize()
    effective = effective.where(local.dt.hour < CLOSE_HOUR, effective + pd.Timedelta(days=1))
    pos = np.searchsorted(days.values, effective.values, side="left")
    valid = pos < len(days)
    out = pd.Series(pd.NaT, index=published.index, dtype="datetime64[ns]")
    out[valid] = days.values[pos[valid]]
    return out


# ---------------------------------------------------------------------------
# 3. Cours de bourse
# ---------------------------------------------------------------------------
def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache: str | Path | None = None,
) -> pd.DataFrame:
    """Cours de clôture ajustés (dividendes et splits), une colonne par ticker."""
    if cache and Path(cache).exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    import yfinance as yf

    data = yf.download(
        sorted(set(tickers)), start=start, end=end, auto_adjust=True, progress=False
    )
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    if not isinstance(data.columns, pd.MultiIndex):
        close.columns = [tickers[0]]
    close = close.dropna(axis=1, how="all")
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        close.to_csv(cache)
    return close
