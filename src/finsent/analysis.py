"""Du sentiment aux rendements : construction du panel, tests statistiques, backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Construction du panel (ticker x jour)
# ---------------------------------------------------------------------------
def daily_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    """Agrège les titres par (ticker, jour de bourse).

    `news` doit contenir : ticker, trading_day, score, label.
    """
    g = news.groupby(["ticker", "trading_day"])
    out = pd.DataFrame(
        {
            "score": g["score"].mean(),
            "n_news": g.size(),
            "share_pos": g["label"].apply(lambda s: (s == "positive").mean()),
            "share_neg": g["label"].apply(lambda s: (s == "negative").mean()),
        }
    ).reset_index()
    return out.rename(columns={"trading_day": "date"})


def returns_panel(prices: pd.DataFrame, market: str | None = None) -> pd.DataFrame:
    """Rendements quotidiens en format long : ticker, date, ret, ret_next, abn, abn_next.

    ret      = rendement de la clôture t-1 à la clôture t (réaction le jour même)
    ret_next = rendement de la clôture t à la clôture t+1 (ce qu'on cherche à prévoir)
    abn      = rendement anormal = rendement de l'action - rendement du marché
    """
    rets = prices.sort_index().pct_change(fill_method=None)
    if market and market in rets:
        mkt = rets[market]
        stocks = rets.drop(columns=market)
    else:
        stocks = rets
        mkt = rets.mean(axis=1)  # marché = moyenne équipondérée des actions
    abn = stocks.sub(mkt, axis=0)

    def long(df, name):
        s = df.stack(future_stack=True)
        s.index.names = ["date", "ticker"]
        return s.rename(name)

    panel = pd.concat(
        [
            long(stocks, "ret"),
            long(stocks.shift(-1), "ret_next"),
            long(abn, "abn"),
            long(abn.shift(-1), "abn_next"),
        ],
        axis=1,
    ).reset_index()
    return panel.dropna(subset=["ret"])


def build_panel(daily: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Jours avec au moins un titre, joints aux rendements du jour et du lendemain."""
    panel = daily.merge(returns, on=["ticker", "date"], how="inner")
    return panel.dropna(subset=["ret_next"]).sort_values(["date", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tests statistiques
# ---------------------------------------------------------------------------
def correlation_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, desc in [
        ("ret", "Rendement du jour même (t)"),
        ("ret_next", "Rendement du lendemain (t+1)"),
        ("abn_next", "Rendement anormal du lendemain (t+1)"),
    ]:
        d = panel[["score", target]].dropna()
        r, p = stats.pearsonr(d["score"], d[target])
        rho, p_s = stats.spearmanr(d["score"], d[target])
        rows.append(
            {"cible": desc, "n": len(d), "pearson": r, "p_pearson": p, "spearman": rho, "p_spearman": p_s}
        )
    return pd.DataFrame(rows)


def regression(panel: pd.DataFrame, target: str = "ret_next"):
    """MCO : ret_next ~ score + ret + log(n_news), erreurs-types groupées par date.

    On contrôle par le rendement du jour (effet de retournement à court terme) et
    par le nombre de titres (l'attention médiatique). Les erreurs-types groupées
    par date tiennent compte du fait que toutes les actions bougent ensemble un même jour.
    """
    import statsmodels.api as sm

    d = panel.dropna(subset=[target, "score", "ret"]).copy()
    d["log_n_news"] = np.log1p(d["n_news"])
    X = sm.add_constant(d[["score", "ret", "log_n_news"]])
    groups = pd.factorize(d["date"])[0]
    return sm.OLS(d[target], X).fit(cov_type="cluster", cov_kwds={"groups": groups})


def hit_rate(panel: pd.DataFrame, threshold: float = 0.0) -> dict:
    """Part des jours où le signe du sentiment annonce le bon signe du rendement t+1."""
    d = panel[(panel["score"].abs() > threshold) & (panel["ret_next"] != 0)].dropna(subset=["ret_next"])
    hits = (np.sign(d["score"]) == np.sign(d["ret_next"])).astype(int)
    test = stats.binomtest(hits.sum(), len(hits), 0.5) if len(hits) else None
    return {
        "n": len(hits),
        "hit_rate": hits.mean() if len(hits) else np.nan,
        "p_value": test.pvalue if test else np.nan,
    }


def quantile_returns(panel: pd.DataFrame, n_bins: int = 5, target: str = "ret_next") -> pd.DataFrame:
    """Rendement moyen t+1 par quintile de sentiment (avec erreur-type)."""
    d = panel.dropna(subset=[target]).copy()
    d["bin"] = pd.qcut(d["score"].rank(method="first"), n_bins, labels=range(1, n_bins + 1))
    g = d.groupby("bin", observed=True)[target]
    out = pd.DataFrame({"mean": g.mean(), "sem": g.sem(), "n": g.size()})
    out["score_moyen"] = d.groupby("bin", observed=True)["score"].mean()
    return out.reset_index()


# ---------------------------------------------------------------------------
# Backtest d'une stratégie long-short simple
# ---------------------------------------------------------------------------
def long_short_backtest(
    panel: pd.DataFrame,
    threshold: float = 0.5,
    cost_bps: float = 5.0,
    target: str = "ret_next",
) -> pd.DataFrame:
    """Chaque jour t, à la clôture :
    - achète (équipondéré) les actions dont le sentiment du jour > +threshold,
    - vend à découvert celles dont le sentiment < -threshold,
    - garde la position un jour, puis recommence.

    `cost_bps` = coût de transaction par aller simple, en points de base. La
    position est entièrement renouvelée chaque jour (hypothèse prudente).
    """
    rows = []
    for date, day in panel.dropna(subset=[target]).groupby("date"):
        longs = day.loc[day["score"] > threshold, target]
        shorts = day.loc[day["score"] < -threshold, target]
        if longs.empty and shorts.empty:
            continue
        legs = (not longs.empty) + (not shorts.empty)
        gross = (longs.mean() if not longs.empty else 0.0) - (shorts.mean() if not shorts.empty else 0.0)
        cost = legs * 2 * cost_bps / 1e4  # entrée + sortie sur chaque jambe
        rows.append({"date": date, "gross": gross, "net": gross - cost, "n_long": len(longs), "n_short": len(shorts)})
    bt = pd.DataFrame(rows).set_index("date").sort_index()
    if not bt.empty:
        bt["cum_gross"] = (1 + bt["gross"]).cumprod()
        bt["cum_net"] = (1 + bt["net"]).cumprod()
    return bt


def performance(daily_returns: pd.Series) -> dict:
    r = daily_returns.dropna()
    if r.empty:
        return {}
    cum = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    vol = r.std() * np.sqrt(TRADING_DAYS)
    return {
        "jours_investis": len(r),
        "rendement_total": cum.iloc[-1] - 1,
        "rendement_annualise": cum.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "volatilite_annualisee": vol,
        "sharpe": r.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else np.nan,
        "max_drawdown": (cum / cum.cummax() - 1).min(),
    }


# ---------------------------------------------------------------------------
# Étude d'événement : que se passe-t-il autour des jours très positifs / négatifs ?
# ---------------------------------------------------------------------------
def event_study(
    panel: pd.DataFrame,
    returns: pd.DataFrame,
    threshold: float = 0.5,
    window: int = 5,
) -> pd.DataFrame:
    """Rendement anormal cumulé moyen (CAR) de t-window à t+window.

    Le jour 0 est le jour de l'actualité. La courbe montre si le marché a réagi
    avant (fuite d'info / l'actualité commente une hausse déjà passée), le jour
    même, ou après (dérive = signal exploitable).
    """
    abn = returns.pivot(index="date", columns="ticker", values="abn").sort_index()
    pos_index = {d: i for i, d in enumerate(abn.index)}
    offsets = range(-window, window + 1)
    out = {}
    for name, mask in [
        ("positif", panel["score"] > threshold),
        ("negatif", panel["score"] < -threshold),
    ]:
        paths = []
        for _, ev in panel[mask].iterrows():
            i = pos_index.get(ev["date"])
            if i is None or ev["ticker"] not in abn or i - window < 0 or i + window >= len(abn):
                continue
            path = abn[ev["ticker"]].iloc[i - window : i + window + 1].to_numpy()
            if not np.isnan(path).any():
                paths.append(path)
        arr = np.array(paths) if paths else np.full((1, len(offsets)), np.nan)
        car = np.cumsum(arr, axis=1)
        out[f"car_{name}"] = car.mean(axis=0)
        out[f"sem_{name}"] = car.std(axis=0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.nan
        out[f"n_{name}"] = len(paths)
    return pd.DataFrame(out, index=pd.Index(list(offsets), name="jour"))
