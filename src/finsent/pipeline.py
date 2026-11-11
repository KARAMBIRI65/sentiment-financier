"""Enchaîne toutes les étapes. Utilisé par scripts/run_pipeline.py."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis, data, plots, sentiment


@dataclass
class Config:
    news_path: str | None = "data/raw_partner_headlines.csv"
    start: str = "2016-01-01"
    end: str = "2020-06-30"
    tickers: list[str] | None = None
    top_n_tickers: int = 50
    max_per_ticker: int | None = 2000
    market: str = "SPY"
    threshold: float = 0.5
    cost_bps: float = 5.0
    out_dir: str = "results"
    cache_dir: str = "data/cache"
    demo: bool = False
    skip_eval: bool = False
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
def evaluate_models(cfg: Config, finbert=None) -> pd.DataFrame:
    """Compare FinBERT, TF-IDF + régression logistique et un lexique sur PhraseBank."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split

    pb = data.load_phrasebank() if not cfg.demo else _demo().synthetic_phrasebank()
    train, test = train_test_split(pb, test_size=0.2, stratify=pb["label"], random_state=42)

    models = {
        "Lexique (référence naïve)": sentiment.LexiconScorer(),
        "TF-IDF + régression logistique": sentiment.TfidfLogRegScorer().fit(train["text"], train["label"]),
    }
    if finbert is not None:
        models["FinBERT"] = finbert

    out = Path(cfg.out_dir) / "figures"
    rows = []
    order = ["negative", "neutral", "positive"]
    for name, model in models.items():
        pred = sentiment.predict_label(model.predict_proba(test["text"]))
        rows.append(
            {
                "modèle": name,
                "accuracy": accuracy_score(test["label"], pred),
                "f1_macro": f1_score(test["label"], pred, average="macro"),
            }
        )
        cm = confusion_matrix(test["label"], pred, labels=order)
        slug = name.split()[0].lower().replace("-", "")
        plots.confusion(cm, ["nég.", "neutre", "pos."], name, out / f"confusion_{slug}.png")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def run_market_study(cfg: Config, scorer) -> dict:
    out = Path(cfg.out_dir)
    cache = Path(cfg.cache_dir)

    # 1. Titres
    if cfg.demo:
        demo = _demo()
        tick = [f"T{i:02d}" for i in range(12)]
        tmp = cache / "demo_news.csv"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        demo.synthetic_news(tick, cfg.start, cfg.end).to_csv(tmp, index=False)
        news = data.load_headlines(tmp)
    else:
        news = data.load_headlines(
            cfg.news_path, cfg.tickers, cfg.start, cfg.end, cfg.top_n_tickers, cfg.max_per_ticker
        )
    tickers = sorted(news["ticker"].unique())

    # 2. Cours (marché inclus)
    if cfg.demo:
        prices = demo.synthetic_prices(tickers, cfg.start, cfg.end)
        market = None
    else:
        end_plus = (pd.Timestamp(cfg.end) + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
        prices = data.download_prices(
            tickers + [cfg.market], cfg.start, end_plus, cache / f"prices_{cfg.start}_{cfg.end}.csv"
        )
        market = cfg.market
    news = news[news["ticker"].isin(prices.columns)]

    # 3. Sentiment de chaque titre
    proba = sentiment.score_with_cache(
        news["headline"], scorer, None if cfg.demo else cache / "finbert_scores.csv"
    )
    news = news.assign(
        score=sentiment.sentiment_score(proba), label=sentiment.predict_label(proba)
    )
    news["trading_day"] = data.assign_trading_day(news["published"], prices.index)
    news = news.dropna(subset=["trading_day"])

    # 4. Panel et tests
    returns = analysis.returns_panel(prices, market)
    panel = analysis.build_panel(analysis.daily_sentiment(news), returns)

    corr = analysis.correlation_table(panel)
    reg = analysis.regression(panel)
    hits = analysis.hit_rate(panel, cfg.threshold)
    quint = analysis.quantile_returns(panel)
    bt = analysis.long_short_backtest(panel, cfg.threshold, cfg.cost_bps)
    perf = {
        "avant_frais": analysis.performance(bt["gross"]) if len(bt) else {},
        "apres_frais": analysis.performance(bt["net"]) if len(bt) else {},
    }
    es = analysis.event_study(panel, returns, cfg.threshold)

    # 5. Graphiques et sauvegarde
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plots.score_distribution(news["score"], fig_dir / "distribution_sentiment.png")
    plots.quantile_bars(quint, fig_dir / "rendement_par_quintile.png")
    if len(bt):
        plots.cumulative(bt, fig_dir / "strategie_long_short.png")
    plots.event_study_plot(es, fig_dir / "etude_evenement.png")

    summary = {
        "n_titres": int(len(news)),
        "n_tickers": len(tickers),
        "n_observations_panel": int(len(panel)),
        "periode": [cfg.start, cfg.end],
        "taux_de_reussite": hits,
        "regression": {
            "coef_score": float(reg.params["score"]),
            "t_stat_score": float(reg.tvalues["score"]),
            "p_value_score": float(reg.pvalues["score"]),
            "r2": float(reg.rsquared),
        },
        "strategie": perf,
    }
    corr.to_csv(out / "correlations.csv", index=False)
    quint.to_csv(out / "quintiles.csv", index=False)
    es.to_csv(out / "etude_evenement.csv")
    (out / "regression.txt").write_text(str(reg.summary()))
    (out / "resume.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json))
    return {"news": news, "panel": panel, "returns": returns, "corr": corr, "reg": reg,
            "hits": hits, "quintiles": quint, "backtest": bt, "perf": perf, "event_study": es,
            "summary": summary}


def _json(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return str(x)


def _demo():
    from . import demo
    return demo
