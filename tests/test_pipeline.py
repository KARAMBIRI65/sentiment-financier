import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finsent import analysis, data, demo, sentiment  # noqa: E402
from finsent.pipeline import Config, evaluate_models, run_market_study  # noqa: E402


def test_dates_avec_et_sans_fuseau():
    s = pd.Series(["2020-06-05 10:30:54-04:00", "2020-06-05 00:00:00", "2020-06-05T20:00:00Z"])
    out = data.parse_news_dates(s)
    assert out.iloc[0].hour == 10
    assert out.iloc[1].hour == 0          # sans fuseau : supposé heure de New York
    assert out.iloc[2].hour == 16         # 20h UTC = 16h New York (heure d'été)


def test_jour_de_bourse_sans_regarder_le_futur():
    days = pd.bdate_range("2020-06-01", "2020-06-12")  # lundi 1er au vendredi 12
    pub = data.parse_news_dates(pd.Series([
        "2020-06-03 09:00:00",   # mercredi matin         -> mercredi
        "2020-06-03 16:30:00",   # mercredi après clôture -> jeudi
        "2020-06-06 12:00:00",   # samedi                 -> lundi
        "2020-06-05 15:59:00",   # vendredi avant clôture -> vendredi
    ]))
    got = data.assign_trading_day(pub, days).dt.strftime("%Y-%m-%d").tolist()
    assert got == ["2020-06-03", "2020-06-04", "2020-06-08", "2020-06-05"]


def test_rendement_du_lendemain_bien_decale():
    prices = pd.DataFrame({"A": [100, 110, 121.0], "B": [100, 100, 100.0]},
                          index=pd.bdate_range("2020-01-01", periods=3))
    r = analysis.returns_panel(prices).set_index(["date", "ticker"])
    d1 = prices.index[1]
    assert np.isclose(r.loc[(d1, "A"), "ret"], 0.10)
    assert np.isclose(r.loc[(d1, "A"), "ret_next"], 0.10)
    assert np.isclose(r.loc[(d1, "A"), "abn"], 0.05)   # marché = moyenne (A+B)/2


def test_cache_des_scores(tmp_path):
    texts = pd.Series(["profit rises", "profit rises", "shares fall"], index=[10, 11, 12])
    p = sentiment.score_with_cache(texts, sentiment.LexiconScorer(), tmp_path / "c.csv")
    assert list(p.index) == [10, 11, 12]
    assert p.loc[10, "positive"] == 1 and p.loc[12, "negative"] == 1
    assert len(pd.read_csv(tmp_path / "c.csv")) == 2   # doublon scoré une seule fois


def test_backtest_signal_parfait():
    days = pd.bdate_range("2020-01-01", periods=50)
    rng = np.random.default_rng(0)
    rows = []
    for d in days:
        for t in "ABCD":
            ret_next = rng.normal(0, 0.01)
            rows.append({"date": d, "ticker": t, "score": np.sign(ret_next), "ret_next": ret_next})
    bt = analysis.long_short_backtest(pd.DataFrame(rows), threshold=0.5, cost_bps=0)
    assert (bt["gross"] > 0).all()


def test_pipeline_complet_en_mode_demo(tmp_path):
    cfg = Config(demo=True, start="2019-01-01", end="2020-12-31",
                 out_dir=str(tmp_path / "res"), cache_dir=str(tmp_path / "cache"))
    (tmp_path / "res" / "figures").mkdir(parents=True)
    ev = evaluate_models(cfg)
    assert set(ev.columns) >= {"accuracy", "f1_macro"}
    res = run_market_study(cfg, sentiment.LexiconScorer())
    assert res["summary"]["n_observations_panel"] > 100
    assert (tmp_path / "res" / "figures" / "etude_evenement.png").exists()
