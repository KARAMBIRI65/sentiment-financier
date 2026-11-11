"""Lance toute l'étude en une commande.

Exemples :
    python scripts/run_pipeline.py                       # étude complète (FinBERT + vraies données)
    python scripts/run_pipeline.py --tickers AAPL MSFT   # seulement quelques actions
    python scripts/run_pipeline.py --demo                # vérifie l'installation (données factices)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finsent import plots, sentiment  # noqa: E402
from finsent.pipeline import Config, evaluate_models, run_market_study  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--news", default="data/raw_partner_headlines.csv", help="CSV des titres d'actualité")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default="2020-06-30")
    p.add_argument("--tickers", nargs="*", help="liste de tickers (sinon les plus couverts)")
    p.add_argument("--top-n", type=int, default=50, help="nombre d'actions les plus couvertes à garder")
    p.add_argument("--max-per-ticker", type=int, default=2000, help="titres max par action (temps de calcul)")
    p.add_argument("--threshold", type=float, default=0.5, help="seuil de sentiment pour trader")
    p.add_argument("--cost-bps", type=float, default=5.0, help="frais par transaction, en points de base")
    p.add_argument("--out", default="results")
    p.add_argument("--skip-eval", action="store_true", help="ne pas évaluer les modèles sur PhraseBank")
    p.add_argument("--demo", action="store_true", help="données synthétiques, sans FinBERT ni Internet")
    a = p.parse_args()

    cfg = Config(
        news_path=a.news, start=a.start, end=a.end, tickers=a.tickers, top_n_tickers=a.top_n,
        max_per_ticker=a.max_per_ticker, threshold=a.threshold, cost_bps=a.cost_bps,
        out_dir=a.out, demo=a.demo, skip_eval=a.skip_eval,
    )
    if a.demo:
        cfg.start, cfg.end = "2019-01-01", "2020-12-31"
        cfg.out_dir = "results_demo"
    plots.setup_style()
    (Path(cfg.out_dir) / "figures").mkdir(parents=True, exist_ok=True)

    scorer = sentiment.LexiconScorer() if a.demo else sentiment.FinBertScorer()

    if not cfg.skip_eval:
        print("1/2  Évaluation des modèles sur Financial PhraseBank…")
        ev = evaluate_models(cfg, None if a.demo else scorer)
        ev.to_csv(Path(cfg.out_dir) / "evaluation_modeles.csv", index=False)
        print(ev.round(3).to_string(index=False))

    print("2/2  Sentiment des titres et lien avec les rendements…")
    res = run_market_study(cfg, scorer)
    s = res["summary"]
    print(res["corr"].round(4).to_string(index=False))
    print(f"Taux de réussite : {s['taux_de_reussite']['hit_rate']:.1%} "
          f"(n={s['taux_de_reussite']['n']}, p={s['taux_de_reussite']['p_value']:.3f})")
    r = s["regression"]
    print(f"Régression : coef score = {r['coef_score']:.5f}, t = {r['t_stat_score']:.2f}")
    net = s["strategie"]["apres_frais"]
    if net:
        print(f"Stratégie après frais : Sharpe {net['sharpe']:.2f}, "
              f"rendement annualisé {net['rendement_annualise']:.1%}")
    print(f"Résultats enregistrés dans {cfg.out_dir}/")
    if a.demo:
        print("ATTENTION : mode démo, données factices. Ces chiffres ne veulent rien dire.")


if __name__ == "__main__":
    main()
