"""Génère notebooks/analyse_sentiment.ipynb (à relancer si on modifie le texte)."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# Le sentiment des actualités financières prédit-il les rendements du lendemain ?

**Objectif.** Mesurer le ton (positif, négatif, neutre) de titres d'actualité avec **FinBERT**, un modèle de langage spécialisé en finance, puis tester si ce ton aide à prévoir le rendement d'une action le jour de bourse suivant.

**Plan.**
1. Évaluer FinBERT sur des phrases financières annotées (Financial PhraseBank) et le comparer à deux références plus simples.
2. Scorer des dizaines de milliers de titres d'actualité datés.
3. Aligner chaque titre sur le bon jour de bourse, **sans regarder dans le futur**.
4. Tester le lien sentiment → rendement : corrélations, régression, quintiles.
5. Backtester une stratégie long-short, avec et sans frais.
6. Étude d'événement : comment le prix se comporte autour d'une actualité forte.

Toutes les données sont publiques et gratuites (voir `data/README.md`)."""),
    md("""## 0. Installation

Sur **Google Colab** (GPU gratuit : *Exécution > Modifier le type d'exécution > GPU*), décommente la cellule suivante. En local, fais simplement `pip install -r requirements.txt`.

`DEMO = True` lance tout sur des données factices, sans Internet ni FinBERT : pratique pour vérifier que le code tourne, mais les chiffres n'ont alors aucun sens."""),
    code("""# !git clone https://github.com/<ton-pseudo>/sentiment-financier.git
# %cd sentiment-financier
# !pip install -q -r requirements.txt"""),
    code("""import sys, os
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import pandas as pd
from finsent import analysis, data, plots, sentiment
from finsent.pipeline import Config

DEMO = os.environ.get("FINSENT_DEMO", "0") == "1"   # passe à True pour un test rapide
plots.setup_style()
pd.set_option("display.float_format", "{:.4f}".format)

cfg = Config(
    news_path="data/raw_partner_headlines.csv",
    start="2016-01-01", end="2020-06-30",
    top_n_tickers=50,        # les 50 actions les plus suivies par la presse
    max_per_ticker=2000,     # limite le temps de calcul de FinBERT
    market="SPY",            # l'ETF S&P 500 sert de référence « marché »
    threshold=0.5,           # seuil de sentiment pour entrer en position
    cost_bps=5.0,            # 5 points de base de frais par transaction
    demo=DEMO,
)
if DEMO:
    cfg.start, cfg.end, cfg.out_dir = "2019-01-01", "2020-12-31", "results_demo"
FIG = Path(cfg.out_dir) / "figures"
FIG.mkdir(parents=True, exist_ok=True)"""),
    md("""## 1. Le modèle est-il fiable ? Évaluation sur Financial PhraseBank

Financial PhraseBank contient ~2 300 phrases d'actualité financière annotées à l'unanimité par des experts (positif / négatif / neutre). On compare :

- **un lexique** : on compte les mots positifs et négatifs (référence naïve) ;
- **TF-IDF + régression logistique** : un modèle classique entraîné sur 80 % des phrases ;
- **FinBERT** (`ProsusAI/finbert`) : un BERT ré-entraîné sur des textes financiers.

⚠️ FinBERT a été en partie entraîné sur PhraseBank : son score ici est **optimiste**. On le garde comme vérification, pas comme preuve de performance."""),
    code("""from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from finsent import demo as demo_data

pb = demo_data.synthetic_phrasebank() if DEMO else data.load_phrasebank("sentences_allagree")
print(pb["label"].value_counts())
train, test = train_test_split(pb, test_size=0.2, stratify=pb["label"], random_state=42)

scorer = sentiment.LexiconScorer() if DEMO else sentiment.FinBertScorer()
models = {
    "Lexique": sentiment.LexiconScorer(),
    "TF-IDF + LogReg": sentiment.TfidfLogRegScorer().fit(train["text"], train["label"]),
}
if not DEMO:
    models["FinBERT"] = scorer

order = ["negative", "neutral", "positive"]
for name, model in models.items():
    pred = sentiment.predict_label(model.predict_proba(test["text"]))
    print(f"=== {name} ===")
    print(classification_report(test["label"], pred, labels=order, digits=3))
    plots.confusion(confusion_matrix(test["label"], pred, labels=order),
                    ["nég.", "neutre", "pos."], name, FIG / f"confusion_{name.split()[0].lower().replace(\"-\", \"\")}.png")"""),
    md("""## 2. Scorer les titres d'actualité

Le **score de sentiment** d'un titre vaut `P(positif) − P(négatif)`, entre −1 et +1. Les scores sont mis en cache (`data/cache/`) : FinBERT ne tourne qu'une fois par titre."""),
    code("""if DEMO:
    tick = [f"T{i:02d}" for i in range(12)]
    Path("data/cache").mkdir(parents=True, exist_ok=True)
    demo_data.synthetic_news(tick, cfg.start, cfg.end).to_csv("data/cache/demo_news.csv", index=False)
    news = data.load_headlines("data/cache/demo_news.csv")
else:
    news = data.load_headlines(cfg.news_path, cfg.tickers, cfg.start, cfg.end,
                               cfg.top_n_tickers, cfg.max_per_ticker)
print(f"{len(news):,} titres, {news['ticker'].nunique()} actions")
news.head()"""),
    code("""%%time
proba = sentiment.score_with_cache(news["headline"], scorer,
                                   None if DEMO else "data/cache/finbert_scores.csv")
news["score"] = sentiment.sentiment_score(proba)
news["label"] = sentiment.predict_label(proba)
print(news["label"].value_counts(normalize=True))
plots.score_distribution(news["score"], FIG / "distribution_sentiment.png");"""),
    code("""# Les titres les plus positifs et les plus négatifs : vérification « à l'œil »
pd.concat([news.nlargest(5, "score"), news.nsmallest(5, "score")])[["ticker", "published", "headline", "score"]]"""),
    md("""## 3. Cours de bourse et alignement temporel

C'est l'étape où la plupart des projets se trompent. Règle utilisée :

- titre publié **avant 16h** (heure de New York) un jour de bourse → associé à **ce jour t** ;
- publié **après la clôture**, le week-end ou un jour férié → associé au **jour de bourse suivant**.

Ainsi, le signal du jour t n'utilise que de l'information connue avant la clôture de t, et on cherche à prévoir le rendement **de la clôture de t à la clôture de t+1**.

On calcule aussi le **rendement anormal** (rendement de l'action − rendement du S&P 500) pour retirer l'effet « tout le marché monte »."""),
    code("""tickers = sorted(news["ticker"].unique())
if DEMO:
    prices, market = demo_data.synthetic_prices(tickers, cfg.start, cfg.end), None
else:
    end_plus = (pd.Timestamp(cfg.end) + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    prices = data.download_prices(tickers + [cfg.market], cfg.start, end_plus,
                                  f"data/cache/prices_{cfg.start}_{cfg.end}.csv")
    market = cfg.market
news = news[news["ticker"].isin(prices.columns)].copy()
news["trading_day"] = data.assign_trading_day(news["published"], prices.index)
news = news.dropna(subset=["trading_day"])

returns = analysis.returns_panel(prices, market)
panel = analysis.build_panel(analysis.daily_sentiment(news), returns)
print(f"{len(panel):,} couples (action, jour) avec au moins un titre")
panel.head()"""),
    md("""## 4. Le sentiment est-il lié aux rendements ?

On regarde deux choses bien différentes :

- **le jour même (t)** : le titre *décrit*-il ce que fait le prix ? On s'attend à un lien fort (« Apple chute de 5 % » est négatif…).
- **le lendemain (t+1)** : le titre *annonce*-t-il le mouvement suivant ? C'est la vraie question pour un investisseur, et si les marchés sont efficaces, le lien doit être faible."""),
    code("""corr = analysis.correlation_table(panel)
corr"""),
    code("""reg = analysis.regression(panel)
print(reg.summary().tables[1])"""),
    code("""print(analysis.hit_rate(panel, cfg.threshold))
quint = analysis.quantile_returns(panel)
plots.quantile_bars(quint, FIG / "rendement_par_quintile.png")
quint"""),
    md("""## 5. Backtest d'une stratégie long-short

Chaque soir, on **achète** les actions dont le sentiment du jour dépasse +0,5 et on **vend à découvert** celles sous −0,5, puis on solde le lendemain. Les frais (5 pb par transaction) sont comptés à chaque entrée et sortie.

C'est volontairement simple : l'enjeu est de voir si un éventuel signal **survit aux frais**."""),
    code("""bt = analysis.long_short_backtest(panel, cfg.threshold, cfg.cost_bps)
plots.cumulative(bt, FIG / "strategie_long_short.png")
pd.DataFrame({"avant frais": analysis.performance(bt["gross"]),
              "après frais": analysis.performance(bt["net"])})"""),
    md("""## 6. Étude d'événement

Rendement anormal cumulé moyen de 5 jours avant à 5 jours après une actualité très positive ou très négative. Lecture :

- la courbe monte **avant** le jour 0 → l'actualité commente un mouvement déjà passé ;
- saut **au jour 0** → le marché réagit dans la journée ;
- dérive **après** le jour 0 → le marché réagit lentement, et c'est là qu'un signal serait exploitable."""),
    code("""es = analysis.event_study(panel, returns, cfg.threshold)
plots.event_study_plot(es, FIG / "etude_evenement.png")
es"""),
    md("""## 7. Conclusions et limites

À compléter avec **tes** chiffres après exécution. Points à discuter :

- **Jour même vs lendemain** : comparer les deux corrélations. Un lien fort le jour même et faible le lendemain est cohérent avec des marchés qui intègrent vite l'information publique.
- **Significativité** : regarder la t-stat de la régression (> 2 en valeur absolue ≈ significatif à 5 %).
- **Frais** : un signal qui disparaît après frais n'est pas exploitable.
- **Limites** : horodatage des titres parfois imprécis (date sans heure), biais du survivant (on ne garde que les actions encore cotées sur Yahoo), FinBERT n'a lu que le titre et pas l'article, période limitée.
- **Pistes** : pondérer par la source, tester d'autres horizons (t+2, t+5), comparer avec un LLM récent, appliquer la méthode à des titres français sur le CAC 40."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
})
out = Path(__file__).resolve().parents[1] / "notebooks" / "analyse_sentiment.ipynb"
nbf.write(nb, out)
print(f"écrit : {out}")
