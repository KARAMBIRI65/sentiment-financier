# Sentiment des actualités financières et rendements boursiers

> Le ton des titres d'actualité, mesuré par FinBERT, aide-t-il à prévoir le rendement d'une action le lendemain ?

Projet personnel de NLP appliqué à la finance de marché : je mesure le sentiment de dizaines de milliers de titres d'actualité avec un modèle de langage spécialisé (**FinBERT**), je les aligne proprement sur les jours de bourse, puis je teste statistiquement et économiquement (backtest avec frais) le lien avec les rendements.

**Compétences mobilisées :** NLP (transformers, Hugging Face), apprentissage supervisé (TF-IDF + régression logistique), économétrie (MCO, erreurs-types groupées), séries financières, backtest, étude d'événement, tests unitaires.

---

## Résultats

> _Section à remplir après avoir lancé le notebook sur les vraies données. Remplace chaque `…` par tes chiffres (ils sont aussi dans `results/resume.json`)._

| Question | Résultat |
|---|---|
| Précision de FinBERT sur PhraseBank (F1 macro) | … |
| Corrélation sentiment / rendement du **jour même** | … |
| Corrélation sentiment / rendement du **lendemain** | … |
| Coefficient de la régression (t-stat) | … |
| Sharpe de la stratégie avant / après frais | … / … |

![Étude d'événement](results/figures/etude_evenement.png)

**Ce que j'en conclus :** … _(2 ou 3 phrases : par exemple, si le lien est fort le jour même mais faible le lendemain, le marché intègre vite l'information publique ; et si le signal disparaît après frais, il n'est pas exploitable tel quel.)_

---

## Méthode

```
Titres d'actualité ──► FinBERT ──► score = P(positif) − P(négatif)
                                        │
             alignement sur le jour de bourse (règle de clôture 16h New York)
                                        │
Cours Yahoo Finance ──► rendements t et t+1, rendements anormaux vs S&P 500
                                        │
     corrélations · régression · quintiles · backtest long-short · étude d'événement
```

1. **Évaluation du modèle.** FinBERT est comparé à un lexique de mots et à un modèle TF-IDF + régression logistique sur Financial PhraseBank (80 % entraînement / 20 % test).
2. **Scoring.** Chaque titre reçoit un score entre −1 et +1. Les scores sont mis en cache pour ne calculer qu'une fois.
3. **Alignement sans biais d'anticipation.** Un titre publié avant 16h (heure de New York) compte pour ce jour ; après la clôture, le week-end ou un jour férié, il compte pour le jour de bourse suivant. On prévoit ensuite le rendement de la clôture de t à celle de t+1. Cette règle est vérifiée par des tests unitaires.
4. **Tests statistiques.** Corrélations de Pearson et Spearman, régression `ret(t+1) ~ sentiment(t) + ret(t) + log(nb titres)` avec erreurs-types groupées par date, taux de bonnes prédictions de signe (test binomial), rendement moyen par quintile de sentiment.
5. **Backtest.** Chaque soir : achat des actions au sentiment > +0,5, vente à découvert de celles < −0,5, position soldée le lendemain, 5 pb de frais par transaction.
6. **Étude d'événement.** Rendement anormal cumulé de 5 jours avant à 5 jours après une actualité forte.

## Données (toutes gratuites)

| Donnée | Source |
|---|---|
| Phrases annotées | [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank), Malo et al. (2014), téléchargé automatiquement |
| Titres d'actualité datés | [Daily Financial News for 6000+ Stocks](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests) (Kaggle, compte gratuit) |
| Cours de bourse | Yahoo Finance via `yfinance`, téléchargés automatiquement |
| Modèle | [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert), Araci (2019) |

Détails de téléchargement : [`data/README.md`](data/README.md).

## Lancer le projet

```bash
git clone https://github.com/<ton-pseudo>/sentiment-financier.git
cd sentiment-financier
pip install -r requirements.txt

# 1. Vérifier que tout s'installe bien (données factices, 10 secondes, sans Internet)
python scripts/run_pipeline.py --demo

# 2. Télécharger les titres Kaggle dans data/ (voir data/README.md), puis l'étude complète
python scripts/run_pipeline.py
# ou, pas à pas et commenté :
jupyter notebook notebooks/analyse_sentiment.ipynb
```

Sans GPU, FinBERT met environ 10 à 20 minutes pour 100 000 titres. Sur **Google Colab** avec un GPU gratuit, c'est quelques minutes. Options utiles : `--tickers AAPL MSFT TSLA`, `--top-n 20`, `--max-per-ticker 500` pour aller plus vite.

```bash
pytest   # tests : dates, alignement sans regard vers le futur, décalage des rendements, backtest
```

## Structure

```
├── notebooks/analyse_sentiment.ipynb   # l'analyse commentée, étape par étape
├── src/finsent/
│   ├── data.py         # PhraseBank, titres, cours, alignement sur les jours de bourse
│   ├── sentiment.py    # FinBERT, TF-IDF + LogReg, lexique, cache des scores
│   ├── analysis.py     # panel, corrélations, régression, backtest, étude d'événement
│   ├── plots.py        # graphiques
│   ├── pipeline.py     # enchaînement complet
│   └── demo.py         # données factices pour les tests
├── scripts/run_pipeline.py             # tout en une commande
├── tests/test_pipeline.py
└── data/README.md                      # où trouver les données
```

## Limites et pistes

- Les titres sans heure précise sont supposés publiés à minuit, donc avant l'ouverture.
- Biais du survivant : seules les actions encore disponibles sur Yahoo Finance sont gardées.
- FinBERT ne lit que le titre, pas l'article ; il a aussi été en partie entraîné sur PhraseBank, donc son score sur ce jeu est optimiste.
- Pistes : autres horizons (t+2, t+5), pondération par source, comparaison avec un LLM récent, même méthode sur des titres français et le CAC 40.

## Références

- Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*. arXiv:1908.10063.
- Malo, P. et al. (2014). *Good debt or bad debt: Detecting semantic orientations in economic texts*. JASIST.
- Tetlock, P. (2007). *Giving Content to Investor Sentiment: The Role of Media in the Stock Market*. Journal of Finance.

---

Issa Aziz KARAMBIRI, élève ingénieur en mathématiques appliquées (data, IA, finance). Licence MIT.
