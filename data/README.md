# Données

Les données ne sont pas dans le dépôt (trop lourdes, et leurs licences demandent de les télécharger à la source).

| Donnée | Source (gratuite) | Comment l'obtenir |
|---|---|---|
| Phrases annotées pour évaluer le modèle | Financial PhraseBank (Malo et al., 2014) | Téléchargée automatiquement depuis Hugging Face |
| Titres d'actualité datés | Kaggle, « Daily Financial News for 6000+ Stocks » (Miguel Aenlle) | Voir ci-dessous |
| Cours de bourse | Yahoo Finance | Téléchargés automatiquement par `yfinance` |

## Titres d'actualité (Kaggle)

1. Crée un compte gratuit sur kaggle.com.
2. Ouvre le jeu de données : https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests
3. Télécharge `raw_partner_headlines.csv` (ou `raw_analyst_ratings.csv`) et place-le dans ce dossier `data/`.

En ligne de commande (après avoir mis ta clé API dans `~/.kaggle/kaggle.json`) :

```bash
pip install kaggle
kaggle datasets download -d miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests -p data --unzip
```

N'importe quel autre CSV marche s'il a une colonne de texte (`headline` ou `title`), une colonne de date et une colonne de ticker (`stock`, `ticker` ou `symbol`).

Le dossier `data/cache/` garde les scores FinBERT et les cours déjà téléchargés, pour ne pas tout recalculer.
