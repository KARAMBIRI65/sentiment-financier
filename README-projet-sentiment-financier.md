# Sentiment des actualités et bourse

Est-ce que le ton des titres d'actualité sur une entreprise aide à prévoir son action le lendemain ?

Je note chaque titre avec FinBERT, un modèle de langue pour la finance, puis je compare ces notes aux rendements des actions.

J'utilise Financial PhraseBank, des titres d'actualité trouvés sur Kaggle et les cours de Yahoo Finance.

Un titre publié après la clôture compte pour le jour suivant, pour ne pas utiliser une information qu'on n'avait pas encore.

Projet en cours, je n'ai pas encore de résultats sur les vraies données.

Pour le lancer, `pip install -r requirements.txt` puis `python scripts/run_pipeline.py --demo`.

Issa Aziz Karambiri
