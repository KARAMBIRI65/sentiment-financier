"""Modèles de sentiment : FinBERT et deux références simples pour comparer.

Chaque modèle expose `predict_proba(texts) -> DataFrame[positive, negative, neutral]`.
Le score de sentiment d'un titre est P(positif) - P(négatif), entre -1 et +1.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["positive", "negative", "neutral"]


def sentiment_score(proba: pd.DataFrame) -> pd.Series:
    return proba["positive"] - proba["negative"]


def predict_label(proba: pd.DataFrame) -> pd.Series:
    return proba[LABELS].idxmax(axis=1)


# ---------------------------------------------------------------------------
# FinBERT (Araci, 2019) : BERT ré-entraîné sur des textes financiers
# ---------------------------------------------------------------------------
class FinBertScorer:
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        batch_size: int = 32,
        max_length: int = 64,
        device: str | None = None,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.batch_size = batch_size
        self.max_length = max_length
        # Ordre des sorties du modèle (ProsusAI/finbert : positive, negative, neutral)
        self.labels = [self.model.config.id2label[i].lower() for i in range(self.model.config.num_labels)]

    def predict_proba(self, texts) -> pd.DataFrame:
        texts = list(texts)
        chunks = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(
                    texts[i : i + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                probs = self.torch.softmax(self.model(**batch).logits, dim=-1)
                chunks.append(probs.cpu().numpy())
        arr = np.vstack(chunks) if chunks else np.empty((0, len(self.labels)))
        return pd.DataFrame(arr, columns=self.labels)[LABELS]


# ---------------------------------------------------------------------------
# Référence 1 : TF-IDF + régression logistique (entraînée sur PhraseBank)
# ---------------------------------------------------------------------------
class TfidfLogRegScorer:
    def __init__(self, C: float = 4.0):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        self.pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(C=C, max_iter=2000, class_weight="balanced"),
        )

    def fit(self, texts, labels):
        self.pipe.fit(list(texts), list(labels))
        return self

    def predict_proba(self, texts) -> pd.DataFrame:
        proba = self.pipe.predict_proba(list(texts))
        df = pd.DataFrame(proba, columns=self.pipe.classes_)
        for lab in LABELS:
            if lab not in df:
                df[lab] = 0.0
        return df[LABELS]


# ---------------------------------------------------------------------------
# Référence 2 : mini-lexique (compter les mots positifs et négatifs)
# ---------------------------------------------------------------------------
POSITIVE_WORDS = {
    "beat", "beats", "gain", "gains", "growth", "grow", "grows", "higher", "increase",
    "increased", "jump", "jumps", "outperform", "profit", "profits", "raise", "raises",
    "rally", "record", "rise", "rises", "rose", "soar", "soars", "strong", "surge",
    "surges", "up", "upgrade", "upgraded", "upgrades", "buy", "bullish", "improve",
    "improved", "exceed", "exceeds", "positive", "boost", "boosts",
}
NEGATIVE_WORDS = {
    "cut", "cuts", "decline", "declined", "declines", "decrease", "down", "downgrade",
    "downgraded", "downgrades", "drop", "drops", "fall", "falls", "fell", "loss",
    "losses", "lower", "miss", "misses", "plunge", "plunges", "sell", "slump", "weak",
    "warning", "warns", "bearish", "lawsuit", "fraud", "bankruptcy", "layoffs",
    "negative", "underperform", "tumble", "tumbles", "sink", "sinks", "crash",
}
_WORD = re.compile(r"[a-z']+")


class LexiconScorer:
    """Référence naïve : utile pour montrer ce que FinBERT apporte en plus."""

    def predict_proba(self, texts) -> pd.DataFrame:
        rows = []
        for t in texts:
            words = _WORD.findall(str(t).lower())
            pos = sum(w in POSITIVE_WORDS for w in words)
            neg = sum(w in NEGATIVE_WORDS for w in words)
            if pos == neg:
                rows.append((0.0, 0.0, 1.0))
            elif pos > neg:
                rows.append((1.0, 0.0, 0.0))
            else:
                rows.append((0.0, 1.0, 0.0))
        return pd.DataFrame(rows, columns=LABELS)


# ---------------------------------------------------------------------------
# Cache : FinBERT est lent sur CPU, on ne score chaque titre qu'une fois
# ---------------------------------------------------------------------------
def score_with_cache(texts: pd.Series, scorer, cache_path: str | Path | None = None) -> pd.DataFrame:
    """Renvoie les probabilités pour chaque texte (même index que `texts`)."""
    unique = pd.Series(texts.unique())
    cached = pd.DataFrame(columns=["text", *LABELS])
    if cache_path and Path(cache_path).exists():
        cached = pd.read_csv(cache_path)
    todo = unique[~unique.isin(cached["text"])]
    if len(todo):
        new = scorer.predict_proba(todo.tolist())
        new.insert(0, "text", todo.values)
        cached = pd.concat([cached, new], ignore_index=True) if len(cached) else new
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            cached.to_csv(cache_path, index=False)
    lookup = cached.drop_duplicates("text").set_index("text")[LABELS]
    out = lookup.reindex(texts.values)
    out.index = texts.index
    return out.astype(float)
