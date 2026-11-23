"""Deterministic synthetic review data shared by integration and E2E tests.

Every class has its own vocabulary so tiny models learn a clean signal; no real
dataset text is used anywhere.
"""

from pathlib import Path
import random

import joblib
import pandas as pd

from ml_baseline import create_nb_pipeline

VOCAB = {
    "Positive": ["delicious", "fantastic", "love", "fresh", "crunchy", "tasty", "wonderful", "yum"],
    "Neutral": ["okay", "average", "fine", "decent", "plain", "mediocre", "acceptable", "ordinary"],
    "Negative": ["awful", "stale", "terrible", "gross", "rotten", "disgusting", "waste", "bad"],
}
SCORES = {"Positive": (4, 5), "Neutral": (3,), "Negative": (1, 2)}
FILLER = ["the", "this", "snack", "coffee", "product", "flavor", "was", "really", "and", "bag"]
PROBES = {
    "Positive": "delicious fantastic fresh tasty love it",
    "Neutral": "okay average fine decent plain",
    "Negative": "awful stale terrible gross waste",
}


def review_text(rng: random.Random, sentiment: str) -> str:
    words = rng.sample(VOCAB[sentiment], 4) + rng.sample(FILLER, 3)
    rng.shuffle(words)
    return " ".join(words).capitalize() + "."


def raw_reviews(n_per_class: int = 60, seed: int = 0) -> pd.DataFrame:
    """Rows shaped like Kaggle's Reviews.csv, plus rows the cleaner must drop."""
    rng = random.Random(seed)
    rows = []
    for sentiment in VOCAB:
        for _ in range(n_per_class):
            i = len(rows) + 1
            rows.append({
                "Id": i, "ProductId": f"B{i:09d}", "UserId": f"U{i}", "ProfileName": f"user{i}",
                "HelpfulnessNumerator": 0, "HelpfulnessDenominator": 0,
                "Score": rng.choice(SCORES[sentiment]), "Time": 1_300_000_000 + i,
                "Summary": sentiment, "Text": f"<br />{review_text(rng, sentiment)} &amp; more",
            })
    base = dict(rows[0])
    n = len(rows)
    rows += [
        {**base, "Id": n + 1, "ProductId": "BDUPLICATE"},           # same review, other variant
        {**base, "Id": n + 2, "UserId": "UBLANK", "Text": "   "},   # blank text
        {**base, "Id": n + 3, "UserId": "UBAD", "Score": 9},        # invalid score
    ]
    return pd.DataFrame(rows)


def write_raw_reviews(path: Path, **kwargs) -> Path:
    raw_reviews(**kwargs).to_csv(path, index=False)
    return path


def train_texts(n_per_class: int = 30, seed: int = 1) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    texts, labels = [], []
    for sentiment in VOCAB:
        for _ in range(n_per_class):
            texts.append(review_text(rng, sentiment).lower())
            labels.append(sentiment)
    return texts, labels


FLIP = {"Positive": "Negative", "Negative": "Positive", "Neutral": "Neutral"}


def save_model(path: Path, representation: str = "bow", flipped: bool = False) -> Path:
    """Fit a small NB pipeline on synthetic text and save it like the real runner.

    ``flipped`` swaps Positive/Negative labels, giving a second model whose predictions
    visibly differ, so tests can tell which model actually served a request.
    """
    texts, labels = train_texts()
    if flipped:
        labels = [FLIP[label] for label in labels]
    joblib.dump(create_nb_pipeline(representation).fit(texts, labels), path)
    return path


def vega_charts(at) -> list:
    """Vega-Lite chart elements in an AppTest run, across Streamlit versions.

    Streamlit 1.40 reports `arrow_vega_lite_chart`; newer releases `vega_lite_chart`.
    """
    return [*at.get("arrow_vega_lite_chart"), *at.get("vega_lite_chart")]
