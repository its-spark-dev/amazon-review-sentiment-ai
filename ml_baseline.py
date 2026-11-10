"""Dataset-independent sentiment baseline for CSE 3521 Group Psi.

Call ``fit`` only with the shared training partition, then ``predict`` on
validation/test texts. Data preparation and split creation belong upstream.
"""

from numbers import Integral
from typing import Literal, Sequence

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


CLASS_LABELS = ("Negative", "Neutral", "Positive")


def rating_to_sentiment(rating: int) -> str:
    """Map integer ratings 1–2/3/4–5 to Negative/Neutral/Positive.

    Raise TypeError for non-integers (including bool) and ValueError for
    integers outside 1–5. Ratings are never rounded or coerced from strings.
    """
    if isinstance(rating, bool) or not isinstance(rating, Integral):
        raise TypeError("rating must be an integer from 1 to 5")
    if not 1 <= rating <= 5:
        raise ValueError("rating must be from 1 to 5")
    if rating <= 2:
        return "Negative"
    if rating == 3:
        return "Neutral"
    return "Positive"


def create_vectorizer(
    representation: Literal["bow", "tfidf"] = "tfidf",
) -> CountVectorizer | TfidfVectorizer:
    """Return a fresh, unfitted unigram vectorizer with nonnegative features.

    Both configurations lowercase tokens and retain stop words. BoW returns
    token counts; TF-IDF uses IDF smoothing and L2 normalization. Tokenization
    uses scikit-learn's default word pattern (tokens of at least two characters).
    These are initial baseline settings, not tuned modeling decisions.
    """
    if representation == "bow":
        return CountVectorizer(lowercase=True, ngram_range=(1, 1), stop_words=None)
    if representation == "tfidf":
        return TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 1),
            stop_words=None,
            use_idf=True,
            smooth_idf=True,
            norm="l2",
        )
    raise ValueError("representation must be 'bow' or 'tfidf'")


def create_nb_pipeline(
    representation: Literal["bow", "tfidf"] = "tfidf",
) -> Pipeline:
    """Return an unfitted text vectorizer → Multinomial Naive Bayes pipeline.

    ``fit(training_texts, training_labels)`` learns vocabulary, IDF (if used),
    and classifier together. ``predict(held_out_texts)`` only transforms texts.
    """
    return Pipeline(
        [
            ("vectorizer", create_vectorizer(representation)),
            ("classifier", MultinomialNB(alpha=1.0)),
        ]
    )


def evaluate_predictions(
    y_true: Sequence[str], y_pred: Sequence[str]
) -> dict:
    """Return metrics for nonempty, equally sized sequences of sentiment labels.

    Output contains accuracy, macro_precision, macro_recall, macro_f1,
    per_class (precision/recall/f1 by label), class_order, and confusion_matrix.
    Matrix rows are true labels; columns are predictions, in CLASS_LABELS order.
    All three classes participate in macro averages. Undefined metrics are 0.
    Values are plain Python floats/lists/dicts for downstream reporting.
    Unknown labels, empty inputs, or unequal lengths raise ValueError.
    """
    if isinstance(y_true, str) or isinstance(y_pred, str):
        raise ValueError("inputs must be sequences of labels, not strings")
    true_labels, predicted_labels = list(y_true), list(y_pred)
    if not true_labels or len(true_labels) != len(predicted_labels):
        raise ValueError("inputs must be nonempty and have equal lengths")
    if any(label not in CLASS_LABELS for label in true_labels + predicted_labels):
        raise ValueError(f"labels must belong to {CLASS_LABELS}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=CLASS_LABELS, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, label in enumerate(CLASS_LABELS)
        },
        "class_order": list(CLASS_LABELS),
        "confusion_matrix": confusion_matrix(
            true_labels, predicted_labels, labels=CLASS_LABELS
        ).tolist(),
    }
