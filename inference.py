"""Model loading and single-review inference for the Streamlit app.

The app never trains models. It loads a fitted scikit-learn pipeline that the
ML workflow saved locally (default ``models/sentiment_pipeline.joblib``, or the
path in ``SENTIMENT_MODEL_PATH``). Only load model files your team produced:
joblib/pickle files can execute code when loaded.
"""

import os
from pathlib import Path

import joblib
from sklearn.utils.validation import check_is_fitted

from ml_baseline import CLASS_LABELS

DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "sentiment_pipeline.joblib"
MAX_REVIEW_CHARS = 5000


def model_path() -> Path:
    """Return the configured model path (env override, else the default)."""
    return Path(os.environ.get("SENTIMENT_MODEL_PATH", DEFAULT_MODEL_PATH))


def load_model(path: Path):
    """Load a fitted pipeline, or return None if the file does not exist."""
    path = Path(path)
    if not path.is_file():
        return None
    model = joblib.load(path)
    check_is_fitted(model)
    return model


def validate_review(text) -> str:
    """Return stripped review text or raise ValueError with a user-facing message."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a review to analyze.")
    text = text.strip()
    if len(text) > MAX_REVIEW_CHARS:
        raise ValueError(f"Reviews are limited to {MAX_REVIEW_CHARS:,} characters.")
    return text


def predict_review(model, text: str) -> dict:
    """Predict one review's sentiment.

    Returns ``{"label": str, "probabilities": {label: float} | None}``.
    Probabilities follow ``CLASS_LABELS`` order and are None if the model
    does not expose ``predict_proba``.
    """
    text = validate_review(text)
    label = str(model.predict([text])[0])
    probabilities = None
    if hasattr(model, "predict_proba"):
        row = model.predict_proba([text])[0]
        by_class = dict(zip((str(c) for c in model.classes_), row))
        probabilities = {c: float(by_class.get(c, 0.0)) for c in CLASS_LABELS}
    return {"label": label, "probabilities": probabilities}
