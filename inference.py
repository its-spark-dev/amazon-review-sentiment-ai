"""Model loading and single-review inference for the Streamlit app.

The app never trains models. It loads fitted scikit-learn pipelines that the
ML workflow saved locally: every ``*.joblib`` file in ``models/`` (or the
directory in ``SENTIMENT_MODEL_DIR``), or only the single file named by
``SENTIMENT_MODEL_PATH`` when that is set. Only load model files your team
produced: joblib/pickle files can execute code when loaded.

``ensure_models_release`` fetches the approved artifacts for a GitHub release
tag. Expected SHA-256 hashes live in the tracked ``models/release_manifest.json``
and every downloaded file is verified before it replaces anything on disk.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import urllib.parse
import urllib.request

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from ml_baseline import CLASS_LABELS
from scripts.clean_reviews import clean_text

MODELS_DIR = Path(__file__).parent / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "sentiment_pipeline.joblib"
MAX_REVIEW_CHARS = 5000
RELEASE_MANIFEST = MODELS_DIR / "release_manifest.json"
REPORT_MODEL_RELEASE = "v0.2.0-rc2"
RELEASE_URL = "https://github.com/its-spark-dev/amazon-review-sentiment-ai/releases/download/{tag}/{name}"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CHUNK = 1 << 20

logger = logging.getLogger(__name__)


def model_path() -> Path:
    """Return the configured model path (env override, else the default)."""
    return Path(os.environ.get("SENTIMENT_MODEL_PATH", DEFAULT_MODEL_PATH))


def model_paths() -> list[Path]:
    """Return the available model artifacts, default model first.

    ``SENTIMENT_MODEL_PATH`` restricts the app to that one file. Otherwise every
    ``*.joblib`` file in ``SENTIMENT_MODEL_DIR`` (default ``models/``) is listed
    by name, with ``sentiment_pipeline.joblib`` first when present.
    """
    if "SENTIMENT_MODEL_PATH" in os.environ:
        path = model_path()
        return [path] if path.is_file() else []
    directory = Path(os.environ.get("SENTIMENT_MODEL_DIR", MODELS_DIR))
    paths = sorted(p for p in directory.glob("*.joblib") if p.is_file())
    return sorted(paths, key=lambda p: p.name != DEFAULT_MODEL_PATH.name)


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
    does not expose ``predict_proba``. For BoW/TF-IDF pipelines, raises
    ValueError before classification if no vocabulary features are recognized.
    This checks the fitted vocabulary, not the input's language or reliability.
    """
    text = validate_review(text)
    text = clean_text(text)
    if not text:
        raise ValueError("Enter a review containing text after HTML cleanup.")
    # TfidfVectorizer inherits CountVectorizer. Transform only: never fit or
    # change the release pipeline when checking a single review.
    if isinstance(model, Pipeline) and isinstance(model.steps[0][1], CountVectorizer):
        if model.steps[0][1].transform([text]).nnz == 0:
            raise ValueError(
                "No words in this review are recognized by the selected model. "
                "Try an English food review with more detail."
            )
    label = str(model.predict([text])[0])
    probabilities = None
    if hasattr(model, "predict_proba"):
        row = model.predict_proba([text])[0]
        by_class = dict(zip((str(c) for c in model.classes_), row))
        probabilities = {c: float(by_class.get(c, 0.0)) for c in CLASS_LABELS}
    return {"label": label, "probabilities": probabilities}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _release_entries(tag: str, manifest_path: Path) -> dict[str, str]:
    """Return ``{filename: sha256}`` for ``tag``, or raise ValueError if it is unknown or malformed."""
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    entries = manifest.get(tag) if isinstance(manifest, dict) else None
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"unknown model release tag {tag!r} in {manifest_path}")
    for name, expected in entries.items():
        if Path(name).name != name or not name.endswith(".joblib"):
            raise ValueError(f"invalid model filename {name!r} for release {tag!r}")
        if not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
            raise ValueError(f"invalid sha256 for {name!r} in release {tag!r}")
    return entries


def _download_verified(url: str, target: Path, expected: str, timeout: float) -> None:
    """Download ``url`` next to ``target``, check its hash, then atomically move it into place."""
    # Same directory keeps os.replace atomic; ".part" keeps it out of the *.joblib glob.
    tmp = tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", suffix=".part", delete=False)
    tmp_path = Path(tmp.name)
    try:
        digest = hashlib.sha256()
        with tmp, urllib.request.urlopen(url, timeout=timeout) as response:
            while chunk := response.read(_CHUNK):
                digest.update(chunk)
                tmp.write(chunk)
            tmp.flush()
            os.fsync(tmp.fileno())
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(f"{target.name}: sha256 {actual} does not match expected {expected}")
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def ensure_models_release(
    tag: str,
    model_dir: Path = MODELS_DIR,
    *,
    manifest_path: Path = RELEASE_MANIFEST,
    timeout: float = 60,
) -> list[Path]:
    """Make ``model_dir`` hold the verified artifacts pinned to release ``tag``.

    Files whose SHA-256 already matches the manifest are reused; anything missing
    or different is downloaded from the release, verified and atomically swapped
    in. Failures are logged with the filename, then raised as ``ValueError`` (bad
    tag, manifest or hash) or ``OSError`` (filesystem or network).
    """
    try:
        entries = _release_entries(tag, manifest_path)
    except (OSError, ValueError) as exc:
        logger.error("Model release %s: cannot read manifest %s: %s", tag, manifest_path, exc)
        raise
    model_dir = Path(model_dir)
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Model release %s: cannot create %s: %s", tag, model_dir, exc)
        raise

    paths = []
    for name, expected in entries.items():
        target = model_dir / name
        url = RELEASE_URL.format(tag=urllib.parse.quote(tag, safe=""), name=name)
        try:
            if target.is_file() and _sha256(target) == expected:
                logger.debug("Model release %s: %s already verified", tag, name)
            else:
                logger.info("Model release %s: downloading %s", tag, name)
                _download_verified(url, target, expected, timeout)
        except (OSError, ValueError) as exc:
            logger.error("Model release %s: failed to prepare %s from %s: %s", tag, name, url, exc)
            raise
        paths.append(target)
    return paths
