"""Clean the group's review sample.

Reads the sampled reviews, applies the cleaning steps in `clean`, and writes
the result. `split_reviews.py` then splits that result into train and test.

Usage:
    python scripts/clean_reviews.py
    python scripts/clean_reviews.py --src path/to/sample.csv --dst path/to/cleaned.csv
"""

import argparse
import html
from pathlib import Path
import re
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for ml_baseline

from ml_baseline import rating_to_sentiment  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SEED = 48  # Group Psi
DEFAULT_SRC = DATA_DIR / f"group_sample_seed{SEED}.csv"
DEFAULT_DST = DATA_DIR / "cleaned_reviews.csv"


# Rows matching on all of these (raw text) are the same review, usually posted
# on several product variants.
DEDUP_KEY = ["UserId", "ProfileName", "Time", "Text"]

_TAG = re.compile(r"</?[a-zA-Z][^>]*>")  # real tags only; keeps "<3" and "a < b"
_WHITESPACE = re.compile(r"\s+")
_SENTIMENT = {score: rating_to_sentiment(score) for score in range(1, 6)}


def clean_text(text: str) -> str:
    """Strip HTML tags and entities, collapse whitespace, and lowercase."""
    text = _TAG.sub(" ", text)
    text = html.unescape(text)  # after tag removal, so "&lt;3" stays "<3"
    return _WHITESPACE.sub(" ", text).strip().lower()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return the cleaned reviews as a new frame with a `Sentiment` column.

    Steps, in order:
    1. Drop rows with missing/blank Text or a Score that is not an integer 1-5.
    2. Drop every row of a DEDUP_KEY group whose Scores disagree (ambiguous label).
    3. Keep the first row of each DEDUP_KEY group.
    4. Normalize Text with `clean_text`; drop rows left blank.
    5. Add Sentiment (1-2 Negative, 3 Neutral, 4-5 Positive) and reset the index.
    """
    score = pd.to_numeric(df["Score"], errors="coerce")
    has_text = df["Text"].notna() & df["Text"].astype("string").str.strip().ne("").fillna(False)
    out = df[has_text & score.isin(_SENTIMENT)].copy()
    out["Score"] = out["Score"].astype(int)

    distinct_scores = out.groupby(DEDUP_KEY, dropna=False)["Score"].transform("nunique")
    out = out[distinct_scores == 1].drop_duplicates(DEDUP_KEY)

    out["Text"] = out["Text"].map(clean_text)
    out = out[out["Text"] != ""]

    out["Sentiment"] = out["Score"].map(_SENTIMENT)
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_SRC,
        help=f"input CSV (default: data/{DEFAULT_SRC.name})",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=DEFAULT_DST,
        help=f"output CSV (default: data/{DEFAULT_DST.name})",
    )
    args = parser.parse_args()

    if not args.src.exists():
        parser.error(f"Source file not found: {args.src}")

    df = pd.read_csv(args.src)
    cleaned = clean(df)
    cleaned.to_csv(args.dst, index=False)

    print(f"Read {len(df)} rows from {args.src}")
    print(f"Saved {len(cleaned)} rows to {args.dst}")
    print(cleaned["Sentiment"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
