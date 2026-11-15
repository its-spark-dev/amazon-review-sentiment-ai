"""Split the cleaned reviews into the shared train and test sets.

Reads the output of `clean_reviews.py` and writes an 80/20 split, stratified by
`Sentiment`. Rows whose cleaned `Text` is identical always land on the same
side, so a repeated review cannot leak from training into test data.

Usage:
    python data/split_reviews.py
    python data/split_reviews.py --src path/to/cleaned.csv --train path/to/train.csv --test path/to/test.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent

SEED = 48  # Group Psi
TEST_SIZE = 0.2
DEFAULT_SRC = DATA_DIR / "cleaned_reviews.csv"
DEFAULT_TRAIN = DATA_DIR / "train_reviews.csv"
DEFAULT_TEST = DATA_DIR / "test_reviews.csv"


def split(
    df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) as new frames with the same columns and row order.

    The split is drawn over distinct `Text` values, stratified by the
    `Sentiment` of each text's first row, and every row then follows its text.
    Sizes and class proportions are therefore exact for distinct texts and can
    be off by a few rows overall when texts repeat.
    """
    first = df.drop_duplicates("Text")
    _, test_first = train_test_split(
        first, test_size=test_size, random_state=seed, stratify=first["Sentiment"]
    )
    in_test = df["Text"].isin(test_first["Text"])
    return df[~in_test].reset_index(drop=True), df[in_test].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_SRC,
        help=f"cleaned input CSV (default: data/{DEFAULT_SRC.name})",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=DEFAULT_TRAIN,
        help=f"training output CSV (default: data/{DEFAULT_TRAIN.name})",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=DEFAULT_TEST,
        help=f"test output CSV (default: data/{DEFAULT_TEST.name})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help=f"fraction of reviews held out for testing (default: {TEST_SIZE})",
    )
    parser.add_argument("--seed", type=int, default=SEED, help=f"random seed (default: {SEED})")
    args = parser.parse_args()

    if not args.src.exists():
        parser.error(f"Source file not found: {args.src}")
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")

    df = pd.read_csv(args.src)
    train, test = split(df, test_size=args.test_size, seed=args.seed)
    train.to_csv(args.train, index=False)
    test.to_csv(args.test, index=False)

    print(f"Read {len(df)} rows from {args.src}")
    for name, part, path in (("train", train, args.train), ("test", test, args.test)):
        print(f"Saved {len(part)} {name} rows ({len(part) / len(df):.1%}) to {path}")
        print(part["Sentiment"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
