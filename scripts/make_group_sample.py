"""Generate the group's 50,000-review sample from the raw Amazon Fine Food Reviews data.

Draws a fixed-seed random sample from Reviews.csv (downloaded from
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) and writes it
to data/group_sample_seed<SEED>.csv by default.

Usage:
    python scripts/make_group_sample.py
    python scripts/make_group_sample.py --reviews path/to/Reviews.csv --out path/to/output.csv
"""

import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SEED = 48  # Group Psi
SAMPLE_SIZE = 50_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--reviews",
        type=Path,
        default=DATA_DIR / "Reviews.csv",
        help="path to the raw Reviews.csv (default: data/Reviews.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path (default: data/group_sample_seed<SEED>.csv)",
    )
    parser.add_argument("--seed", type=int, default=SEED, help=f"random seed (default: {SEED})")
    args = parser.parse_args()

    out_path = args.out or DATA_DIR / f"group_sample_seed{args.seed}.csv"

    if not args.reviews.exists():
        parser.error(f"Reviews file not found: {args.reviews}")

    df = pd.read_csv(args.reviews)
    sample = df.sample(n=SAMPLE_SIZE, random_state=args.seed)
    sample.to_csv(out_path, index=False)

    print(f"Saved {len(sample)} rows to {out_path}")
    print(sample[["Score", "Text"]].head(3))


if __name__ == "__main__":
    main()
