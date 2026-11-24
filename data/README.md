# Amazon Fine Food Reviews dataset

Obtain the dataset from the [Amazon Fine Food Reviews page on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews), published by SNAP. Review the dataset description and usage terms on the source page before using it.

## Local download (when needed)

1. Open the source page and sign in to Kaggle if prompted.
2. Download the dataset manually and extract the archive locally.
3. Place the extracted `Reviews.csv` at `data/Reviews.csv`.

The release-backed Streamlit demo and synthetic ML tests do not require the dataset.
The demo downloads approved model artifacts, not reviews. This repository does not
download the dataset automatically, and no dataset is tracked. Preparation scripts are versioned in `scripts/`;
raw and generated CSV files belong in `data/`. Run preparation commands from
the repository root.

## Group Psi sample and submitted M1 findings

The project sample is **50,000 reviews**, using **`random_state=48`**. The
submitted Milestone 1 charter reports the following initial exploration. These
values are attributed to that report; they have not been recomputed from data
in this repository and do not describe a finalized cleaned training set.

| Finding | Reported value |
| --- | --- |
| Sentiment distribution | 14.4% Negative, 7.6% Neutral, 78.0% Positive (rounded) |
| Review length, words | Mean 79.93; median 56; maximum 1,901 |
| Review length, characters | Mean 434.46; median 302; maximum 11,321 |
| Duplicate rows repeating an earlier row | 3,720, matching user, profile name, timestamp, and review text |
| Missing values | 1 profile name and 2 summaries; no missing or blank review text |
| Short reviews | No review text shorter than 5 words |

M1 notes that most duplicate groups span different `ProductId` values, consistent
with product variants sharing reviews. Duplicate handling must be resolved before
the shared split so repeated reviews do not leak between training and test data.
The skew toward Positive motivates the planned majority-class comparison and
macro/per-class evaluation; no model performance is reported here.

## M2 data handoff

Abderrachid Sghir is the Data Engineer. The submitted timeline assigns data
cleaning, two feature representations (BoW and TF-IDF), and a shared **80/20
train/test split** to the Data Engineer for M2. Coordinate use of the existing
vectorizer factories in `ml_baseline.py` with the ML Engineer rather than
maintaining divergent feature definitions.

Confirm the cleaned review text fields, aligned sentiment labels, duplicate
policy, and reproducible split details before real-data fitting. Ratings 1–2
map to Negative, 3 to Neutral, and 4–5 to Positive; ratings provide the target,
not a feature for the current text baseline. Fit vocabulary and IDF only on
training text, then transform held-out text. Cleaning and the split are
implemented by the scripts below. M3 adds a training-only grouped validation
partition, documented in [M3 training](../docs/M3_TRAINING.md), while preserving
the existing outer train/test files.

### Preparation commands

Run from the repository root. All scripts default to reading and writing CSV
files in the repository's `data/` directory:

```bash
python scripts/make_group_sample.py
python scripts/clean_reviews.py
python scripts/split_reviews.py
```

Sampling selects 50,000 reviews with seed 48. Default data paths are resolved
relative to the repository, independent of the current working directory.
Use each script's `--help` for optional input and output path overrides;
relative override paths are resolved from the current working directory.

### Cleaning rules (`scripts/clean_reviews.py`)

The cleaning command above reads `data/group_sample_seed48.csv` and writes
`data/cleaned_reviews.csv`. It uses only the sample, not the full `Reviews.csv`.
In order, it:

1. Drops rows with missing or blank `Text`, or a `Score` that is not an integer 1–5.
2. Drops every row of a duplicate group (same `UserId`, `ProfileName`, `Time`,
   and raw `Text`) whose ratings disagree, since the label is ambiguous.
3. Keeps the first row of each remaining duplicate group.
4. Normalizes `Text`: HTML tags become a space, HTML entities are unescaped,
   whitespace is collapsed and stripped, and the text is lowercased. Punctuation,
   URLs, and digits are kept. Other columns are unchanged.
5. Adds a `Sentiment` column from `Score` using `rating_to_sentiment`.

On the Group Psi sample this leaves 46,278 of 50,000 reviews (36,080 Positive,
6,655 Negative, 3,543 Neutral).

Known limitations of these rules on the sample:

- **22 cleaned texts still appear more than once (49 rows, 27 beyond the first
  of each).** Every group comes from a single user. In 13 groups the user
  posted the same text at a different `Time`, and in 4 the `ProfileName`
  differs, so the rows never matched the duplicate key. In the other 5 the key
  matched except for raw `Text` differences (for example markup or spacing)
  that step 4 removed. Two groups
  have differing ratings, and in one of them the labels differ (ratings 4 and
  3, Positive and Neutral). Cleaning keeps all of these rows; the split below
  keeps each group on one side so they cannot leak between training and test
  data.
- **Angle-bracket asides such as `<sigh>` and `<grin>` are kept.** Reviewers
  typed these as text (stored as `&lt;sigh&gt;` in the raw data), so they are
  not HTML markup and are not stripped; 12 cleaned reviews contain them.

### Train/test split (`scripts/split_reviews.py`)

The split command above reads `data/cleaned_reviews.csv` and writes
`data/train_reviews.csv` and `data/test_reviews.csv`, with the same columns and
row order as the cleaned file. The split is:

- **80/20**, with `random_state=48`, so rerunning it gives identical files.
- **Stratified by `Sentiment`.**
- **Grouped by cleaned `Text`.** The split is drawn over distinct texts, and
  every row follows its text, so the repeated texts described above are never
  on both sides. No rows are dropped.

On the Group Psi sample:

| Set | Rows | Positive | Negative | Neutral |
| --- | --- | --- | --- | --- |
| Train | 37,023 | 28,864 (78.0%) | 5,324 (14.4%) | 2,835 (7.7%) |
| Test | 9,255 | 7,216 (78.0%) | 1,331 (14.4%) | 708 (7.6%) |

The two files share no `Id` and no `Text`. Fit vectorizers and models on the
training file only.

## Storage and collaboration

- Keep downloaded archives, raw data, and processed datasets local. The root `.gitignore` excludes the contents of `data/` except this guide.
- Never commit Kaggle credentials, API tokens, or other secrets. No Kaggle API setup is required for the scaffold.
- Record preprocessing, the agreed sentiment mapping, split details, and data quality decisions in the project report. The shared CSV interface, hashes and measured run provenance are documented in the M2/M3 results; coordinate any future changes rather than silently replacing those files.
- Share the source link and reproducible processing instructions with contributors instead of committing the dataset.
