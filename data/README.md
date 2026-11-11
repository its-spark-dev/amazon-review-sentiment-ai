# Amazon Fine Food Reviews dataset

Obtain the dataset from the [Amazon Fine Food Reviews page on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews), published by SNAP. Review the dataset description and usage terms on the source page before using it.

## Local download (when needed)

1. Open the source page and sign in to Kaggle if prompted.
2. Download the dataset manually and extract the archive locally.
3. Place the extracted `Reviews.csv` at `data/Reviews.csv`.

The Streamlit placeholder and synthetic ML tests do not require any data. This repository does not download the dataset automatically, and no dataset or preparation script is tracked.

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
training text, then transform held-out text. The real-data preparation/split and
any additional validation protocol are not implemented in this repository.

## Storage and collaboration

- Keep downloaded archives, raw data, and processed datasets local. The root `.gitignore` excludes the contents of `data/` except this guide.
- Never commit Kaggle credentials, API tokens, or other secrets. No Kaggle API setup is required for the scaffold.
- Record preprocessing, the agreed sentiment mapping, split details, and data quality decisions in the project notebooks and report. The mapping and planned 80/20 split are documented above; cleaning and handoff details still need confirmation.
- Share the source link and reproducible processing instructions with contributors instead of committing the dataset.
