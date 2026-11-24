# ML baseline — Group Psi

CSE 3521, Autumn 2026, The Ohio State University. This contribution provides
dataset-independent building blocks for the original classical baseline, using
scikit-learn. M1 and M2 have been submitted. This module has been exercised on
the shared real-data split through
`run_m2_baseline.py`. See [M2 results](M2_BASELINE_RESULTS.md) for measured results.

## Implemented API

Import functions from the root `ml_baseline` module, running Python from the
repository root. No installation as a package is required.

| Function | Inputs | Output |
| --- | --- | --- |
| `rating_to_sentiment(rating)` | One integer star rating, 1–5 | `Negative` for 1–2, `Neutral` for 3, `Positive` for 4–5 |
| `create_vectorizer(representation="tfidf")` | `"bow"` or `"tfidf"` | A new, unfitted `CountVectorizer` or `TfidfVectorizer` |
| `create_nb_pipeline(representation="tfidf")` | `"bow"` or `"tfidf"` | A new, unfitted vectorizer + `MultinomialNB(alpha=1.0)` pipeline |
| `evaluate_predictions(y_true, y_pred)` | Nonempty, equal-length sequences of the three sentiment labels | A dictionary of accuracy, macro metrics, per-class metrics, class order, and a confusion matrix |

Ratings are not coerced: booleans, floats, strings, and missing values raise
`TypeError`; out-of-range integers raise `ValueError`. Evaluation rejects empty,
unequal-length, or unknown-label inputs with `ValueError`.

Both vectorizers use lowercase word unigrams, no stop-word list, and the default
scikit-learn tokenizer (tokens of at least two characters). BoW produces counts;
TF-IDF uses smoothed IDF and L2 normalization. These are initial reusable settings,
not tuned choices. Each call returns a fresh estimator.

Evaluation keys are `accuracy`, `macro_precision`, `macro_recall`, `macro_f1`,
`per_class`, `class_order`, and `confusion_matrix`. Per-class dictionaries contain
`precision`, `recall`, and `f1`. Confusion matrix rows are true labels and columns
are predicted labels, ordered **Negative, Neutral, Positive**. Macro metrics
always average over these three classes, including absent classes. Undefined
precision/recall/F1 values are zero. The dictionary uses ordinary Python numbers
and lists so downstream code can serialize it without conversion.

## Run the synthetic tests

After activating the Python 3.11 environment described in the root README:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -m "not integration and not e2e"
```

Tests use only in-memory text and labels; no Kaggle data, credentials, downloads,
or external services are used by the tests. They check rating validation, count
and TF-IDF features, unseen tokens, both pipeline configurations, and metrics
against hand-calculated expected values.

## In-memory example

Run from the repository root:

```python
from ml_baseline import create_nb_pipeline, evaluate_predictions

training_texts = ["awful stale food", "ordinary average food", "excellent tasty food"]
training_labels = ["Negative", "Neutral", "Positive"]
pipeline = create_nb_pipeline("bow")
pipeline.fit(training_texts, training_labels)
predictions = pipeline.predict(["awful food", "average food", "excellent food"])
metrics = evaluate_predictions(["Negative", "Neutral", "Positive"], predictions)
```

This example demonstrates the API only. Synthetic checks are not evidence of
real-world model quality.

## Data handoff and future work

Abderrachid Sghir is the Data Engineer responsible for data collection,
cleaning, and organization. M1 reports initial exploration of the required
**50,000-review sample with `random_state=48`**, including class imbalance and
duplicate reviews; see [the data guide](../data/README.md). These reported
findings are separate from this module's synthetic validation. Dataset files are not committed. Sampling, cleaning, and the shared split are
implemented in `scripts/`; see the data guide for preparation commands. The rating helper
defines the agreed label mapping without reading or modifying data.

The submitted M1 timeline planned a shared **80/20 train/test split** for M2,
alongside cleaning and BoW/TF-IDF preparation. Those shared files and duplicate
rules are now recorded in the data guide. Coordinate future changes to the
split, text/label interface and vectorizer definitions with Data and ML rather
than replacing them silently. The implemented M3 training-only validation
protocol is described in [M3 training](M3_TRAINING.md). The ML pipeline expects a sequence
of nonempty review strings and aligned sentiment label strings. Fit the entire
pipeline **only on training texts and labels**, then call `predict` on validation
or test texts. Never fit a vectorizer on the full dataset or refit it on held-out
data; this would leak vocabulary and, for TF-IDF, document-frequency information.
Training texts must contain tokens recognized by the vectorizer.
Ratings define the target labels and are not included as input features.

Initial real-data fitting and evaluation are recorded in [M2 results](M2_BASELINE_RESULTS.md). Logistic
Regression and the feedforward neural network are implemented in the separate
[M3 workflow](M3_TRAINING.md); its release-backed hosted staging app has been
[verified](M3_RESULTS.md#hosted-staging-status). The prepared-split M2 runner can
export a local BoW pipeline, and its predictions have been checked through the
existing Streamlit UI.
M2 assigns initial baseline training to Sang Park;
the submitted timeline targets comparison of all models and app integration in
M3, followed by final model results and deployment in M4.

## M2 orchestration and reporting

See [M2 analysis tools](M2_ANALYSIS.md) for the implemented shared-split runner,
train-majority comparison, metrics/figure exports, informative features, and
traceable prediction/error lists. These utilities reuse this module and are
validated with synthetic inputs and the shared prepared split; see the M2 results handoff.
