# M2 baseline execution and analysis

`m2_analysis.py` reuses the existing Naive Bayes pipelines to compare BoW,
TF-IDF, and a train-majority classifier on the same externally supplied split.
Only synthetic inputs have been used to validate these tools. They do not load,
clean, sample, or split Amazon data, train additional model types, integrate
Streamlit, tune parameters, or serialize fitted models.

## Run without any dataset

From the repository root, with the Python 3.11 environment active:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m m2_analysis --synthetic-demo
```

The demo writes to `outputs/m2/synthetic-demo/` (ignored by Git). It intentionally
includes a synthetic misclassification to exercise error export; these outputs
are not experimental results. Repeated exports require a new or empty directory
to prevent overwrites or stale files. For example:

```bash
python -m m2_analysis --synthetic-demo --output-dir outputs/m2/another-demo
```

`matplotlib` is the only added dependency and generates PNGs without a GUI.

## In-memory interface for the eventual data handoff

```python
from m2_analysis import run_baselines, export_results

# train_texts/train_labels and test_texts/test_labels come from the Data Engineer.
result = run_baselines(
    train_texts, train_labels, test_texts, test_labels,
    review_ids=test_review_ids,
    data_description="Describe the actual sample, cleaning version and split here",
    top_n=10,
)
export_results(result, "outputs/m2/my-run")
```

This is an interface sketch, not a real-data loader. Inputs must be aligned
sequences of nonblank strings and sentiment labels (`Negative`, `Neutral`,
`Positive`). All three training classes are required for this three-class
comparison. Test sets can omit classes. Optional review IDs must be unique,
nonblank strings aligned with test rows; without IDs, zero-based `test_index`
provides tracing. Always retain review text for error inspection.

The caller owns sample provenance, duplicate handling, and the shared 80/20
train/test split. No code here verifies cross-partition duplicate identity or
creates that split. Keep sampling seed 48 separate from any agreed split seed.
Do not include target ratings in input text. Agree on any validation/tuning
protocol within training data rather than optimizing against the held-out test.

## Returned results and exported outputs

`run_baselines` returns `metadata` and `runs` keyed by `majority`, `bow_nb`, and
`tfidf_nb`. Each run contains an in-memory `estimator`, `metrics`, `predictions`,
`errors`, and `features` (None for majority). Metadata records data description,
train/test sizes and class counts, scikit-learn version, and baseline settings.

- `analysis.json`: metadata, full evaluation, all prediction/error rows, and
  feature analysis; fitted estimators are excluded.
- `metrics.csv`: comparison table with accuracy and macro precision/recall/F1.
  Per-class precision/recall/F1 and ordered confusion matrices are in JSON.
- `<model>_predictions.csv` and `<model>_errors.csv`: test index, optional review
  ID, original supplied text, true label, and prediction. Empty error files still
  have headers. All errors are exported; selecting 5–10 representative examples,
  identifying patterns, and writing interpretations remain human work.
- `<model>_confusion.png`: true labels on rows, predictions on columns, always
  ordered Negative, Neutral, Positive (including absent classes).
- `bow_nb_features.png` and `tfidf_nb_features.png`: informative-feature charts
  from each fitted NB model, suitable for reviewing feature associations.

Majority uses only training-label counts via `DummyClassifier(most_frequent)`;
scikit-learn resolves tied counts by its sorted class order. Never substitute the
M1 sample's approximate 78% Positive share for this held-out evaluation.
Vocabulary and TF-IDF IDF are fitted on training text only. Predictions and
analysis do not refit either vectorizer.

`informative_features(pipeline, top_n=10)` reports vocabulary size and ranks
Positive features by `log P(token|Positive) - log P(token|Negative)`, reversing
that difference for Negative features. Each list contains at most `top_n`
strictly positive contrasts, with lexical tie-breaking; small vocabularies can
produce fewer than ten entries. Neutral is still a model class but is not part
of this pairwise explanation. These scores are neither word-frequency counts
nor causal effects, and they describe the fitted model, not every prediction.

Outputs include review text and must stay local: use the ignored `outputs/m2/`
folder. Custom output paths are not automatically ignored. Never commit real
reviews or generated results inadvertently. For M2, the Data Engineer and ML
Engineer must still agree on the cleaned-text interface, split, and shared
feature definitions. Real-data metrics, feature lists, 5–10 inspected errors,
and written conclusions remain pending that handoff.
