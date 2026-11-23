# M3 model training and comparison

This workflow extends M2 with Logistic Regression and a feedforward MLP. It does
not modify the app or the existing shared test CSV. Run from the repository root
with Python 3.11 and the existing requirements installed:

```bash
python -m unittest discover -s tests -v
python scripts/run_m3_comparison.py
```

Local prerequisites: `data/train_reviews.csv` and `data/test_reviews.csv` generated
by the unchanged preparation scripts. Both require `Id`, `Text`, `Sentiment`;
optional Score is checked for consistency. The runner rejects ID/Text overlap.
No dataset is downloaded, resampled, cleaned, or repartitioned by this runner.

## Selection protocol

1. Load the outer training file and create a fixed, seed-48, approximately 80/20
   inner fit/validation split using the shared grouped-text splitting function.
   Identical Text values stay together. Stratification uses each distinct text's
   first label; row-level class proportions are approximate. Split IDs are saved
   locally in `selection.json`. They are not committed.
2. Fit each vocabulary/IDF only on inner-fit text, then transform validation text.
3. Use the same maximum 10,000 word-unigram features in M3 comparisons, with
   lowercase/default tokenization and no stop-word list. TF-IDF retains its
   smoothed IDF/L2 defaults. This is an explicitly different feature budget from
   the preserved M2 36,995-feature experiment. Tied counts are resolved lexically; vocabulary selection is training-only.
   No dense corpus conversion is used.
4. Select LR separately for BoW and TF-IDF over C={0.5,2.0} and class_weight=
   {None,balanced}. Solver is lbfgs with max_iter=5000; nonconvergence raises an
   error instead of being silently accepted. The iteration budget was raised
   after a validation-stage convergence failure, before any M3 test evaluation.
5. Fit one TF-IDF MLP architecture: input -> 32 ReLU hidden units -> three-class
   softmax output, Adam, alpha=0.001, learning rate=0.001, batch size up to 256,
   shuffle=True, seed 48. Evaluate epochs 1–15 and select by validation Macro-F1.
   This is a feedforward neural network; it does not model word order.
6. Score NB and majority on the same validation split. Recommend a model by
   validation Macro-F1 across all six configurations. Candidate-order and
   earliest-epoch ties are deterministic. Save all selection decisions BEFORE
   final fitting/test evaluation. The test labels never enter selection.
7. Refit each chosen configuration on the full outer training file. MLP uses the
   selected epoch count and a freshly fitted full-training vectorizer. Evaluate
   the original shared test only after the settings are frozen.

The M2 test set was already inspected. M3 reuses it for consistent comparison;
it is not a new, never-observed dataset. Do not iterate on M3 test scores. Use
training-only validation for further choices, and disclose any follow-up changes.

The fixed small hidden layer was chosen before test evaluation to introduce
nonlinear feature combinations while bounding parameter count and local CPU cost.
Hidden width was not searched, so this is not a claim that 32 units is optimal.

## Loss curves and timing

Epoch curves are calculated after each update over the full inner-fit and
validation partitions using `predict_proba` and `log_loss`. Both curves are
unregularized cross-entropy on the same feature space, not a mix of training
objective and validation accuracy. Epoch selection uses Macro-F1, not loss.
The final full-training refit is separate from these diagnostic curves.

Every model's reported training time includes its own training-vectorizer
fit/transform and classifier fit. It excludes model selection, validation loss
logging, test prediction, plotting and serialization. The majority baseline has
no vectorizer. All numerical fitting uses threadpool_limits(1). Timings describe
one local run, not hardware-independent benchmark claims. Selection time is
reported separately.

## Outputs

Default: `outputs/m3/comparison/` (ignored). Existing nonempty output directories
are refused. Override with `--output-dir`; `--epochs`, `--max-features` and
`--seed` define a new experiment and must be decided before looking at test data.

- `selection.json`: inner IDs, complete LR candidate scores, MLP epoch history,
  selected settings and validation-based recommendation.
- `comparison.md`: accuracy, macro precision/recall/F1 and training seconds.
- `mlp_loss.png`: real training/validation loss, with selected epoch marked.
- `evaluation/analysis.json`: all metrics, per-class results, confusion matrices,
  predictions/errors, input hashes, environment and full timing/selection protocol.
- `evaluation/*_confusion.png`: one matrix for each of the six comparisons.
- `*.joblib`: five fitted text-vectorizer/classifier pipelines for local handoff.
  App changes and artifact selection/installation are outside this PR. The M2
  `models/sentiment_pipeline.joblib` file is never overwritten.
- `requirements-run.txt`: exact package versions; Python version is in metadata.

Artifacts include review text and split identities. Keep the entire output folder
local. Only load trusted joblib files. The final M3 team PDF, model/app integration,
UI screenshots and cloud deployment are separate work.

## Canonical report/demo version

The current report/demo release is **v0.2.0-rc2**. See [M3 results](M3_RESULTS.md)
and [machine-readable provenance](M3_RELEASE.json). Earlier November 15 results
and rc1 are historical and must not be mixed into the current tables/figures.
Use `--output-dir outputs/m3/release-rc2` for a fresh canonical rebuild. The runner
exports `model_hashes.json` as well as the existing records. Do not overwrite an
existing release: any intentional retraining needs a new version and updated
provenance/manifest. See the README for the hash-verified release demo command.
