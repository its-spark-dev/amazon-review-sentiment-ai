# M3 report and demonstration — canonical release v0.2.0-rc2

Canonical results updated November 23, 2026; main deployment verification updated
November 24. This replaces the November 15 M3 experimental results
for report/demo parity. It is not confirmation of final PDF submission. Hosted
main and staging validation is recorded at the end of this document.
All current tables, timings, loss records and five model artifacts below belong
to ONE deterministic re-execution. M2 results remain unchanged in their own doc.
Machine-readable source: [M3_RELEASE.json](M3_RELEASE.json).

## Why the older report and rc1 differed

Forensic comparison found 9,783 shared vocabulary tokens and 217 exclusive tokens
on each side of the 10,000-feature cutoff. All exclusive tokens occur exactly
seven times in the same outer training corpus. Shared NB feature-count columns
match exactly, and class counts also match. This identifies tied-frequency feature
selection as the first demonstrated divergence, not a changed label mapping or
proof of a changed sample. sklearn's default max_features ranking uses an unstable
argsort for equal counts; the exact machine/build conditions of the rc1 run were
not recorded, so we do not claim to reconstruct every cause of its final weights.

The new fitter explicitly ranks training tokens by descending count, then lexical
order, retaining a fixed vocabulary in standard sklearn estimators. Selection
fits this rule only on inner training. Final fitting repeats it on full outer
training. We reran validation selection and final evaluation without changing the
candidate grid, architecture, data split or metric to improve observed test scores.
The source correction is motivated by reproducibility, not test-based tuning.
Historical rc1 is retained, but is not the current report/demo release.

## Frozen protocol

50,000 seed-48 sampled reviews -> 46,278 cleaned rows -> 37,023 outer train /
9,255 outer test. Training-only grouped seed-48 validation: 29,622 fit / 7,401
validation rows. No identical cleaned Text or Id crosses partitions. The outer
test hashes are unchanged from M2. The old test has already been inspected;
this rerun is not a new unseen prospective evaluation.

All models use at most 10,000 lowercase word unigrams, no stop-word list. BoW
uses counts and TF-IDF uses smoothed IDF/L2. LR searches C={0.5,2.0} and weights
{None,balanced}. NB alpha=1.0 is fixed. Vocabulary/IDF selection never fits on
validation/test. Recommendation and settings are frozen by validation macro-F1
before final test evaluation, followed by full outer-training refits.

| Model | Validation Macro-F1 | Selected setting |
| --- | ---: | --- |
| majority | 0.2921 | `{}` |
| bow_nb | 0.6070 | `{}` |
| bow_lr | 0.6022 | `{"C": 0.5, "class_weight": null}` |
| tfidf_nb | 0.3480 | `{}` |
| tfidf_lr | 0.6170 | `{"C": 0.5, "class_weight": "balanced"}` |
| tfidf_mlp | 0.6057 | `{"epochs": 13}` |

The validation-selected recommendation is **TF-IDF LR**, C=0.5, balanced weights.
MLP architecture: 10,000 TF-IDF inputs -> 32 ReLU hidden units -> three softmax
probabilities. Adam: learning_rate_init=0.001, alpha=0.001, batch_size=256,
shuffle=True, seed 48. The small fixed hidden layer bounds local CPU/memory cost;
its width was not searched. MLP epoch selection uses Macro-F1, not loss.

## Held-out comparison from the canonical run

| Model | Accuracy | Macro precision | Macro recall | Macro-F1 | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| majority | 0.7797 | 0.2599 | 0.3333 | 0.2921 | 0.009 |
| bow_nb | 0.8167 | 0.6103 | 0.6260 | 0.6177 | 1.316 |
| tfidf_nb | 0.7956 | 0.5468 | 0.3726 | 0.3647 | 1.368 |
| bow_lr | 0.8351 | 0.6278 | 0.5808 | 0.5991 | 5.901 |
| tfidf_lr | 0.7693 | 0.5957 | 0.6838 | 0.6162 | 2.106 |
| tfidf_mlp | 0.8401 | 0.6374 | 0.5828 | 0.6021 | 3.855 |

Seconds measure one-thread training feature extraction (including deterministic
vocabulary selection) plus classifier fit. Selection, loss logging, test inference,
serialization and plotting are excluded. Timings are one local run, not repeated
benchmarks. MLP has the highest test accuracy, while BoW NB has the highest test
Macro-F1; neither observation overrides the validation-based recommendation.
M2 used an uncapped 36,995-feature budget, so changes cannot be attributed solely
to classifier choice. This M3 comparison uses a shared 10,000-feature budget.

| Model | Class | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| majority | Negative | 0.0000 | 0.0000 | 0.0000 |
| majority | Neutral | 0.0000 | 0.0000 | 0.0000 |
| majority | Positive | 0.7797 | 1.0000 | 0.8762 |
| bow_nb | Negative | 0.6224 | 0.6514 | 0.6366 |
| bow_nb | Neutral | 0.2945 | 0.3319 | 0.3121 |
| bow_nb | Positive | 0.9141 | 0.8948 | 0.9043 |
| tfidf_nb | Negative | 0.8457 | 0.1195 | 0.2093 |
| tfidf_nb | Neutral | 0.0000 | 0.0000 | 0.0000 |
| tfidf_nb | Positive | 0.7945 | 0.9983 | 0.8848 |
| bow_lr | Negative | 0.6717 | 0.5965 | 0.6319 |
| bow_lr | Neutral | 0.3208 | 0.2048 | 0.2500 |
| bow_lr | Positive | 0.8910 | 0.9410 | 0.9153 |
| tfidf_lr | Negative | 0.5907 | 0.7025 | 0.6417 |
| tfidf_lr | Neutral | 0.2405 | 0.5452 | 0.3338 |
| tfidf_lr | Positive | 0.9558 | 0.8036 | 0.8731 |
| tfidf_mlp | Negative | 0.6784 | 0.6101 | 0.6424 |
| tfidf_mlp | Neutral | 0.3426 | 0.1921 | 0.2462 |
| tfidf_mlp | Positive | 0.8911 | 0.9461 | 0.9178 |

Confusion matrices: rows=true, columns=predicted, order Negative/Neutral/Positive.

**majority**

```text
[0, 0, 1331]
[0, 0, 708]
[0, 0, 7216]
```

**bow_nb**

```text
[867, 171, 293]
[159, 235, 314]
[367, 392, 6457]
```

**tfidf_nb**

```text
[159, 0, 1172]
[17, 0, 691]
[12, 0, 7204]
```

**bow_lr**

```text
[794, 108, 429]
[161, 145, 402]
[227, 199, 6790]
```

**tfidf_lr**

```text
[935, 274, 122]
[176, 386, 146]
[472, 945, 5799]
```

**tfidf_mlp**

```text
[812, 97, 422]
[160, 136, 412]
[225, 164, 6827]
```

## Neural-network loss analysis

Selected epoch: **13**, validation Macro-F1 0.6057.
Training loss falls from 0.5870 to 0.1113;
validation loss reaches its minimum at epoch 6 (0.4041)
and rises to 0.5117 at epoch 15. This is consistent with
overfitting. At the selected epoch, train/validation loss is
0.1343/0.4809. Selection intentionally uses macro-F1,
not minimum cross-entropy. The diagnostic curves are from inner training,
not from the full-training refit. Loss history is in the provenance JSON and
`mlp_loss.png` in the release attachments, so the figure is tied to the same run.

The MLP can represent nonlinear feature combinations, but the measured result does
not establish superiority over all classical models. Limitations include overfit,
more parameters, lower interpretability and no word-order modeling. Class weighting
in LR improves minority recall at a cost to accuracy; discuss both macro and
per-class metrics. Single split/seed results are not statistical superiority claims.

## Exact report/demo handoff

Use only the model hashes in `models/release_manifest.json` under **v0.2.0-rc2**.
`docs/M3_RELEASE.json` records all five model hashes, input hashes, source hashes,
settings, full loss history, metrics, timing and runtime versions. Raw reviews and
split IDs are excluded. Model binaries, report-only loss PNG and provenance are
published as a new prerelease, without replacing rc1.

Run/rebuild instructions: [M3 training](M3_TRAINING.md). Downloads use the immutable
release assets plus hash verification; a rerun may have different serialization
bytes/timings, so cut a new release rather than replacing approved artifacts.

```bash
MODELS_VERSION_TAG=v0.2.0-rc2 python -m streamlit run app.py
```

Release mode lists ONLY the five verified report models, ignoring unrelated old
local joblib files. A legacy release tag is rejected with instructions to update
the setting. No-tag local mode still supports development model discovery.
Both local and release inference now apply the SAME HTML/entity/whitespace/lowercase
cleaner as data preparation, then reject inputs empty after cleaning. Models expect
cleaned Text; Summary and rating are not features.

## Hosted main status

**Latest verification: November 24, 2026 (EST)** after
[PR #23](https://github.com/its-spark-dev/amazon-review-sentiment-ai/pull/23)
merged into main at `9a29272`. The app was fully rebooted: automatic code refresh
had shown the new UI while retaining the previous zero-feature inference
behavior; the reboot resolved that observed mismatch. No model tag, weights,
training settings or reported test results changed.

| Hosted check | November 24 result |
| --- | --- |
| New-session default | `tfidf_lr`, with the rc2 recommendation visible |
| Probability display | Predicted-class probability, one decimal, explicit uncalibrated-estimate explanation |
| Positive/negative probes | 10 passed: two synthetic reviews per model; labels and displayed probabilities match canonical local rc2 |
| Zero-feature input | 15 passed: emoji-only, Korean and unseen-token probes for each of five models; no label/probability displayed |
| Recovery after rejection | Five passed: a recognized review produces a result again for each model |
| Existing validation | Empty and HTML/whitespace-only input rejection passed |

The check covered model switching across all five pipelines. The vocabulary
guard does not establish language detection, out-of-domain detection or calibrated
uncertainty. These are serving checks, not new held-out performance estimates.
The underlying artifacts remain `v0.2.0-rc2`, served on Python 3.11.18.

Current public screenshots: [TF-IDF LR prediction](assets/demo.jpg) and
[zero-feature input guard](assets/input-guard.jpg). The prediction example shows
98.6% Positive probability, with the explicit reliability limitation. The detailed
synthetic check log remains local at `outputs/m3/main-serving-fix/validation.json`;
it is not required to run the app. Main's post-merge CI passed all six jobs.

### Initial main deployment (November 23)

The [main app](https://amazon-sentiment-ai.streamlit.app/) was deployed from
`main` after [PR #19](https://github.com/its-spark-dev/amazon-review-sentiment-ai/pull/19)
passed all six CI checks and merged at `216720c`. Its runtime is Python 3.11.18,
with the pinned root `requirements.txt` and `MODELS_VERSION_TAG="v0.2.0-rc2"`.
The separate staging deployment was retained.

After a full reboot, 14 hosted checks passed on November 23, 2026 (EST;
verification log timestamp: November 24, 00:22 UTC). Each of the five models
produced the expected positive/negative labels and displayed probabilities
matching staging and the canonical local rc2 models. HTML/plain-text parity,
markup-only rejection, and empty-input rejection also passed. A screenshot of
the main app's TF-IDF LR positive prediction and the verification log were saved
locally at that time; the current public captures above supersede that screenshot.
These synthetic checks validate deployment and
inference behavior, not new held-out performance results or retraining.

## Hosted staging status

The [staging app](https://amazon-sentiment-ai-preview.streamlit.app/) was verified
on November 23, 2026 after setting `MODELS_VERSION_TAG="v0.2.0-rc2"` and fully
rebooting the Streamlit Community Cloud app. It serves the `staging` branch.
All five models passed positive/negative synthetic checks with displayed
probabilities matching the canonical local models. HTML/plain-text parity,
markup-only rejection and empty-input rejection also passed: 14 hosted checks
in total. These are app checks, not additional performance measurements.

The cloud runtime is Python 3.14; this report's training environment is Python
3.11 as recorded in the provenance JSON. The hosted checks do not establish
equivalent training behavior across Python versions. Two screenshots of the live
app were captured locally: TF-IDF LR predicting Positive and TF-IDF MLP predicting
Negative. Screenshots and validation logs are not committed.

The team must still include those screenshots in the 4–6 page PDF and confirm
individual progress logs and submission. Main deployment verification is complete
as recorded above; final PDF submission is not confirmed. AI assisted
implementation and these draft notes; final
interpretation and report writing should be checked and owned by the team.
