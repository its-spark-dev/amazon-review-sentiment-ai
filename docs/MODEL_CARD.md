# Model card: Amazon Review Sentiment AI rc2

## Purpose and scope

Release **v0.2.0-rc2** is a set of five scikit-learn text-classification pipelines
for Group Psi's CSE 3521 course project. It demonstrates reproducible comparison
and interactive inference on food-product review text. Intended inputs are English
reviews. It is not a transformer, language model, multilingual classifier, or
validated commercial review-analysis service.

The main [Streamlit demo](https://amazon-sentiment-ai.streamlit.app/) lets visitors
choose a model. The validation-recommended pipeline is **tfidf_lr**, which the
app selects initially when available. Valid user choices persist across reruns;
single-file local overrides remain supported. A demonstration input is not performance evidence.

## Data and target

The [SNAP Amazon Fine Foods source](https://snap.stanford.edu/data/web-FineFoods.html)
describes historical reviews spanning October 1999–October 2012. This project uses
the [Kaggle CSV distribution](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).
The sample is 50,000 reviews with seed 48; cleaning retains 46,278 rows.

| Rating | Label |
| --- | --- |
| 1–2 | Negative |
| 3 | Neutral |
| 4–5 | Positive |

The target is the reviewer's star rating mapped to sentiment, not an independent
human annotation of the text. Input features use Text only, not Score, Summary,
user identity or product identity. The [data guide](../data/README.md) records
HTML/entity/whitespace/lowercase normalization, duplicate handling and split rules.
The same cleaner is used by inference. The dataset and trained binaries are not
committed; models are distributed separately as team-produced release assets.
No code license is declared. Dataset usage terms require separate verification;
this card does not grant redistribution rights.

## Training and selection

- Outer train/test: 37,023 / 9,255 rows, grouped by identical cleaned Text, seed 48.
- Inner fit/validation: 29,622 / 7,401 rows from outer training only, same grouping.
- All learned vocabulary/IDF uses the applicable training partition only. Tied
  token frequencies at the 10,000-feature cutoff are ordered lexically.
- BoW uses counts; TF-IDF uses smoothed IDF and L2 normalization. Both retain
  stop words and use lowercase word unigrams with sklearn's default tokenizer.
- NB: alpha=1.0. LR: validation search over C={0.5,2.0}, weights={None,balanced}.
- MLP: 10,000 inputs, one 32-unit ReLU hidden layer, three softmax probabilities;
  Adam, alpha=0.001, initial learning rate=0.001, batch size=256, seed 48.
  Epoch 13 is selected from 1–15 by validation Macro-F1, not minimum loss.
- Settings are frozen before full-outer-training refits and test evaluation.

TF-IDF LR (C=0.5, balanced weights) has the highest validation Macro-F1, 0.6170.
The other selected LR setting is BoW C=0.5 with no class weights. MLP training
loss decreases while validation loss rises after its epoch-6 minimum, indicating
overfitting risk. [M3 results](M3_RESULTS.md) contains the full history and protocol.

## Evaluation

The fixed test contains 1,331 Negative, 708 Neutral and 7,216 Positive reviews.
Macro metrics average across all three classes. Confusion matrices have true
labels in rows and predictions in columns, in Negative/Neutral/Positive order.

| Configuration | Test accuracy | Macro-F1 | Neutral precision | Neutral recall |
| --- | ---: | ---: | ---: | ---: |
| Majority baseline | 77.97% | 0.2921 | 0.00% | 0.00% |
| BoW NB | 81.67% | 0.6177 | 29.45% | 33.19% |
| TF-IDF LR (validation recommendation) | 76.93% | 0.6162 | 24.05% | 54.52% |
| TF-IDF MLP | 84.01% | 0.6021 | 34.26% | 19.21% |

MLP's highest observed accuracy does not override validation-based selection.
All canonical scores, per-class metrics, model hashes, input/source hashes,
runtime versions and timings are in [M3_RELEASE.json](M3_RELEASE.json).
Synthetic software tests and hosted smoke checks are not model-quality measures.

## Limits of the evidence

The test was inspected in M2 and reused for M3. Its scores describe this shared
split rather than a new unseen prospective evaluation. The rc2 rerun corrected
feature-cutoff reproducibility with the same candidate grid and split, not a
search for better test performance. There is one split/seed and no uncertainty
interval or significance claim. M2 used a different, 36,995-feature budget;
cross-milestone changes cannot be attributed solely to classifier choice.

Exact-text grouping does not group users, products, time, or near-duplicate text.
On the canonical test, 1,650/9,255 rows share a UserId with training and
6,289/9,255 share a ProductId. These IDs are not model inputs, but performance
does not establish cold-start user/product or temporal generalization. The audit
does not change the course's frozen split or claim that these counts prove leakage
for the stated shared-split task.

Neutral is challenging and the three models trade minority recall against
majority accuracy. Unigrams omit explicit word order; the MLP does not restore
sequence input. Class weighting, architecture and feature choice differ across
models, so the comparison is not an isolated causal experiment on architecture.
No fairness, modern out-of-domain, language-specific or adversarial evaluation
has been performed.

The app labels predict_proba output as “Predicted-class probability”, with one
decimal place and an explicit uncalibrated-estimate explanation. No probability
calibration has been validated; a high value can still accompany a wrong label.
After shared text cleanup, BoW/TF-IDF pipelines with **zero recognized features**
produce an insufficient-vocabulary message instead of a label or probabilities.
This is not language or out-of-domain detection: one recognized token is enough
to pass the guard, and partially recognized inputs can still be unreliable.
This serving check changes neither fitted model parameters nor probability values
for supported inputs.

## Artifact and runtime contract

Use `MODELS_VERSION_TAG=v0.2.0-rc2` for the approved demo. Release mode verifies
each artifact against [release_manifest.json](../models/release_manifest.json),
then exposes only those five files. A trusted manifest and origin matter:
hash equality binds bytes, not the truth of a model's predictions. Joblib files
can execute code when loaded; do not load arbitrary external artifacts.

Training used Python 3.11.14. The main deployment was checked on Python 3.11.18;
staging was checked on Python 3.14. Direct runtime dependency versions are pinned
in requirements.txt, but it is not a complete transitive environment lock.
Hosted checks establish only the observed serving behavior on the recorded date,
not identical training across Python versions. Follow [training instructions](M3_TRAINING.md)
for the current package versions and reproduction outputs. Dataset preparation is
required for training; the release demo and synthetic tests need no dataset.

## Ownership and status

Group Psi maintains the course deliverables; team roles and contribution attribution
are in the [README](../README.md#team-attribution-and-contributing). AI assistance
is disclosed there and in M2 documentation. M1/M2 are submitted; M3 PDF updating
and M4's final report/presentation remain separate from technical completion.
See [portfolio readiness](PORTFOLIO_READINESS.md) for proposed improvements and
acceptance criteria. This card adds documentation and does not change rc2.
