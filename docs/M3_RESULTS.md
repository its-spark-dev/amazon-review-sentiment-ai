# M3 ML results — November 15, 2026

This is the ML handoff, not confirmation of M3 submission or app completion.
The original 37,023-row training and 9,255-row test CSVs were not modified.
The inner grouped seed-48 split contains 29,622 fit and 7,401 validation rows.
See [training protocol](M3_TRAINING.md) for executable commands and exact rules.

## Validation-selected settings

The primary criterion was macro-F1 across all three classes. Choices were written
to local selection.json before any final test evaluation. No M3 test scores were
used to adjust settings. All M3 vectorizers cap vocabulary at 10,000 features;
M2's uncapped 36,995-feature results are preserved as a different experiment.

| Configuration | Chosen setting | Validation Macro-F1 |
| --- | --- | ---: |
| majority | `Fixed baseline` | 0.2921 |
| bow_nb | `Fixed baseline` | 0.6063 |
| bow_lr | `{"C": 0.5, "class_weight": null, "validation_macro_f1": 0.6014775980751442}` | 0.6015 |
| tfidf_nb | `Fixed baseline` | 0.3480 |
| tfidf_lr | `{"C": 0.5, "class_weight": "balanced", "validation_macro_f1": 0.6166239048051888}` | 0.6166 |
| tfidf_mlp | `{"epochs": 11, "validation_macro_f1": 0.6097517546887548}` | 0.6098 |

The **validation-selected recommendation is TF-IDF Logistic Regression**, with
C=0.5 and balanced class weights. The MLP uses epoch 11 from the fixed 1–15 epoch
budget. The architecture is 10,000 sparse TF-IDF inputs -> 32 ReLU hidden units
-> 3 softmax probabilities. Adam uses learning rate 0.001, alpha 0.001, batch
size 256 and random_state=48. Final models were refitted on all outer training
rows with frozen settings. The MLP is a feedforward network, not a sequence model.

## Shared held-out comparison

| Model | Accuracy | Macro precision | Macro recall | Macro-F1 | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| majority | 0.7797 | 0.2599 | 0.3333 | 0.2921 | 0.009 |
| bow_nb | 0.8165 | 0.6092 | 0.6245 | 0.6163 | 0.703 |
| tfidf_nb | 0.7954 | 0.5461 | 0.3721 | 0.3639 | 0.682 |
| bow_lr | 0.8354 | 0.6297 | 0.5822 | 0.6008 | 5.498 |
| tfidf_lr | 0.7695 | 0.5958 | 0.6835 | 0.6161 | 1.290 |
| tfidf_mlp | 0.8443 | 0.6511 | 0.5872 | 0.6089 | 2.730 |

Precision/recall/F1 are macro averages with undefined classes zero-filled. The
reported seconds include training feature extraction plus classifier fitting,
using one numerical-library thread, and exclude selection, test prediction and
export. These are single-run timings on the local arm64 environment, not repeated
benchmarks. Selection including validation/loss logging took 25.786 seconds.

MLP has the highest measured test accuracy (84.43%), while capped BoW NB has the
highest test Macro-F1 (0.6163), narrowly above TF-IDF LR (0.6161). That numerical
difference is not evidence of statistical superiority and does not replace the
validation-based recommendation. The MLP does not outperform all classical
baselines on the primary metric. Balanced TF-IDF LR trades Positive precision/
recall balance for better minority recognition; its lower accuracy should be
interpreted together with per-class scores.

## Per-class test results

| Model | Class | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| majority | Negative | 0.0000 | 0.0000 | 0.0000 |
| majority | Neutral | 0.0000 | 0.0000 | 0.0000 |
| majority | Positive | 0.7797 | 1.0000 | 0.8762 |
| bow_nb | Negative | 0.6221 | 0.6506 | 0.6361 |
| bow_nb | Neutral | 0.2915 | 0.3277 | 0.3085 |
| bow_nb | Positive | 0.9140 | 0.8951 | 0.9044 |
| tfidf_nb | Negative | 0.8441 | 0.1180 | 0.2070 |
| tfidf_nb | Neutral | 0.0000 | 0.0000 | 0.0000 |
| tfidf_nb | Positive | 0.7944 | 0.9983 | 0.8847 |
| bow_lr | Negative | 0.6729 | 0.5965 | 0.6324 |
| bow_lr | Neutral | 0.3253 | 0.2090 | 0.2545 |
| bow_lr | Positive | 0.8911 | 0.9410 | 0.9153 |
| tfidf_lr | Negative | 0.5913 | 0.7010 | 0.6415 |
| tfidf_lr | Neutral | 0.2400 | 0.5452 | 0.3333 |
| tfidf_lr | Positive | 0.9562 | 0.8042 | 0.8736 |
| tfidf_mlp | Negative | 0.6905 | 0.6168 | 0.6516 |
| tfidf_mlp | Neutral | 0.3720 | 0.1949 | 0.2558 |
| tfidf_mlp | Positive | 0.8908 | 0.9500 | 0.9195 |

## MLP loss and limitations

The local `mlp_loss.png` shows unregularized cross-entropy after every epoch on
inner-fit and validation rows. Training loss falls from 0.5875 to 0.1082, while
validation loss reaches its minimum around epoch 6 (0.4025) and rises to 0.5108
by epoch 15. This divergence is consistent with overfitting. Epoch 11 is selected
because validation Macro-F1 (0.6098), not validation loss, was the predeclared
criterion. At that epoch the train/validation losses are 0.1579/0.4503.

The MLP can model nonlinear feature combinations, which the linear LR and the
NB conditional-independence formulation do not represent in the same way. This
is an architectural advantage, not a demonstrated universal performance gain.
Its limitations include more parameters/epoch choices, overfitting, less direct
feature interpretation, and lack of word-order information. The MLP's Neutral
recall is only 19.49%, compared with 54.52% for balanced TF-IDF LR. Loss curves
are diagnostics from inner training, not curves from the final full-train refit.

Future experiments must use inner validation; this shared test was already
inspected in M2 and is not an unseen prospective evaluation. Small single-split
score differences and single-seed neural-network results require caution. No
causal claim about why a model wins is established by these comparisons alone.

## Reproducibility and app handoff

Input SHA-256 values match the M2 run:

- train: `593f210cf56e2673fc3707473acccb1ca2673e5ca5d15f6817b68eb33f3f222f`
- test: `2c01b2781917986afaa34b6cf5502a0cdca642ea2fe7504ac1da985ca6c662f3`

Measured environment: python 3.11.14, scikit-learn 1.9.1, numpy 2.4.6, scipy 1.17.1, pandas 3.1.0, joblib 1.6.0, matplotlib 3.11.2, threadpoolctl 3.8.0.

`outputs/m3/comparison/` holds the exact selection IDs/candidate scores, epoch
history, metrics and predictions, six confusion matrices, timing comparison,
pinned run package versions, and five fitted pipeline artifacts. These remain
ignored/local and include raw review text. The final classifiers use their own
full-training vocabulary and expose predict/predict_proba for local handoff.
No app source or existing M2 app artifact is changed. Text-cleaning parity and
multi-model UI integration still require coordination with the App Engineer.

The team must still integrate the app, capture two screenshots, assemble the
4–6 page M3 PDF, and confirm the revised timeline and individual progress logs.
AI assistance was used for implementation and these technical notes; final
analysis and submission writing must be checked and owned by the team.
