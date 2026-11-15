# M2 classical baseline: measured results, November 15, 2026

This is a technical handoff for Group Psi's M2 report, not the final submission.
Numbers below come from the local real-data run, not the synthetic demonstration.
The final report authors should verify the interpretation and write their own
analysis. AI assistance was used for implementation and these draft technical notes.

## Protocol and reproducibility

- Source: Kaggle SNAP Amazon Fine Food Reviews; sample 50,000, seed 48.
- Use the merged preparation scripts unchanged: cleaning leaves 46,278 rows.
- Shared seed-48 split: 37,023 train / 9,255 test, grouped by cleaned Text and
  stratified over distinct texts using their first label. The ratio is approximate
  at row level because repeated texts move together. No Id or Text overlap.
- Train Negative/Neutral/Positive counts: 5,324 / 2,835 / 28,864.
- Test Negative/Neutral/Positive counts: 1,331 / 708 / 7,216.
- Input is cleaned Text only. Summary and Score are not features. Score defines
  the Negative (1–2), Neutral (3), and Positive (4–5) target.
- Fixed settings: lowercase word unigrams, no stop-word removal, alpha=1.0.
  TF-IDF uses smoothed IDF and L2 normalization. Vocabulary/IDF fit training only.
- No tuning, validation search, or model selection against test results. Both
  planned representations are reported; no extra models were trained.
- The optional app artifact is the predeclared BoW NB model, not an automatically
  selected winner. It contains only the vectorizer/classifier, not the data cleaner.

From the activated Python 3.11 environment in the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/make_group_sample.py
python scripts/clean_reviews.py
python scripts/split_reviews.py
python scripts/run_m2_baseline.py --model-output models/sentiment_pipeline.joblib
```

The first command requires internet access; sampling requires local `data/Reviews.csv`.
Do not rerun preparation unnecessarily after agreeing on the shared split. The
runner validates input columns, IDs, labels, Score consistency when available,
and cross-partition ID/Text overlap without modifying data. It refuses existing
nonempty output directories and existing model output paths. Use `--output-dir`
for a fresh local directory or omit `--model-output` to evaluate without saving.

Local output: `outputs/m2/real-baseline/` (ignored by Git), including JSON, CSV,
confusion/feature PNGs, and `requirements-run.txt`. Full review text stays local.
`analysis.json` records input SHA-256 hashes and exact environment versions.

Train SHA-256: `593f210cf56e2673fc3707473acccb1ca2673e5ca5d15f6817b68eb33f3f222f`

Test SHA-256: `2c01b2781917986afaa34b6cf5502a0cdca642ea2fe7504ac1da985ca6c662f3`

Executed environment: Python 3.11.14; scikit-learn 1.9.1; numpy 2.4.6;
scipy 1.17.1; pandas 3.1.0; joblib 1.6.0; matplotlib 3.11.2; Streamlit 1.66.1.
Use the local pinned run requirements to reproduce the measured environment.

## Held-out comparison

Precision/recall/F1 below are macro averages over all three classes, with undefined
metrics zero-filled. Accuracy alone hides the majority model's minority failures.

| Model | Accuracy | Macro precision | Macro recall | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Training-majority (Positive) | 77.97% | 0.2599 | 0.3333 | 0.2921 |
| BoW + Multinomial NB | 83.39% | 0.6326 | 0.5364 | 0.5524 |
| TF-IDF + Multinomial NB | 77.97% | 0.4266 | 0.3337 | 0.2931 |

BoW improves accuracy by 5.42 percentage points and Macro-F1 by 0.2603 over the
majority rule on this split. These are baseline results, not a final model ranking.

| Model / class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| BoW / Negative | 0.6580 | 0.5695 | 0.6106 |
| BoW / Neutral | 0.3711 | 0.0833 | 0.1361 |
| BoW / Positive | 0.8687 | 0.9563 | 0.9104 |
| TF-IDF / Negative | 0.5000 | 0.0015 | 0.0030 |
| TF-IDF / Neutral | 0.0000 | 0.0000 | 0.0000 |
| TF-IDF / Positive | 0.7798 | 0.9997 | 0.8762 |

Confusion matrices: rows = true labels, columns = predictions;
order = **Negative, Neutral, Positive**.

```text
Majority              BoW NB                TF-IDF NB
[   0, 0, 1331]       [ 758, 44,  529]       [   2, 0, 1329]
[   0, 0,  708]       [ 135, 59,  514]       [   0, 0,  708]
[   0, 0, 7216]       [ 259, 56, 6901]       [   2, 0, 7214]
```

BoW correctly identifies only 59 of 708 Neutral examples. TF-IDF predicts no
Neutral cases and only four Negative cases in total. A hypothesis is that L2
normalization, smoothing, and the strong Positive prior leave too little evidence
for minority classes under these defaults; this was not established by an ablation.
Investigate alternatives using training-only validation in M3, not repeated test
optimization. Planned Logistic Regression and MLP remain unimplemented.

## Feature engineering

Both representations learn **36,995 features** because they share the tokenizer
and training corpus. BoW counts occurrences; TF-IDF weights document rarity and
normalizes each review. Existing feature plots visualize model associations.

Rank by log P(token|Positive) minus log P(token|Negative), reversed for the
Negative direction. Neutral is excluded from this pairwise explanation but
remains in classification. Entries are in descending contrast order.

| Representation | Positive top 10 | Negative top 10 |
| --- | --- | --- |
| BoW | cravings, hooked, calm, pleasantly, heavenly, soothing, influenster, mccann, mornings, winner | returnable, emails, monophosphate, nastiest, nutricity, mee, monoglycerides, salvage, titanium, disgustingly |
| TF-IDF | hooked, perfect, delicious, refreshing, pleasantly, wonderful, excellent, easy, smooth, beat | returnable, worst, refund, threw, emails, refunded, inedible, disappointing, disgusting, undrinkable |

Some BoW terms are product/ingredient names rather than general sentiment words.
The ranking has no minimum document-frequency filter, so interpretability and
stability need care. Scores and vocabulary size are in local `analysis.json`;
`bow_nb_features.png` and `tfidf_nb_features.png` provide the M2 visualization.

## Ten inspected BoW misclassifications

Selection: the **first ten BoW errors in preserved test-row order**, not examples
selected to maximize a narrative. Full texts remain in the local errors CSV.
The following are paraphrases and possible explanations, not causal attribution.

| Review Id | True → predicted | Observation and possible failure |
| --- | --- | --- |
| 453881 | Negative → Positive | Complaint concerns a calorie-pack/serving-size mismatch and unsuccessful replacement. Product and numeric details dominate; dissatisfaction is partly relational. |
| 387606 | Negative → Positive | Clearly disappointed with weak coffee despite trying more pods. An explicit-negative counterexample: failure is not restricted to subtle sarcasm or mixed reviews. |
| 338492 | Neutral → Positive | Mostly favorable pump-function wording despite a three-star label. Rating-derived Neutral can differ from surface tone. |
| 534562 | Positive → Neutral | Four-star coffee review mixes freshness/aroma praise with criticism of strength; the overall rating is Positive. |
| 16769 | Negative → Positive | Initial taste praise is followed by sweetness/calorie objections and a decision not to repurchase. Contrast and final intent matter. |
| 289204 | Neutral → Positive | General enthusiasm for pickles and shipping coexists with product disappointment and price objections. Mixed target sentiment is flattened into unigram evidence. |
| 206523 | Neutral → Positive | Prompt delivery and packaging praise precede stale-taste concerns and refusal to reorder. Delivery and product quality express different sentiments. |
| 186200 | Neutral → Positive | Sleep benefit is weighed against grogginess and unpleasant effects. A long mixed review with a three-star outcome. |
| 168402 | Neutral → Negative | Coffee quality is praised but damaged pods are unusable. The rating balances positive product taste and negative delivery condition. |
| 100392 | Positive → Negative | Harsh wording describes undesirable hay, followed by praise and repurchase intent for this product. Context/reference and sentiment scope are important. |

Actionable hypotheses for M3: negation/contrast bigrams; stronger handling of
minority/Neutral decisions; compare the planned discriminative model; inspect
rating/text disagreement. Evaluate those choices within training-only validation.
These ten cases support qualitative observations, not population error rates.

## Handoff checklist

Measured tables, features, charts, and ten inspected cases are available for Yash
and Zachary. The team still needs to assemble the 3–5 page M2 PDF, update the
team timeline, and write each member's progress paragraph. A factual ML progress
record is: implemented CSV handoff validation and reproducible execution, ran
fixed-setting BoW/TF-IDF NB and majority comparisons on the shared split, exported
features/errors/figures, inspected the first ten BoW errors, and saved a local
BoW pipeline for the app. No deployment or M3 models are claimed complete.
