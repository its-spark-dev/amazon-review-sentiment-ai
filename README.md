# Amazon Review Sentiment AI

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat)
![M2 Complete](https://img.shields.io/badge/M2-Complete-2E7D32?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat)

Three-class sentiment analysis of Amazon Fine Food Reviews, with reproducible
classical ML baselines, evaluation tools, and a local Streamlit inference UI.

## Project Overview

Built by **Group Psi** for **CSE 3521 — Survey of Artificial Intelligence I**,
Autumn 2026, at **The Ohio State University**. Instructor: **Dr. Thomas Torku**.

The task maps ratings **1–2 to Negative, 3 to Neutral, and 4–5 to Positive**.
Models use review text as input; ratings define the target and are not features.
The planned final system will compare at least two feature representations,
two classical models, and a deep learning model through a deployed Streamlit app.

## Current Status

As of **November 15, 2026**:

| Milestone | Deadline | Status |
| --- | --- | --- |
| M1 — Project Charter & Scope | Nov 10, 2026 | Submitted |
| M2 — Initial Solution | Nov 15, 2026 | Technical implementation complete; submission not confirmed |
| M3 — Refined Solution & App Prototype | Nov 22, 2026 | Planned |
| M4 — Final Report | Nov 29, 2026 | Planned |

Milestone deadlines are **11:59 PM EST**. M1 was extended from November 8 to
November 10. The **M2 Complete** badge denotes technical completion, not confirmed
submission or completion of the entire project. Final M2 PDF assembly, the updated
team timeline, and individual progress logs remain team submission tasks.

## Key Results — Milestone 2

A **50,000-review sample with seed 48** retained **46,278 reviews after cleaning**.
The shared split contains **37,023 training / 9,255 test reviews**, approximately
80/20. Identical cleaned texts stay in one partition to prevent text overlap;
stratification is performed over distinct texts using their first labels.

Both Bag of Words (BoW) and TF-IDF learn **36,995 features from training text
only**. Fixed-setting Multinomial Naive Bayes models are compared with a baseline
that always predicts the most common training label, Positive.

| Model | Test Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Majority-class baseline | 77.97% | 0.2921 |
| **BoW + Multinomial Naive Bayes** | **83.39%** | **0.5524** |
| TF-IDF + Multinomial Naive Bayes | 77.97% | 0.2931 |

BoW + Naive Bayes is the **best measured M2 baseline**, not the final model.
Neutral classification remains challenging: its BoW recall is only **8.33%**.
TF-IDF + Naive Bayes predicts almost every test review as Positive under the
current defaults. No hyperparameter tuning or model selection used test results.

See [M2 baseline results](docs/M2_BASELINE_RESULTS.md) for the complete evaluation,
feature rankings, confusion matrices, ten inspected errors, and reproducibility
record. These are measured real-data results; synthetic regression tests serve
a separate purpose and are not performance evidence.

## Implementation

- Reproducible sampling, text cleaning, and the shared grouped train/test split.
- Three-class label mapping, BoW/TF-IDF extraction, and Naive Bayes pipelines.
- Majority-class comparison, macro/per-class metrics, and confusion matrices.
- Local JSON/CSV exports, feature visualizations, and traceable error lists.
- Streamlit review input, sample reviews, predicted labels, and class probabilities.
- Automated synthetic regression tests for data preparation, ML tools, and the UI.

**No dataset or trained model artifact is committed.** A fresh clone runs the UI
with a no-model notice and analysis disabled. Predictions require a compatible
locally saved pipeline at `models/sentiment_pipeline.joblib`, or the path set by
`SENTIMENT_MODEL_PATH`. Model additions, replacements, and removals are detected
on the next app rerun; unchanged files reuse the cached resource.

API and usage details: [ML baseline](docs/ML_BASELINE.md) ·
[M2 execution and analysis](docs/M2_ANALYSIS.md).

## Project Structure

```text
amazon-review-sentiment-ai/
├── README.md
├── app.py                    # Streamlit interface
├── inference.py              # Model loading and review prediction
├── ml_baseline.py            # Labels, vectorizers, NB pipelines, metrics
├── m2_analysis.py            # Comparison and analysis exports
├── requirements.txt          # Runtime dependencies
├── .streamlit/config.toml    # Theme and server settings
├── scripts/                  # Sampling, cleaning, splitting, real-data execution
├── data/                     # Dataset guide; local CSVs are ignored
├── docs/                     # Methods, usage, and measured M2 results
├── tests/                    # Synthetic regression tests
├── models/                   # Local model artifacts; ignored
├── notebooks/                # Reserved for notebooks
└── report/                   # Reserved for written deliverables
```

## Getting Started

Use **Python 3.11**. On macOS or Linux:

```bash
git clone https://github.com/its-spark-dev/amazon-review-sentiment-ai.git
cd amazon-review-sentiment-ai
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m streamlit run app.py
```

On Windows, create the environment with `py -3.11 -m venv .venv` and activate it
with `.venv\Scripts\Activate.ps1` in PowerShell, then run the same installation,
test, and app commands. Open Streamlit's printed local URL, normally
`http://localhost:8501`; stop the server with `Ctrl+C`.

Installation needs package access. Tests and the no-model UI do **not** require
the dataset, credentials, or an external inference service.

To work with real data, first obtain `data/Reviews.csv` and follow the
[data preparation guide](data/README.md). The sampling, cleaning, and splitting
scripts require the dataset and produce local CSVs. Once the shared train/test
files exist, run:

```bash
python scripts/run_m2_baseline.py --model-output models/sentiment_pipeline.joblib
```

The runner never resplits the supplied data. It writes local results under
`outputs/m2/real-baseline/` and optionally saves the predeclared BoW model for the
app. It refuses nonempty output directories and existing model artifacts; use a
fresh `--output-dir` and omit `--model-output` when repeating evaluation without
saving another model. See the [M2 reproduction instructions](docs/M2_BASELINE_RESULTS.md)
for settings, input hashes, and exact measured software versions. Dependency
ranges are in `requirements.txt`; each run also exports `requirements-run.txt`
with pinned package versions locally.

Keep datasets, generated outputs, and secrets out of Git. Only load joblib model
files from trusted team sources: loading them can execute code.

## Roadmap

- **M3 — Planned:** Logistic Regression, a feedforward neural network (MLP),
  training-only validation and further evaluation, full model comparisons, and
  refinement of the app prototype.
- **M4 — Planned:** Final report and presentation slides, Streamlit Community
  Cloud deployment, and final verification. The team timeline schedules an
  **8-minute presentation + 2-minute Q&A on December 3, 2026**.

Logistic Regression, MLP, and cloud deployment are not implemented. There is no
live deployment URL. The intended hosting target is
[Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app),
using `main`, `app.py`, Python 3.11, and the root `requirements.txt`. A deployed
instance will also need access to a compatible model. Configure future secrets
in the platform's secrets settings; never commit `.env` or
`.streamlit/secrets.toml`.

## Team and Acknowledgments

| Member | Confirmed primary role |
| --- | --- |
| Zachary Funk | Project Lead |
| Sang Park | ML Engineer |
| Jacob Myers | App Engineer |
| Yash Amin | Analyst / Writer |
| Abderrachid Sghir | Data Engineer |

Roles follow the submitted M1 charter. Sang implemented the ML execution and
analysis work; Jacob contributed the Streamlit UI and shared data preparation
scripts ([PR #7](https://github.com/its-spark-dev/amazon-review-sentiment-ai/pull/7),
[PR #8](https://github.com/its-spark-dev/amazon-review-sentiment-ai/pull/8)). The
Git history records individual contributions; use focused branches and pull
requests and run relevant tests before proposing changes.

Dataset: **Amazon Fine Food Reviews**, published by SNAP on
[Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).
AI assistance for implementation and draft technical notes is disclosed in the
[M2 results document](docs/M2_BASELINE_RESULTS.md); final analysis and submission
writing should be reviewed and owned by the team.
