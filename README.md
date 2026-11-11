# Amazon Review Sentiment AI

## Project Overview

Three-class sentiment analysis of Amazon Fine Food Reviews by **Group Psi** for
**CSE 3521 — Survey of Artificial Intelligence I**, Autumn 2026, at **The Ohio
State University**. Instructor: **Dr. Thomas Torku**.

The project will compare classical machine learning and a feedforward neural
network, with an interactive application on Streamlit Community Cloud. Course
requirements include a public GitHub repository, at least two feature
representations, at least two classical ML models, and at least one deep or
modern NLP model.

**Status as of November 11, 2026: Milestone 1 submitted; preparing M2 (November 15).**
The submitted M1 charter defines the team roles, problem statement, dataset
summary, technical approach, and timeline. The repository currently provides
the reusable ML baseline and a Streamlit placeholder; real-data model results
are still pending.

## Team

| Confirmed member | Primary role |
| --- | --- |
| Zachary Funk | Project Lead |
| Sang Park | ML Engineer |
| Jacob Myers | App Engineer |
| Yash Amin | Analyst / Writer |
| Abderrachid Sghir | Data Engineer |

Roles follow the submitted M1 charter. GitHub usernames are not listed unless
confirmed by the team.

## Dataset and Sentiment Labels

Source: [Amazon Fine Food Reviews on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).
The Group Psi sample uses **50,000 reviews** with **`random_state=48`**.
The submitted M1 dataset summary reports initial exploration of this sample:
approximately **14.4% Negative, 7.6% Neutral, and 78.0% Positive**, with **3,720
duplicate rows** identified under its user/profile/timestamp/text criterion.
These are reported exploratory findings, not model results or statistics
recomputed by this repository. No dataset or preparation script is committed.
See [data/README.md](data/README.md) for source and local storage instructions.

| Star rating | Sentiment label |
| --- | --- |
| 1–2 | Negative |
| 3 | Neutral |
| 4–5 | Positive |

Abderrachid owns collection, cleaning, and organization of the data. The M1
timeline assigns cleaning, two feature representations, and a shared **80/20
train/test split** to the Data Engineer for M2. The code already provides reusable
BoW/TF-IDF vectorizers; coordinate their use and the text/label handoff across
the data and ML work. Resolve duplicate handling before splitting, and fit
feature extraction only on training data. Any validation or tuning protocol
still needs team agreement; no real-data split is implemented here. Star ratings
define the target labels and are not inputs to the current text baseline.

## Technical Approach

| Area | Team plan |
| --- | --- |
| Features | Bag of Words (BoW) and TF-IDF |
| Classical ML | Multinomial Naive Bayes and Logistic Regression |
| Deep learning | Feedforward Neural Network (MLP) |
| Evaluation | Accuracy, precision, recall, Macro-F1, per-class metrics, and confusion matrices |
| Comparison protocol | A consistent shared split, class-imbalance analysis, and a majority-class baseline |

These plans describe the intended full project; current functionality is listed
below.

## Implementation Status

[PR #1](https://github.com/its-spark-dev/amazon-review-sentiment-ai/pull/1) added
these reusable components in `ml_baseline.py`:

- Validated integer rating-to-sentiment mapping for the three classes.
- Fresh, unfitted BoW and TF-IDF vectorizers.
- A vectorizer + Multinomial Naive Bayes pipeline that can fit and predict in memory.
- Evaluation returning accuracy, macro precision/recall/F1, per-class
  precision/recall/F1, and a confusion matrix ordered **Negative, Neutral, Positive**
  (rows: true labels; columns: predictions).
- Twelve synthetic-data unit tests covering mapping, features, fitting and
  prediction, leakage checks, and known evaluation outputs.

See the [ML baseline guide](docs/ML_BASELINE.md) for API details and an in-memory
example. Validation so far uses synthetic data only and establishes no Amazon
review performance. Real-data training, evaluation, and error analysis await
the cleaned-data handoff and coordinated split. M1 exploration is documented in
the submitted charter, but its preparation and EDA code are not in the repository.
The [M2 analysis tools](docs/M2_ANALYSIS.md) now provide a shared-split execution
flow for BoW/TF-IDF Naive Bayes and a training-majority baseline, metric exports,
feature analysis, confusion matrix plots, and traceable error lists. These tools
are validated only on synthetic data; no real-data comparison is complete.
Logistic Regression, MLP, model serialization, deployment, and live application
predictions are not implemented.
The Streamlit application remains a placeholder with no inference integration.

For M2, Sang will train the initial baseline after the data handoff; Jacob will
build the basic Streamlit interface and review input; Abderrachid will prepare
the shared data/features/split; Yash will document feature engineering, baseline
results, and error analysis; Zachary will coordinate blockers and the submission.
These are the submitted timeline's assignments, not claims of completed work.

## Project Milestones

The deadlines below reflect the course requirements and updated instructor
announcement. **M1 was officially extended from November 8 to November 10, 2026.**
All milestone submission times are **11:59 PM EST**; no time is specified here for
the live presentation.

| Milestone | Deadline | Deliverable | Status |
| --- | --- | --- | --- |
| M1 — Project Charter & Scope | Nov 10, 2026, 11:59 PM EST | Team roles, problem statement, dataset summary, technical plan, timeline, GitHub repository | Submitted |
| M2 — Initial Solution | Nov 15, 2026, 11:59 PM EST | Feature pipeline, initial classical model, baseline evaluation, error analysis | Next milestone |
| M3 — Refined Solution | Nov 22, 2026, 11:59 PM EST | All required models, comparison results, Streamlit prototype | Planned |
| M4 — Final Report | Nov 29, 2026, 11:59 PM EST | Final written report and presentation slides | Planned |
| Live Presentation | Dec 3, 2026 | 8-minute presentation and 2-minute Q&A | Planned |

The submitted team timeline targets model/app integration for M3 and deployment
and final verification for M4. Submission status is based on the team's update;
the submitted PDF is not currently tracked in this repository.

## Repository Structure

```text
amazon-review-sentiment-ai/
├── README.md
├── app.py                    # Streamlit placeholder
├── ml_baseline.py            # Reusable classical ML baseline
├── m2_analysis.py            # Dataset-independent execution and reporting
├── requirements.txt          # Runtime and plotting dependencies
├── data/README.md            # Dataset source and local storage guidance
├── docs/                     # ML baseline API and M2 analysis usage
├── tests/                    # Synthetic baseline and analysis tests
├── notebooks/                # Reserved for exploration and experiments
├── models/                   # Local generated artifacts; ignored by Git
└── report/                   # Reserved for written deliverables
```

Empty directories contain `.gitkeep` placeholders. Dataset files and generated
model artifacts are excluded by `.gitignore`.

## Local Setup and Testing

Use **Python 3.11**. From a terminal on macOS or Linux:

```bash
git clone https://github.com/its-spark-dev/amazon-review-sentiment-ai.git
cd amazon-review-sentiment-ai
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

On Windows, use `py -3.11 -m venv .venv` to create the environment and
`.venv\Scripts\Activate.ps1` in PowerShell to activate it, then run the same pip
and test commands. Tests need no dataset, credentials, or external services.

To launch the placeholder from the repository root with the environment active:

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). Stop
the server with `Ctrl+C`. The placeholder needs no dataset, model files, or secrets.

## Deployment Status

Streamlit Community Cloud deployment is planned; this repository provides no live
deployment URL or integrated model predictions. When ready, follow the
[official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)
and select this repository, branch `main`, entrypoint `app.py`, Python 3.11, and
the root `requirements.txt`. Deploying the current app would show only the
placeholder. Configure any future secrets in Community Cloud's secrets settings;
never commit `.env`, `.streamlit/secrets.toml`, or dataset service credentials.

## Contributing

Use focused branches and pull requests, run the synthetic tests for ML changes,
and coordinate changes across team roles. Keep exploration in `notebooks/` and
written deliverables in `report/`. Document contributions accurately and
distinguish planned methods from validated functionality. Do not commit datasets,
secrets, or generated model artifacts; coordinate shared data and split decisions
with the Data Engineer before real-data experiments.
