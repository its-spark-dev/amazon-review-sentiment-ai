# Amazon Review Sentiment AI

Sentiment analysis of the Amazon Fine Food Reviews dataset for **The Ohio State University, CSE 3521, Autumn 2026 Flagship Group Project**. The project will compare feature representations and classical machine learning approaches with a deep or modern NLP approach, and present the work through Streamlit Community Cloud.

**Current status: scaffold plus a dataset-independent ML baseline module.** The Streamlit app displays a placeholder page. The [ML baseline guide](docs/ML_BASELINE.md) documents rating labels, BoW/TF-IDF features, a Multinomial Naive Bayes pipeline, evaluation, and synthetic tests. Real-data processing, training, evaluation, and application inference are not implemented. No dataset, trained model, or experimental results are included.

## Project requirements

The course requirements supplied for this initialization are the source of truth:

- Maintain a public GitHub repository.
- Analyze sentiment in Amazon Fine Food Reviews.
- Use at least **two feature representations**.
- Compare at least **two classical ML models**.
- Include at least **one deep or modern NLP model**.
- Deploy an application to **Streamlit Community Cloud**.
- Complete four milestones from **November 8 through November 29, 2026**.

Group Psi plans BoW and TF-IDF features, Multinomial Naive Bayes and Logistic Regression, and a feedforward neural network. Ratings 1–2 map to Negative, 3 to Neutral, and 4–5 to Positive. Only the reusable Naive Bayes baseline is implemented; see the [ML baseline guide](docs/ML_BASELINE.md) for scope and the pending Data Engineer handoff.

## Milestones

| Milestone | Date | Deliverables |
| --- | --- | --- |
| 1 | November 8, 2026 | To be confirmed against the official milestone instructions |
| 2 | November 15, 2026 | To be confirmed against the official milestone instructions |
| 3 | November 22, 2026 | To be confirmed against the official milestone instructions |
| 4 | November 29, 2026 | To be confirmed against the official milestone instructions |

The dates above follow the supplied four-week schedule. Add the official milestone deliverables when available; this scaffold does not define submission requirements.

## Repository layout

```text
amazon-review-sentiment-ai/
├── app.py              # Streamlit placeholder; no inference yet
├── requirements.txt    # Dependencies for the initial scaffold
├── data/
│   └── README.md       # Dataset source and local storage guidance
├── notebooks/          # Future exploration and experiments
├── models/             # Future local model artifacts (ignored by Git)
└── report/             # Project report and milestone documentation
```

Empty directories contain `.gitkeep` placeholders so Git preserves the structure. The root `ml_baseline.py` contains the reusable ML components; `tests/` contains synthetic unit tests and `docs/ML_BASELINE.md` describes the ML API and usage.

## Local setup

Use **Python 3.11** for the initial scaffold. From a terminal:

```bash
git clone https://github.com/its-spark-dev/amazon-review-sentiment-ai.git
cd amazon-review-sentiment-ai
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell. Open the local URL printed by Streamlit (normally `http://localhost:8501`). Stop the server with `Ctrl+C`.

The placeholder needs no dataset, model files, or secrets. See [data/README.md](data/README.md) for instructions for obtaining the dataset later. Add ML and notebook dependencies only when their implementation requires them.

## Streamlit Community Cloud

Deployment is a required later project step; no live deployment is claimed by this scaffold. Following the [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app), select this repository, the `main` branch, and `app.py` as the entrypoint. Use Python 3.11 and the root `requirements.txt` for the initial app.

If secrets are needed later, configure them in Community Cloud's secrets settings. Never commit `.env`, `.streamlit/secrets.toml`, or dataset service credentials.

## Team roster

Replace the placeholders once team membership and responsibilities are confirmed.

| Team member | GitHub username | Responsibilities |
| --- | --- | --- |
| TBD | TBD | TBD |

## Contributing

Keep changes focused and document setup changes here. Store exploration in `notebooks/`, written deliverables in `report/`, and local data in `data/`. Do not commit datasets, secrets, or generated model files. Use descriptive commits and coordinate overlapping work with the team; use pull requests for subsequent collaborative changes.
