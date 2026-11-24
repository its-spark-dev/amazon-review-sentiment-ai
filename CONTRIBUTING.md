# Contributing

This is Group Psi's CSE 3521 team project. Preserve contribution attribution and
the distinction between a course result, an experiment and the deployed demo.

## Set up and check

Use Python 3.11 and an activated virtual environment:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -m "not integration and not e2e"
python -m pytest -m integration
python -m playwright install chromium
python -m pytest -m e2e
ruff check .
```

All tests use synthetic data and temporary models. See the [README](README.md)
for running the release demo without a dataset.

## Make a focused change

1. Start from the latest target branch; use a focused branch and pull request.
2. Explain the behavior or document being changed, its evidence and validation.
3. Keep unrelated team components out of the diff. Confirm that public links,
   result tables, examples and attribution are accurate.
4. Run relevant checks and review CI before proposing a merge. Maintainers decide
   whether to merge; do not assume that an open PR is part of main.

## Data and model handling

- Do not commit raw/processed reviews, review IDs, split memberships, credentials,
  downloaded archives or model binaries. Generated output folders stay local.
- Public examples should be synthetic or clearly labeled paraphrases; reports
  should distinguish inspected examples from representative estimates.
- Load only trusted team artifacts. Keep secrets out of source, logs and screenshots.
- The rc2 report/demo artifacts are frozen. Intentional model or preprocessing
  changes require a separately reviewed experiment/release and updated provenance,
  rather than replacement of existing assets. The source-hash regression check
  deliberately detects changes to the canonical training/preparation code.
- Fit vocabulary/IDF and choose settings on training/inner validation only. The
  shared test has already been inspected; do not tune against its score.

No code license is declared yet. Contribution instructions do not assign a
license or override team ownership or the dataset's separate terms. Raise reuse
and licensing questions with the maintainers before assuming permission.
