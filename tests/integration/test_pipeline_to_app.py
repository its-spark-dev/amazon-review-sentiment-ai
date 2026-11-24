"""Raw CSV -> clean -> split -> M2 runner -> saved artifact -> inference -> Streamlit UI."""

import json
from pathlib import Path

import pandas as pd
import pytest

from inference import load_model, predict_review
from ml_baseline import CLASS_LABELS
import synthetic

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pipeline_outputs(tmp_path_factory, run_script):
    tmp_path = tmp_path_factory.mktemp("pipeline")
    raw = synthetic.write_raw_reviews(tmp_path / "Reviews.csv")
    cleaned, train, test = tmp_path / "cleaned.csv", tmp_path / "train.csv", tmp_path / "test.csv"
    model, results = tmp_path / "models" / "sentiment_pipeline.joblib", tmp_path / "results"
    model.parent.mkdir()
    run_script("scripts/clean_reviews.py", "--src", str(raw), "--dst", str(cleaned))
    run_script("scripts/split_reviews.py", "--src", str(cleaned), "--train", str(train), "--test", str(test))
    run_script("scripts/run_m2_baseline.py", "--train", str(train), "--test", str(test),
               "--output-dir", str(results), "--model-output", str(model), "--data-description", "SYNTHETIC")
    return {"raw": raw, "cleaned": cleaned, "train": train, "test": test, "model": model, "results": results}


def test_cleaning_and_split_hand_off_valid_partitions(pipeline_outputs):
    raw = pd.read_csv(pipeline_outputs["raw"])
    cleaned = pd.read_csv(pipeline_outputs["cleaned"])
    train, test = pd.read_csv(pipeline_outputs["train"]), pd.read_csv(pipeline_outputs["test"])

    assert len(cleaned) < len(raw)  # blank text and invalid score dropped
    assert not cleaned["Text"].str.contains("<br|&amp;", regex=True).any()
    assert set(cleaned["Sentiment"]) == set(CLASS_LABELS)
    assert len(train) + len(test) == len(cleaned)
    assert set(train["Text"]).isdisjoint(test["Text"])
    assert set(test["Sentiment"]) == set(CLASS_LABELS)  # stratified


def test_runner_exports_and_artifact(pipeline_outputs):
    results = pipeline_outputs["results"]
    analysis = json.loads((results / "analysis.json").read_text())
    assert analysis["metadata"]["data_description"] == "SYNTHETIC"
    assert (results / "metrics.csv").is_file()
    assert pipeline_outputs["model"].is_file()


def test_saved_artifact_serves_predictions(pipeline_outputs):
    model = load_model(pipeline_outputs["model"])
    for expected, text in synthetic.PROBES.items():
        result = predict_review(model, text)
        assert result["label"] == expected
        assert list(result["probabilities"]) == list(CLASS_LABELS)
        assert sum(result["probabilities"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize("expected", CLASS_LABELS)
def test_app_predicts_with_pipeline_artifact(pipeline_outputs, app_test, expected):
    at = app_test(pipeline_outputs["model"])
    assert not at.exception
    assert not at.info  # no "model missing" notice
    at.text_area(key="review").input(synthetic.PROBES[expected])
    at.button[-1].click().run()

    assert not at.exception
    assert [s.value for s in at.main.subheader] == [expected]
    assert at.main.metric[0].label == "Predicted-class probability"
    assert at.main.metric[0].value.endswith("%")
    assert all(c in at.main.caption[-1].value for c in CLASS_LABELS)
    assert len(synthetic.vega_charts(at)) == 1  # probability bar chart


@pytest.mark.slow
def test_group_sample_script_is_deterministic(tmp_path, run_script):
    """make_group_sample.py hardcodes a 50,000-row sample, so feed it a larger synthetic file."""
    raw = synthetic.write_raw_reviews(tmp_path / "Reviews.csv", n_per_class=17_000)
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    run_script("scripts/make_group_sample.py", "--reviews", str(raw), "--out", str(first))
    run_script("scripts/make_group_sample.py", "--reviews", str(raw), "--out", str(second))

    sample = pd.read_csv(first)
    assert len(sample) == 50_000
    assert sample["Id"].is_unique
    assert Path(first).read_bytes() == Path(second).read_bytes()
