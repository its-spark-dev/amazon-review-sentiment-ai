"""Every model the M3 runner saves must work through inference.py and the app."""

import pytest

from inference import load_model, predict_review
from ml_baseline import CLASS_LABELS
import synthetic

pytestmark = pytest.mark.integration

M3_MODELS = ("bow_nb", "tfidf_nb", "bow_lr", "tfidf_lr", "tfidf_mlp")


@pytest.fixture(scope="module")
def m3_output(tmp_path_factory):
    from scripts.clean_reviews import clean
    from scripts.split_reviews import split

    root = tmp_path_factory.mktemp("m3")
    train, test = split(clean(synthetic.raw_reviews(n_per_class=60, seed=3)))
    train.to_csv(root / "train.csv", index=False)
    test.to_csv(root / "test.csv", index=False)
    out = root / "out"
    from scripts.run_m3_comparison import run

    run(root / "train.csv", root / "test.csv", out, epochs=3, max_features=500)
    return out


def test_runner_saves_every_non_majority_model(m3_output):
    assert sorted(p.stem for p in m3_output.glob("*.joblib")) == sorted(M3_MODELS)
    assert not (m3_output / "majority.joblib").exists()


@pytest.mark.parametrize("name", M3_MODELS)
def test_m3_artifact_inference_contract(m3_output, name):
    model = load_model(m3_output / f"{name}.joblib")
    assert set(map(str, model.classes_)) == set(CLASS_LABELS)
    result = predict_review(model, synthetic.PROBES["Negative"])
    assert result["label"] in CLASS_LABELS
    assert list(result["probabilities"]) == list(CLASS_LABELS)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    # The reported label is the model's own argmax, so the app's confidence is consistent.
    assert result["probabilities"][result["label"]] == max(result["probabilities"].values())


@pytest.mark.parametrize("name", M3_MODELS)
def test_app_serves_m3_artifact(m3_output, app_test, name):
    at = app_test(m3_output / f"{name}.joblib")
    assert not at.exception and not at.error
    at.text_area(key="review").input(synthetic.PROBES["Positive"])
    at.button[-1].click().run()
    assert not at.exception
    assert at.main.subheader[0].value in CLASS_LABELS
    assert at.main.metric[0].value.endswith("%")


@pytest.fixture
def app_on_dir(monkeypatch):
    """Run app.py with model discovery over a directory (no single-file override)."""
    from pathlib import Path

    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.cache_resource.clear()
    root = Path(__file__).resolve().parents[2]

    def run(directory):
        monkeypatch.delenv("SENTIMENT_MODEL_PATH", raising=False)
        monkeypatch.setenv("SENTIMENT_MODEL_DIR", str(directory))
        return AppTest.from_file(str(root / "app.py"), default_timeout=60).run()

    yield run
    st.cache_resource.clear()


def test_app_discovers_every_m3_artifact(m3_output, app_on_dir):
    at = app_on_dir(m3_output)
    assert not at.exception and not at.error
    selector = at.selectbox(key="model_choice")
    assert sorted(selector.options) == sorted(M3_MODELS)
    assert f"{len(M3_MODELS)} models available" in [c.value for c in at.sidebar.caption]


def test_selector_serves_each_m3_artifact(m3_output, app_on_dir):
    at = app_on_dir(m3_output)
    for name in M3_MODELS:
        at.selectbox(key="model_choice").select(str(m3_output / f"{name}.joblib"))
        at.text_area(key="review").input(synthetic.PROBES["Negative"])
        at.button[-1].click().run()
        assert not at.exception and not at.error, name
        assert any(f"{name}.joblib" in c.value for c in at.sidebar.caption), name
        assert at.main.subheader[0].value in CLASS_LABELS, name
