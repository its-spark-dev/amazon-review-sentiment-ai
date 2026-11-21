"""Edge-case models and artifacts flowing through inference.py into the app."""

import joblib
import pytest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from inference import MAX_REVIEW_CHARS, load_model, predict_review
from ml_baseline import CLASS_LABELS, create_nb_pipeline
import synthetic

pytestmark = pytest.mark.integration


def submit(at, text):
    at.text_area(key="review").input(text)
    at.button[-1].click().run()
    return at


def test_model_without_predict_proba_shows_label_only(tmp_path, app_test):
    texts, labels = synthetic.train_texts()
    model = Pipeline([("vectorizer", CountVectorizer()), ("classifier", LinearSVC())]).fit(texts, labels)
    path = tmp_path / "svm.joblib"
    joblib.dump(model, path)

    at = submit(app_test(path), synthetic.PROBES["Negative"])
    assert not at.exception
    assert [s.value for s in at.main.subheader] == ["Negative"]
    assert not at.main.metric
    assert not synthetic.vega_charts(at)


def test_model_trained_on_two_classes_reports_zero_for_missing(tmp_path, app_test):
    texts, labels = synthetic.train_texts()
    keep = [i for i, label in enumerate(labels) if label != "Neutral"]
    model = create_nb_pipeline("bow").fit([texts[i] for i in keep], [labels[i] for i in keep])
    result = predict_review(model, synthetic.PROBES["Positive"])
    assert list(result["probabilities"]) == list(CLASS_LABELS)
    assert result["probabilities"]["Neutral"] == 0.0

    path = tmp_path / "two_class.joblib"
    joblib.dump(model, path)
    at = submit(app_test(path), synthetic.PROBES["Positive"])
    assert not at.exception
    assert "Neutral: 0.0%" in at.main.caption[-1].value


@pytest.mark.parametrize("make_bad", ["corrupt", "unfitted"])
def test_unloadable_artifact_reports_error_and_disables_submit(tmp_path, app_test, make_bad):
    path = tmp_path / "bad.joblib"
    if make_bad == "corrupt":
        path.write_bytes(b"not a joblib file")
    else:
        joblib.dump(create_nb_pipeline("bow"), path)

    at = app_test(path)
    assert not at.exception
    assert "Could not load the model" in at.error[0].value
    assert at.button[-1].disabled


def test_max_length_boundary(model_file, app_test):
    model = load_model(model_file)
    at_limit = ("fantastic " * (MAX_REVIEW_CHARS // 10))[:MAX_REVIEW_CHARS]
    assert predict_review(model, at_limit)["label"] == "Positive"
    with pytest.raises(ValueError, match="limited"):
        predict_review(model, at_limit + "x")

    at = app_test(model_file)
    assert at.text_area(key="review").max_chars == MAX_REVIEW_CHARS
    submit(at, at_limit)
    assert [s.value for s in at.main.subheader] == ["Positive"]
