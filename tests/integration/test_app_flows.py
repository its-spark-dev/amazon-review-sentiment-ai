"""How the app's widgets, session state and inference work together (AppTest)."""

import pytest

from inference import model_path

pytestmark = pytest.mark.integration

SAMPLE_BUTTONS = ("Positive", "Neutral", "Negative")


def sample_button(at, name):
    return next(b for b in at.button if b.label == name)


@pytest.mark.parametrize("name", SAMPLE_BUTTONS)
def test_sample_button_fills_review_then_predicts(model_file, app_test, name):
    at = app_test(model_file)
    sample_button(at, name).click().run()
    review = at.text_area(key="review").value
    assert review and review == at.session_state["review"]

    at.button[-1].click().run()
    assert not at.exception
    assert len(at.main.subheader) == 1
    assert at.main.metric[0].label == "Confidence"


def test_typed_review_overrides_sample(model_file, app_test):
    at = app_test(model_file)
    sample_button(at, "Positive").click().run()
    at.text_area(key="review").input("awful stale terrible gross")
    at.button[-1].click().run()
    assert [s.value for s in at.main.subheader] == ["Negative"]


@pytest.mark.parametrize("text", ["", "   \n\t "])
def test_blank_submit_warns_without_prediction(model_file, app_test, text):
    at = app_test(model_file)
    at.text_area(key="review").input(text)
    at.button[-1].click().run()
    assert not at.exception
    assert at.warning[0].value == "Enter a review to analyze."
    assert not at.main.subheader


def test_sidebar_shows_configured_model_and_labels(model_file, app_test):
    at = app_test(model_file)
    assert model_path() == model_file
    sidebar = at.sidebar
    assert [h.value for h in sidebar.header] == ["About"]
    assert any(str(model_file) in c.value for c in sidebar.caption)
    legend = " ".join(m.value for m in sidebar.markdown)
    for label in SAMPLE_BUTTONS:
        assert f"**{label}**" in legend


def test_no_prediction_rendered_before_submit(model_file, app_test):
    at = app_test(model_file)
    assert not at.main.subheader and not at.main.metric
    assert not at.button[-1].disabled
