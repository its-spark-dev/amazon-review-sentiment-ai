"""Watcher reports, background prefetch and submit working together in the app (AppTest).

The browser half of the watcher can't run in AppTest, so these tests set the
state it would report (``form_watch.pending``) and the rerun flag its callback
sets, exactly as a real report does, then rerun the script.
"""

from pathlib import Path
from unittest import mock

import pytest

import form_watcher
import prefetch
import synthetic

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def report(at, text, model=None):
    """Simulate the watcher reporting the form's pending values (one rerun)."""
    at.session_state[form_watcher.KEY] = {"pending": {"text": text, "model": model}}
    at.session_state["_watch_rerun"] = True
    return at.run()


def submit(at, text):
    at.text_area(key="review").input(text)
    at.button[-1].click().run()
    assert not at.exception
    return [s.value for s in at.main.subheader]


@pytest.fixture
def counted_predict():
    with mock.patch("prefetch.predict_review", wraps=prefetch.predict_review) as predict:
        yield predict


def test_report_prefetches_and_submit_reuses_it(model_file, app_test, counted_predict):
    at = report(app_test(model_file), synthetic.PROBES["Positive"])
    assert not at.exception
    assert counted_predict.call_count == 1  # computed in the background, before any submit
    assert at.main.subheader == []          # ...and nothing is shown yet
    assert submit(at, synthetic.PROBES["Positive"]) == ["Positive"]
    assert counted_predict.call_count == 1  # the submit reused the prefetched outcome


def test_submit_of_different_text_computes_it(model_file, app_test, counted_predict):
    at = report(app_test(model_file), synthetic.PROBES["Positive"])
    assert submit(at, synthetic.PROBES["Negative"]) == ["Negative"]
    assert counted_predict.call_count == 2


def test_replaced_model_file_is_not_served_from_a_stale_prefetch(model_file, app_test):
    at = report(app_test(model_file), synthetic.PROBES["Positive"])
    synthetic.save_model(model_file.with_name("flipped.joblib"), flipped=True).replace(model_file)
    assert submit(at, synthetic.PROBES["Positive"]) == ["Negative"]


def test_prefetched_warning_matches_submit(model_file, app_test):
    at = report(app_test(model_file), "quuxzzblorb snarglexyz")
    at.text_area(key="review").input("quuxzzblorb snarglexyz")
    at.button[-1].click().run()
    assert "No words in this review are recognized" in at.warning[0].value
    assert at.main.subheader == []


def test_result_stays_through_watcher_reruns_until_another_action(model_file, app_test):
    at = app_test(model_file)
    assert submit(at, synthetic.PROBES["Positive"]) == ["Positive"]
    report(at, synthetic.PROBES["Positive"] + " more words")
    assert [s.value for s in at.main.subheader] == ["Positive"]  # typing doesn't wipe the result
    next(b for b in at.button if b.label == "Negative").click().run()
    assert at.main.subheader == []  # a sample click still clears it, as before


def test_sample_click_prefetches_the_sample(model_file, app_test, counted_predict):
    at = app_test(model_file)
    next(b for b in at.button if b.label == "Positive").click().run()
    assert counted_predict.call_count == 1
    at.button[-1].click().run()
    assert at.main.subheader[0].value == "Positive"
    assert counted_predict.call_count == 1


def test_model_switch_prefetches_on_the_newly_picked_model(tmp_path, monkeypatch, counted_predict):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    synthetic.save_model(tmp_path / "a_normal.joblib")
    synthetic.save_model(tmp_path / "b_flipped.joblib", flipped=True)
    monkeypatch.delenv("SENTIMENT_MODEL_PATH", raising=False)
    monkeypatch.setenv("SENTIMENT_MODEL_DIR", str(tmp_path))
    st.cache_resource.clear()
    try:
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        text = synthetic.PROBES["Positive"]
        report(at, text, model="b_flipped")  # picked in the form, not yet submitted
        assert at.session_state["prefetch_target"]["path"] == tmp_path / "b_flipped.joblib"
        assert counted_predict.call_count == 1
        at.selectbox(key="model_choice").select(str(tmp_path / "b_flipped.joblib"))
        assert submit(at, text) == ["Negative"]
        assert counted_predict.call_count == 1  # served by the prefetch on b_flipped

        report(at, text, model="not_a_model")   # e.g. half-typed filter text: ignored
        assert not at.exception
        assert at.session_state["prefetch_target"]["model"] == "b_flipped"
    finally:
        st.cache_resource.clear()
