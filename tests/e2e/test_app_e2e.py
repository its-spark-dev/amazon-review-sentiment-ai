"""Browser tests: a real user's view of the app served by `streamlit run`."""

import re

from playwright.sync_api import Page, expect
import pytest

from inference import MAX_REVIEW_CHARS
import synthetic

pytestmark = pytest.mark.e2e

ANALYZE = "Analyze sentiment"


def enter_review(page: Page, text: str):
    box = page.get_by_role("textbox", name="Review text")
    box.fill(text)
    box.blur()  # commit the widget value before submitting, as a user's click-away would
    return box


def analyze(page: Page):
    page.get_by_role("button", name=ANALYZE).click()


LABEL_HEADING = re.compile(r"^(Positive|Neutral|Negative)$")


def result_heading(page: Page, label: str | re.Pattern = LABEL_HEADING):
    # Sidebar headings are About/Labels/Model, so a label-named heading is the result.
    return page.get_by_role("heading", name=label, exact=isinstance(label, str))


def assert_no_app_exception(page: Page):
    expect(page.get_by_test_id("stException")).to_have_count(0)


def test_page_loads_with_sidebar(app: Page):
    expect(app).to_have_title(re.compile("Amazon Review Sentiment AI"))
    sidebar = app.get_by_test_id("stSidebar")
    expect(sidebar.get_by_role("heading", name="About")).to_be_visible()
    expect(sidebar.get_by_text("sentiment_pipeline.joblib")).to_be_visible()
    expect(app.get_by_role("button", name=ANALYZE)).to_be_enabled()
    assert_no_app_exception(app)


def test_no_model_shows_notice_and_disables_analyze(page: Page, app_url_no_model):
    page.goto(app_url_no_model)
    expect(page.get_by_text("No trained model found yet")).to_be_visible()
    expect(page.get_by_role("button", name=ANALYZE)).to_be_disabled()
    assert_no_app_exception(page)


@pytest.mark.parametrize("sample", ["Positive", "Negative"])
def test_sample_button_then_analyze(app: Page, sample):
    app.get_by_role("button", name=sample, exact=True).click()
    expect(app.get_by_role("textbox", name="Review text")).not_to_have_value("")
    analyze(app)
    expect(app.get_by_test_id("stMetric")).to_contain_text("Confidence")
    expect(result_heading(app)).to_have_count(1)
    assert_no_app_exception(app)


@pytest.mark.parametrize("label", ["Positive", "Neutral", "Negative"])
def test_typed_review_shows_label_chart_and_breakdown(app: Page, label):
    enter_review(app, synthetic.PROBES[label])
    analyze(app)
    expect(result_heading(app, label)).to_be_visible()
    expect(app.get_by_test_id("stMetricValue")).to_have_text(re.compile(r"^\d+%$"))
    expect(app.get_by_test_id("stVegaLiteChart")).to_be_visible()
    expect(app.get_by_text(re.compile(r"Negative: [\d.]+% · Neutral: [\d.]+% · Positive: [\d.]+%"))).to_be_visible()
    assert_no_app_exception(app)


def test_empty_submit_warns(app: Page):
    analyze(app)
    expect(app.get_by_text("Enter a review to analyze.")).to_be_visible()
    expect(app.get_by_test_id("stMetric")).to_have_count(0)


def test_textarea_caps_input_at_max_chars(app: Page):
    box = app.get_by_role("textbox", name="Review text")
    box.fill("a" * (MAX_REVIEW_CHARS - 5))
    box.press_sequentially("b" * 20)  # typing past maxlength is blocked by the browser
    expect(box).to_have_value("a" * (MAX_REVIEW_CHARS - 5) + "b" * 5)


def test_reanalyze_replaces_previous_result(app: Page):
    enter_review(app, synthetic.PROBES["Positive"])
    analyze(app)
    expect(result_heading(app, "Positive")).to_be_visible()
    enter_review(app, synthetic.PROBES["Negative"])
    analyze(app)
    expect(result_heading(app, "Negative")).to_be_visible()
    expect(result_heading(app, "Positive")).to_have_count(0)
