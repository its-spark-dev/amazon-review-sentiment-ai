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
    expect(app.get_by_test_id("stMetric")).to_contain_text("Predicted-class probability")
    expect(result_heading(app)).to_have_count(1)
    assert_no_app_exception(app)


@pytest.mark.parametrize("label", ["Positive", "Neutral", "Negative"])
def test_typed_review_shows_label_chart_and_breakdown(app: Page, label):
    enter_review(app, synthetic.PROBES[label])
    analyze(app)
    expect(result_heading(app, label)).to_be_visible()
    expect(app.get_by_test_id("stMetricValue")).to_have_text(re.compile(r"^\s*\d+\.\d%\s*$"))
    expect(app.get_by_text(re.compile("Probabilities are uncalibrated model estimates"))).to_be_visible()
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


@pytest.mark.parametrize("text", ["🍪🍪🍪", "이 음식은 맛있어요", "quuxzzblorb snarglexyz"])
def test_unrecognized_input_withholds_prediction_and_can_recover(app: Page, text):
    enter_review(app, synthetic.PROBES["Positive"])
    analyze(app)
    expect(result_heading(app, "Positive")).to_be_visible()
    enter_review(app, text)
    analyze(app)
    expect(app.get_by_text(re.compile("No words in this review are recognized"))).to_be_visible()
    expect(result_heading(app)).to_have_count(0)
    expect(app.get_by_test_id("stMetric")).to_have_count(0)
    expect(app.get_by_test_id("stVegaLiteChart")).to_have_count(0)
    enter_review(app, synthetic.PROBES["Negative"])
    analyze(app)
    expect(result_heading(app, "Negative")).to_be_visible()
    expect(app.get_by_text(re.compile("No words in this review are recognized"))).to_have_count(0)
    assert_no_app_exception(app)


# --- Background prefetch and double-buffered result----------------------------------------------

READY = "window.__reviewPrefetch && window.__reviewPrefetch.data && window.__reviewPrefetch.data.ready"


def wait_for_prefetch(page: Page, text: str, model: str | None = None):
    """Wait until the server pushed a finished prefetch for exactly this form state."""
    page.wait_for_function(
        f"({READY}) && {READY}.text === {text!r}" + (f" && {READY}.model === {model!r}" if model else ""),
        timeout=10_000,
    )


CHART = "[data-testid=stVegaLiteChart] canvas, [data-testid=stVegaLiteChart] svg"


def watch_flip(page: Page):
    """Remember the chart drawn in the hidden back slot, and whether a click flips it into view."""
    return page.evaluate(f"""() => {{
        window.__flipped = false;
        new MutationObserver(() => {{
            if (document.getElementById('rp-flip')) window.__flipped = true;
        }}).observe(document.head, {{childList: true}});
        const back = document.querySelector('.st-key-' + window.__reviewPrefetch.data.back);
        window.__buffered = back && back.querySelector({CHART!r});
        return !!window.__buffered && !window.__buffered.checkVisibility({{visibilityProperty: true}});
    }}""")


def buffered_chart_is_shown(page: Page) -> bool:
    """The chart on screen is the very element the browser drew while it was hidden."""
    return page.evaluate("""() => !!window.__buffered && document.contains(window.__buffered)
        && window.__buffered.checkVisibility({visibilityProperty: true})""")


def type_review(page: Page, text: str):
    box = page.get_by_role("textbox", name="Review text")
    box.click()
    box.fill(text)  # fires an input event, like typing, and is not submitted
    return box


def test_pause_prefetches_and_renders_hidden_then_click_flips_it_into_view(app: Page):
    text = synthetic.PROBES["Positive"]
    type_review(app, text)
    wait_for_prefetch(app, text)
    # The result is already drawn in the page, chart included, but hidden until Analyze.
    app.wait_for_function(
        f"document.querySelector('.st-key-' + window.__reviewPrefetch.data.back + ' ' + {CHART!r})"
    )
    expect(result_heading(app)).to_have_count(0)
    assert watch_flip(app) is True
    analyze(app)
    expect(result_heading(app, "Positive")).to_be_visible()
    assert app.evaluate("window.__flipped") is True
    # The submit run rendered into the same slot: no new chart, and the flip override is gone.
    app.wait_for_function("!document.getElementById('rp-flip')")
    assert buffered_chart_is_shown(app)
    expect(result_heading(app)).to_have_count(1)
    assert_no_app_exception(app)


def test_edit_after_prefetch_does_not_flip_a_stale_result(app: Page):
    type_review(app, synthetic.PROBES["Positive"])
    wait_for_prefetch(app, synthetic.PROBES["Positive"])
    watch_flip(app)
    enter_review(app, synthetic.PROBES["Negative"])
    analyze(app)  # before the 1 s pause: no prefetch for this text yet
    expect(result_heading(app, "Negative")).to_be_visible()
    expect(result_heading(app)).to_have_count(1)
    assert app.evaluate("window.__flipped") is False
    assert_no_app_exception(app)


def test_result_stays_visible_while_typing_more(app: Page):
    enter_review(app, synthetic.PROBES["Positive"])
    analyze(app)
    expect(result_heading(app, "Positive")).to_be_visible()
    text = synthetic.PROBES["Positive"] + " and more"
    type_review(app, text)
    wait_for_prefetch(app, text)  # the watcher rerun has happened
    expect(result_heading(app, "Positive")).to_be_visible()
    assert_no_app_exception(app)


# --- Model selection (several .joblib files in the models directory) ---------------------------


def open_multi(page: Page, url: str) -> Page:
    page.goto(url)
    page.get_by_role("heading", name="Amazon Review Sentiment AI").wait_for()
    return page


def model_select(page: Page):
    return page.get_by_test_id("stSelectbox").filter(has_text="Model")


def choose_model(page: Page, name: str):
    model_select(page).click()
    page.get_by_role("option", name=name, exact=True).click()


def test_selector_lists_models_default_first(page: Page, app_url_multi_model):
    app = open_multi(page, app_url_multi_model)
    sidebar = app.get_by_test_id("stSidebar")
    expect(sidebar.get_by_text("sentiment_pipeline.joblib")).to_be_visible()
    expect(sidebar.get_by_text("3 models available")).to_be_visible()
    model_select(app).click()
    expect(app.get_by_role("option")).to_have_text(["sentiment_pipeline", "b_flipped", "c_corrupt"])
    app.keyboard.press("Escape")
    assert_no_app_exception(app)


def test_switching_model_changes_prediction(page: Page, app_url_multi_model):
    app = open_multi(page, app_url_multi_model)
    enter_review(app, synthetic.PROBES["Positive"])
    analyze(app)
    expect(result_heading(app, "Positive")).to_be_visible()

    choose_model(app, "b_flipped")
    analyze(app)  # the choice lives in the form, so it takes effect on submit
    expect(result_heading(app, "Negative")).to_be_visible()
    expect(app.get_by_test_id("stSidebar").get_by_text("b_flipped.joblib")).to_be_visible()
    assert_no_app_exception(app)


def test_corrupt_model_shows_error_and_user_can_switch_away(page: Page, app_url_multi_model):
    app = open_multi(page, app_url_multi_model)
    choose_model(app, "c_corrupt")
    analyze(app)
    expect(app.get_by_role("alert").filter(has_text="Could not load the model")).to_contain_text("c_corrupt.joblib")
    expect(result_heading(app)).to_have_count(0)

    # A broken artifact must not trap the user: picking a working model recovers.
    choose_model(app, "sentiment_pipeline")
    enter_review(app, synthetic.PROBES["Negative"])
    analyze(app)
    expect(result_heading(app, "Negative")).to_be_visible()
    expect(app.get_by_role("alert").filter(has_text="Could not load the model")).to_have_count(0)
    assert_no_app_exception(app)


def test_recommended_model_defaults_and_user_choice_persists(page: Page, app_url_recommended_model):
    app = open_multi(page, app_url_recommended_model)
    sidebar = app.get_by_test_id("stSidebar")
    expect(sidebar.get_by_text("tfidf_lr.joblib")).to_be_visible()
    expect(app.get_by_role("combobox", name="Model")).to_have_value("tfidf_lr")
    enter_review(app, synthetic.PROBES["Positive"])
    analyze(app)
    expect(result_heading(app, "Positive")).to_be_visible()
    choose_model(app, "bow_lr")
    analyze(app)
    expect(result_heading(app, "Negative")).to_be_visible()
    app.get_by_role("button", name="Positive", exact=True).click()  # rerun outside the form
    expect(app.get_by_role("combobox", name="Model")).to_have_value("bow_lr")
    analyze(app)
    expect(sidebar.get_by_text("bow_lr.joblib")).to_be_visible()
    assert_no_app_exception(app)


def test_model_switch_prefetches_before_submit(page: Page, app_url_multi_model):
    app = open_multi(page, app_url_multi_model)
    text = synthetic.PROBES["Positive"]
    type_review(app, text)
    wait_for_prefetch(app, text, "sentiment_pipeline")
    choose_model(app, "b_flipped")
    wait_for_prefetch(app, text, "b_flipped")  # reported at once, without waiting for typing
    expect(result_heading(app)).to_have_count(0)
    assert watch_flip(app) is True
    analyze(app)
    expect(result_heading(app, "Negative")).to_be_visible()
    assert app.evaluate("window.__flipped") is True
    app.wait_for_function("!document.getElementById('rp-flip')")
    assert buffered_chart_is_shown(app)
    expect(app.get_by_test_id("stSidebar").get_by_text("b_flipped.joblib")).to_be_visible()
    assert_no_app_exception(app)
