"""Streamlit app: enter an Amazon food review and get a three-class sentiment."""

from concurrent.futures import wait
import logging
import os
from pathlib import Path
import time

import pandas as pd
import streamlit as st

import form_watcher
from inference import (
    MAX_REVIEW_CHARS,
    MODELS_DIR,
    REPORT_MODEL_RELEASE,
    ensure_models_release,
    load_model,
    model_paths,
)
from ml_baseline import CLASS_LABELS
from prefetch import PrefetchStore, evaluate, file_version

st.set_page_config(page_title="Amazon Review Sentiment AI")

logger = logging.getLogger("sentiment.app")
PREFETCH_WORKERS = 2
READY_WAIT_S = 0.3  # how long a watcher rerun waits for its prefetch before the browser polls
UNCALIBRATED = "Probabilities are uncalibrated model estimates. High values can still accompany incorrect predictions."

SAMPLE_REVIEWS = {
    "Positive": "These cookies are fantastic. Fresh, crunchy, and my kids love them. Will buy again!",
    "Neutral": "The coffee is okay. Not as bold as I hoped, but it's drinkable and the price is fair.",
    "Negative": "Arrived stale and the bag was torn open. Tasted awful, complete waste of money.",
}


@st.cache_resource(show_spinner=False)
def get_model(path: str, file_version: tuple):
    """Cache an existing artifact by path and filesystem version.

    file_version deliberately participates in Streamlit's cache key, so normal
    writes and atomic replacements invalidate the previous model resource.
    The prefetch thread calls this too, so it never shows a spinner itself;
    the script shows one around a load that is not known to be warm.
    """
    model = load_model(path)
    _warm_models().add((path, file_version))
    return model


@st.cache_resource
def _warm_models() -> set:
    """Keys get_model has loaded; cleared with the cache it describes."""
    return set()


@st.cache_resource
def _executor():
    from concurrent.futures import ThreadPoolExecutor

    return ThreadPoolExecutor(max_workers=PREFETCH_WORKERS, thread_name_prefix="prefetch")


def _prepare(path: str, version: tuple, text: str) -> dict:
    """Background job: load (or reuse) the model and compute the outcome to render."""
    return evaluate(get_model(path, version), text)


def _store() -> PrefetchStore:
    if "prefetch" not in st.session_state:
        st.session_state["prefetch"] = PrefetchStore(_executor())
    return st.session_state["prefetch"]


def render(outcome: dict) -> None:
    """Show a prediction outcome: the warning, or the label, probability, chart and breakdown."""
    if "warning" in outcome:
        st.warning(outcome["warning"])
        return
    view = outcome["view"]
    st.subheader(view["label"])
    if view["probs"] is not None:
        st.metric("Predicted-class probability", view["metric"])
        st.caption(UNCALIBRATED)
        chart = pd.DataFrame({"Probability": view["probs"]}, index=list(CLASS_LABELS))
        st.bar_chart(chart, horizontal=True)
        st.caption(view["breakdown"])


def sidebar(path, count: int) -> None:
    with st.sidebar:
        st.header("About")
        st.write(
            "Group Psi · CSE 3521 · The Ohio State University, Autumn 2026. "
            "Data: Amazon Fine Food Reviews."
        )
        st.subheader("Labels")
        st.markdown("- 1–2 stars: **Negative**\n- 3 stars: **Neutral**\n- 4–5 stars: **Positive**")
        st.subheader("Model")
        st.caption(f"`{path}`" if path else "No model available")
        if count > 1:
            st.caption(f"{count} models available")


def load_selected(path):
    """Load the chosen artifact through the cache, or show an error and return None."""
    try:
        # Check outside the cache: a missing artifact must not cache None.
        if not path.is_file():
            return None
        key = (str(path.resolve()), file_version(path))
        if key in _warm_models():
            return get_model(*key)
        with st.spinner("Loading model…"):
            return get_model(*key)
    except Exception as exc:  # corrupt or incompatible artifact
        st.error(f"Could not load the model `{path.name}`: {exc}")
        return None


def _mark_watch_rerun() -> None:
    st.session_state["_watch_rerun"] = True


def _start_prefetch(path: Path, text: str):
    """Start (or find) the prefetch for ``text`` on ``path``; return its key and Future."""
    try:
        version = file_version(path)
    except OSError:
        return None, None
    store = _store()
    key = store.key(path, version, text)
    resolved = key[0]
    return key, store.submit(key, lambda: _prepare(resolved, version, text))


def _submitted_outcome(selected: Path, model, review: str) -> dict:
    """The outcome for a submit: the prefetched one when it matches, else computed now."""
    started = time.perf_counter()
    source = "computed"
    try:
        future = _store().get(_store().key(selected, file_version(selected), review))
    except OSError:
        future = None
    outcome = None
    if future is not None:
        try:
            outcome = future.result()
            source = "prefetched" if future.done() else "awaited"
        except Exception:  # load failed in the background; the foreground path reports it
            outcome = None
    if outcome is None:
        outcome = evaluate(model, review)
    logger.debug("submit outcome %s in %.1f ms", source, (time.perf_counter() - started) * 1000)
    return outcome


def main() -> None:
    release_tag = os.environ.get("MODELS_VERSION_TAG")  # GitHub release tag to download the models from

    if release_tag:
        if release_tag != REPORT_MODEL_RELEASE:
            st.error(f"Report/demo model version is {REPORT_MODEL_RELEASE}; update MODELS_VERSION_TAG.")
            st.stop()
        # Same directory model_paths() scans, so downloaded artifacts are listed below.
        model_dir = Path(os.environ.get("SENTIMENT_MODEL_DIR", MODELS_DIR))
        try:
            paths = ensure_models_release(release_tag, model_dir)
        except (OSError, ValueError):
            st.error("Could not prepare the release models. Please try again later.")
            st.stop()

    else:
        paths = model_paths()  # local mode; release mode lists only verified report models
    options = [str(p) for p in paths]
    if st.session_state.get("model_choice") not in options:
        if paths:
            default = next((p for p in paths if p.name == "tfidf_lr.joblib"), paths[0])
            st.session_state["model_choice"] = str(default)
        else:
            st.session_state.pop("model_choice", None)
    selected = Path(st.session_state["model_choice"]) if options else None
    sidebar(selected, len(paths))

    st.title("Amazon Review Sentiment AI")
    st.caption("Classify an English food product review as Negative, Neutral, or Positive.")
    if release_tag:
        st.caption("Recommended rc2 model: tfidf_lr, selected by validation Macro-F1.")

    model = load_selected(selected) if selected else None
    if not paths:
        st.info(
            "No trained model found yet. Save a fitted pipeline as a `.joblib` file "
            "in `models/` (or set `SENTIMENT_MODEL_PATH`) to enable predictions."
        )

    watch_rerun = st.session_state.pop("_watch_rerun", False)
    has_selector = len(paths) > 1
    by_stem = {p.stem: p for p in paths}

    def target_for(text: str, model_value):
        """Where a prefetch for the form's pending values should run, or None if unknown."""
        if not has_selector:
            return {"path": selected, "model": None, "text": text}
        model_value = model_value if model_value is not None else selected.stem
        path = by_stem.get(model_value)
        return {"path": path, "model": model_value, "text": text} if path else None

    target = None
    pending = form_watcher.pending() if watch_rerun else None
    if pending is not None and selected is not None:
        st.session_state["pending_model"] = pending["model"]
        target = target_for(pending["text"], pending["model"])

    st.write("Try a sample:")
    cols = st.columns(len(SAMPLE_REVIEWS))
    for col, (name, text) in zip(cols, SAMPLE_REVIEWS.items()):
        if col.button(name, use_container_width=True):
            st.session_state["review"] = text
            if selected is not None:
                target = target_for(text, st.session_state.get("pending_model"))

    if target is not None:
        st.session_state["prefetch_target"] = target
    target = st.session_state.get("prefetch_target")

    with st.form("review_form"):
        review = st.text_area(
            "Review text",
            key="review",
            height=160,
            max_chars=MAX_REVIEW_CHARS,
            placeholder="Paste or type a product review…",
        )
        if has_selector:
            left, right = st.columns([2, 1], vertical_alignment="bottom")
            left.selectbox("Model", options, key="model_choice", format_func=lambda p: Path(p).stem)
        else:
            right = st.container()
        # The model choice lives in this form, so submitting is the only way to apply a new
        # one. Keep the button enabled while there are other models to switch to, or a
        # broken selected artifact would trap the user on it.
        submitted = right.form_submit_button(
            "Analyze sentiment",
            type="primary",
            disabled=model is None and len(paths) <= 1,
            use_container_width=True,
        )

    # Prefetch what the form currently holds, so a submit (and the browser preview) is instant.
    ready, waiting = None, False
    if target is not None and Path(target["path"]).is_file():
        key, future = _start_prefetch(Path(target["path"]), target["text"])
        if future is not None:
            if not future.done():
                wait([future], timeout=READY_WAIT_S)
            outcome = _store().done_outcome(key)
            waiting = not future.done()
            if outcome is not None and "view" in outcome:
                ready = {"text": target["text"], "model": target["model"], "view": outcome["view"]}

    if paths:
        form_watcher.mount(
            data={
                "ready": ready,
                "waiting": waiting,
                "submitted": bool(submitted and model is not None),
                "labels": list(CLASS_LABELS),
            },
            on_report=_mark_watch_rerun,
        )

    if submitted and model is not None:
        outcome = _submitted_outcome(selected, model, review)
        st.session_state["last_outcome"] = (str(selected), outcome)
        render(outcome)
    elif watch_rerun and st.session_state.get("last_outcome", (None,))[0] == str(selected):
        render(st.session_state["last_outcome"][1])  # typing after a result keeps it on screen
    else:
        st.session_state.pop("last_outcome", None)


main()
