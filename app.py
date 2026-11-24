"""Streamlit app: enter an Amazon food review and get a three-class sentiment."""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from inference import (
    MAX_REVIEW_CHARS,
    MODELS_DIR,
    REPORT_MODEL_RELEASE,
    ensure_models_release,
    load_model,
    model_paths,
    predict_review,
)
from ml_baseline import CLASS_LABELS

st.set_page_config(page_title="Amazon Review Sentiment AI")

SAMPLE_REVIEWS = {
    "Positive": "These cookies are fantastic. Fresh, crunchy, and my kids love them. Will buy again!",
    "Neutral": "The coffee is okay. Not as bold as I hoped, but it's drinkable and the price is fair.",
    "Negative": "Arrived stale and the bag was torn open. Tasted awful, complete waste of money.",
}


@st.cache_resource(show_spinner="Loading model…")
def get_model(path: str, file_version: tuple):
    """Cache an existing artifact by path and filesystem version.

    file_version deliberately participates in Streamlit's cache key, so normal
    writes and atomic replacements invalidate the previous model resource.
    """
    return load_model(path)


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
        stat = path.stat()
        version = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino)
        return get_model(str(path.resolve()), version)
    except Exception as exc:  # corrupt or incompatible artifact
        st.error(f"Could not load the model `{path.name}`: {exc}")
        return None


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

    st.write("Try a sample:")
    cols = st.columns(len(SAMPLE_REVIEWS))
    for col, (name, text) in zip(cols, SAMPLE_REVIEWS.items()):
        if col.button(name, use_container_width=True):
            st.session_state["review"] = text

    with st.form("review_form"):
        review = st.text_area(
            "Review text",
            key="review",
            height=160,
            max_chars=MAX_REVIEW_CHARS,
            placeholder="Paste or type a product review…",
        )
        if len(paths) > 1:
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

    if not submitted or model is None:
        return

    try:
        result = predict_review(model, review)
    except ValueError as exc:
        st.warning(str(exc))
        return

    label = result["label"]
    st.subheader(label)
    probs = result["probabilities"]
    if probs:
        st.metric("Predicted-class probability", f"{probs[label]:.1%}" if label in probs else "—")
        st.caption(
            "Probabilities are uncalibrated model estimates. "
            "High values can still accompany incorrect predictions."
        )
        chart = pd.DataFrame({"Probability": [probs[c] for c in CLASS_LABELS]}, index=list(CLASS_LABELS))
        st.bar_chart(chart, horizontal=True)
        st.caption(" · ".join(f"{c}: {probs[c]:.1%}" for c in CLASS_LABELS))


main()
