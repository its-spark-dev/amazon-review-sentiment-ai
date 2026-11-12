"""Streamlit app: enter an Amazon food review and get a three-class sentiment."""

import pandas as pd
import streamlit as st

from inference import MAX_REVIEW_CHARS, load_model, model_path, predict_review
from ml_baseline import CLASS_LABELS

st.set_page_config(page_title="Amazon Review Sentiment AI")

SAMPLE_REVIEWS = {
    "Positive": "These cookies are fantastic. Fresh, crunchy, and my kids love them. Will buy again!",
    "Neutral": "The coffee is okay. Not as bold as I hoped, but it's drinkable and the price is fair.",
    "Negative": "Arrived stale and the bag was torn open. Tasted awful, complete waste of money.",
}


@st.cache_resource(show_spinner="Loading model…")
def get_model(path: str):
    return load_model(path)


def sidebar(path) -> None:
    with st.sidebar:
        st.header("About")
        st.write(
            "Group Psi · CSE 3521 · The Ohio State University, Autumn 2026. "
            "Data: Amazon Fine Food Reviews."
        )
        st.subheader("Labels")
        st.markdown("- 1–2 stars: **Negative**\n- 3 stars: **Neutral**\n- 4–5 stars: **Positive**")
        st.subheader("Model")
        st.caption(f"`{path}`")


def main() -> None:
    path = model_path()
    sidebar(path)

    st.title("Amazon Review Sentiment AI")
    st.caption("Classify a food product review as Negative, Neutral, or Positive.")

    try:
        model = get_model(str(path))
    except Exception as exc:  # corrupt or incompatible artifact
        model = None
        st.error(f"Could not load the model: {exc}")
    else:
        if model is None:
            st.info(
                "No trained model found yet. Save a fitted pipeline to "
                f"`{path.name}` in `models/` (or set `SENTIMENT_MODEL_PATH`) to enable predictions."
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
        submitted = st.form_submit_button("Analyze sentiment", type="primary", disabled=model is None)

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
        st.metric("Confidence", f"{probs[label]:.0%}" if label in probs else "—")
        chart = pd.DataFrame({"Probability": [probs[c] for c in CLASS_LABELS]}, index=list(CLASS_LABELS))
        st.bar_chart(chart, horizontal=True)
        st.caption(" · ".join(f"{c}: {probs[c]:.1%}" for c in CLASS_LABELS))


main()
