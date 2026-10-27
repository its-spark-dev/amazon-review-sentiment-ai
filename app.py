"""Streamlit entrypoint for the initial CSE 3521 project scaffold."""

import streamlit as st


st.set_page_config(page_title="Amazon Review Sentiment AI", page_icon="🍎")

st.title("Amazon Review Sentiment AI")
st.caption("The Ohio State University · CSE 3521 · Autumn 2026 Flagship Group Project")

st.info("Repository scaffold only. Model inference is not implemented yet.")

st.write(
    "This project will analyze sentiment in Amazon Fine Food Reviews and compare "
    "classical machine learning with a deep or modern NLP approach."
)
st.write(
    "Data processing, feature representations, model training, and evaluation "
    "will be added in future milestones. No predictions or model results are available."
)
st.markdown("See the repository README for setup instructions and milestone dates.")
