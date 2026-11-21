"""Shared pytest fixtures; the unittest suites in tests/ do not depend on these."""

import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import synthetic`

import synthetic  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def model_file(tmp_path) -> Path:
    return synthetic.save_model(tmp_path / "sentiment_pipeline.joblib")


@pytest.fixture
def app_test(monkeypatch):
    """Return a factory that runs app.py (AppTest) against a given model path."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.cache_resource.clear()

    def run(model_path: Path):
        monkeypatch.setenv("SENTIMENT_MODEL_PATH", str(model_path))
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()

    yield run
    st.cache_resource.clear()


@pytest.fixture(scope="session")
def run_script():
    """Return a helper that runs a repo script as a subprocess from the repo root."""
    return _run_script


def _run_script(*args: str, check: bool = True):
    import subprocess

    completed = subprocess.run(
        [sys.executable, *args], cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "MPLBACKEND": "Agg"}, check=False,
    )
    if check:
        assert completed.returncode == 0, f"{args[0]} failed:\n{completed.stdout}\n{completed.stderr}"
    return completed
