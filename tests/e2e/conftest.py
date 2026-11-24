"""Live `streamlit run` servers for browser tests: one model, no model, and a models directory."""

from contextlib import contextmanager
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

import synthetic

ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def streamlit_server(log_path: Path, *, model_path: Path | None = None, model_dir: Path | None = None,
                     timeout: float = 60):
    """Serve app.py with either one model file (SENTIMENT_MODEL_PATH) or a models directory."""
    port = _free_port()
    # Never inherit the developer's model settings: each server sees only what the test configures.
    env = {k: v for k, v in os.environ.items() if k not in ("SENTIMENT_MODEL_PATH", "SENTIMENT_MODEL_DIR", "MODELS_VERSION_TAG")}
    env["MPLBACKEND"] = "Agg"
    if model_path is not None:
        env["SENTIMENT_MODEL_PATH"] = str(model_path)
    if model_dir is not None:
        env["SENTIMENT_MODEL_DIR"] = str(model_dir)
    command = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
               "--server.port", str(port), "--server.address", "127.0.0.1",
               "--server.headless", "true", "--browser.gatherUsageStats", "false",
               "--server.fileWatcherType", "none"]
    with open(log_path, "w") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + timeout
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"streamlit exited early:\n{log_path.read_text()}")
                try:
                    with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError(f"streamlit not healthy after {timeout}s:\n{log_path.read_text()}")
                time.sleep(0.5)
            yield url
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


@pytest.fixture(scope="session")
def app_url(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e_model")
    model = synthetic.save_model(root / "sentiment_pipeline.joblib")
    with streamlit_server(root / "streamlit.log", model_path=model) as url:
        yield url


@pytest.fixture(scope="session")
def app_url_no_model(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e_no_model")
    with streamlit_server(root / "streamlit.log", model_path=root / "missing.joblib") as url:
        yield url


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory):
    """A models directory like a real M3 run: the default model, a second model and a broken file.

    ``b_flipped`` swaps Positive/Negative so switching models visibly changes the prediction;
    ``c_corrupt`` is not a joblib file, exercising the per-model load error.
    """
    root = tmp_path_factory.mktemp("e2e_models")
    synthetic.save_model(root / "sentiment_pipeline.joblib")
    synthetic.save_model(root / "b_flipped.joblib", flipped=True)
    (root / "c_corrupt.joblib").write_bytes(b"not a joblib file")
    return root


@pytest.fixture(scope="session")
def app_url_multi_model(tmp_path_factory, models_dir):
    log = tmp_path_factory.mktemp("e2e_multi") / "streamlit.log"
    with streamlit_server(log, model_dir=models_dir) as url:
        yield url


@pytest.fixture
def app(page, app_url):
    page.goto(app_url)
    page.get_by_role("heading", name="Amazon Review Sentiment AI").wait_for()
    return page
