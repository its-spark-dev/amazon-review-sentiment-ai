"""Live `streamlit run` servers for browser tests: one with a model, one without."""

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
def streamlit_server(model_path: Path, log_path: Path, timeout: float = 60):
    port = _free_port()
    env = {**os.environ, "SENTIMENT_MODEL_PATH": str(model_path), "MPLBACKEND": "Agg"}
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
    with streamlit_server(model, root / "streamlit.log") as url:
        yield url


@pytest.fixture(scope="session")
def app_url_no_model(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e_no_model")
    with streamlit_server(root / "missing.joblib", root / "streamlit.log") as url:
        yield url


@pytest.fixture
def app(page, app_url):
    page.goto(app_url)
    page.get_by_role("heading", name="Amazon Review Sentiment AI").wait_for()
    return page
