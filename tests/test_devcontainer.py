"""Exercise the devcontainer launcher without a Codespace or release downloads."""

import base64
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / ".devcontainer/start-app.sh"


def launcher_env(tmp_path, *, codespaces):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("STREAMLIT_", "CODESPACE", "GITHUB_CODESPACES_", "SENTIMENT_", "MODELS_"))}
    return {**env, "PATH": f"{Path(sys.executable).parent}:{env['PATH']}",
            "CODESPACES": str(codespaces).lower(), "CODESPACE_NAME": "psi-test",
            "GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN": "app.github.dev",
            "SENTIMENT_MODEL_PATH": str(tmp_path / "missing.joblib")}


@pytest.mark.parametrize("codespaces,host", [(False, "localhost"), (True, "psi-test-8501.app.github.dev")])
def test_launcher_uses_protected_forwarded_host(tmp_path, codespaces, host):
    # Capture the actual shell argv, including exec and workspace handling.
    stub = tmp_path / "python"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$@"\n')
    stub.chmod(0o755)
    env = launcher_env(tmp_path, codespaces=codespaces)
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    result = subprocess.run(["bash", str(LAUNCHER)], cwd=tmp_path, env=env,
                            capture_output=True, text=True, check=True)
    args = result.stdout.splitlines()
    assert Path(args[0]).resolve() == ROOT.resolve()
    assert args[1:5] == ["-m", "streamlit", "run", "app.py"]
    flags = dict(zip(args[5::2], args[6::2]))
    assert flags["--server.enableCORS"] == "true"
    assert flags["--server.enableXsrfProtection"] == "true"
    assert flags["--browser.serverAddress"] == host
    assert flags["--server.port"] == "8501"


def test_launcher_requires_codespaces_forwarding_metadata(tmp_path):
    env = launcher_env(tmp_path, codespaces=True)
    del env["GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN"]
    result = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Missing forwarding domain" in result.stderr


def test_devcontainer_installs_pinned_requirements():
    config = json.loads((ROOT / ".devcontainer/devcontainer.json").read_text())
    assert config["updateContentCommand"] == "python -m pip install -r requirements.txt"
    assert config["postAttachCommand"]["server"] == "bash .devcontainer/start-app.sh"
    assert config["portsAttributes"]["8501"]["onAutoForward"] == "openBrowser"


@pytest.mark.integration
def test_forwarded_server_enforces_origin_and_xsrf(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = launcher_env(tmp_path, codespaces=True)
    env["STREAMLIT_SERVER_PORT"] = str(port)
    log_path = tmp_path / "streamlit.log"

    def request(method, path, headers=None):
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            conn.request(method, path, headers=headers or {})
            response = conn.getresponse()
            return response.status, dict(response.getheaders())
        finally:
            conn.close()

    def websocket_status(origin):
        return request("GET", "/_stcore/stream", {
            "Origin": origin, "Upgrade": "websocket", "Connection": "Upgrade",
            "Sec-WebSocket-Version": "13", "Sec-WebSocket-Protocol": "streamlit",
            "Sec-WebSocket-Key": base64.b64encode(os.urandom(16)).decode(),
        })[0]

    with log_path.open("w") as log:
        proc = subprocess.Popen(["bash", str(LAUNCHER)], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            while True:
                assert proc.poll() is None, log_path.read_text()
                try:
                    status, headers = request("GET", "/_stcore/health")
                    if status == 200:
                        break
                except OSError:
                    pass
                assert time.monotonic() < deadline, log_path.read_text()
                time.sleep(0.1)
            assert headers.get("access-control-allow-origin") != "*"
            cookie = headers["set-cookie"].split(";", 1)[0]
            assert cookie.startswith("_streamlit_xsrf=")
            assert websocket_status(f"https://psi-test-{port}.app.github.dev") == 101
            assert websocket_status(f"http://127.0.0.1:{port}") == 101
            assert websocket_status("https://unrelated.invalid") == 403
            upload = "/_stcore/upload_file/missing-session/missing-file"
            assert request("PUT", upload)[0] == 403
            assert request("PUT", upload, {"Cookie": cookie, "X-Xsrftoken": "wrong"})[0] == 403
            # A valid token reaches the next check (nonexistent session), not XSRF rejection.
            assert request("PUT", upload, {"Cookie": cookie, "X-Xsrftoken": cookie.split("=", 1)[1]})[0] == 400
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
