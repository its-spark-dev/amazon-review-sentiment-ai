# Codespaces / development container

The Python 3.11 container installs the existing pinned `requirements.txt` and
starts Streamlit on port 8501 when you attach. No separate, unpinned Streamlit
installation is performed. This affects development only; Community Cloud uses
its own deployment configuration.

The launcher keeps **CORS and XSRF protection enabled**. In Codespaces it sets
`browser.serverAddress` to
`${CODESPACE_NAME}-8501.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}`. Outside
Codespaces it uses `localhost`. The server listens on `0.0.0.0` inside the
container so the forwarded port can reach it. Open port **8501** from the Ports
panel in a browser tab; keep the port private. If an embedded preview has cookie
or connection problems, use **Open in Browser** rather than disabling protection.

A fresh clone has no models. To run the dataset-free rc2 demo, stop the attached
server with Ctrl+C, then run from the repository root:

```bash
MODELS_VERSION_TAG=v0.2.0-rc2 bash .devcontainer/start-app.sh
```

The first demo launch downloads and hash-checks approved GitHub release artifacts.
No Kaggle credentials or dataset are needed. Rebuild an existing container after
pulling changes to `devcontainer.json`. Avoid starting a second server while port
8501 is already in use.

## Validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests/test_devcontainer.py
```

Tests check local/Codespaces address selection, incomplete forwarding metadata,
pinned installation, and a live server using a simulated forwarded hostname.
They verify allowed/rejected WebSocket origins and XSRF token enforcement without
datasets, model downloads or external services. They do not provision a hosted
Codespace or verify GitHub's authenticated forwarding proxy.

CI actions use official Node.js 24 releases: checkout v7.0.1, setup-python v7.1.0,
upload-artifact v7.0.2 and cache v6.1.0. This changes the Actions runtime only;
Python 3.11, runtime package pins and rc2 artifact hashes stay unchanged.

References: [Streamlit configuration](https://docs.streamlit.io/develop/api-reference/configuration/config.toml),
[Codespaces port forwarding](https://docs.github.com/en/codespaces/developing-in-a-codespace/forwarding-ports-in-your-codespace).
