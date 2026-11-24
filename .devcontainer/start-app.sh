#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
port="${STREAMLIT_SERVER_PORT:-8501}"
browser_address=localhost
if [[ "${CODESPACES:-false}" == true ]]; then
  # Allow this forwarded host, rather than disabling origin/XSRF checks globally.
  browser_address="${CODESPACE_NAME:?Missing CODESPACE_NAME}-${port}.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:?Missing forwarding domain}"
fi

exec python -m streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port "$port" \
  --server.headless true \
  --server.enableCORS true \
  --server.enableXsrfProtection true \
  --browser.serverAddress "$browser_address"
