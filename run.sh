#!/usr/bin/env bash
# Convenience launcher: creates the venv on first run, then starts the app.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip
  ./.venv/bin/pip install -r src/requirements.txt
fi

exec ./.venv/bin/streamlit run src/app.py
