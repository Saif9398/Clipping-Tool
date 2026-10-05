#!/usr/bin/env bash
# One-shot setup on a fresh Linux/macOS server (needs: python3.12, ffmpeg, git)
set -euo pipefail
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.lock.txt
[ -f .env ] || cp .env.example .env
echo "Setup done. Start the app with:  ./.venv/bin/python -m app.main"
