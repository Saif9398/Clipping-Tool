# One-shot setup on a fresh Windows machine (needs: python 3.12, ffmpeg, git on PATH)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
Write-Host "Setup done. Start the app with:  .\.venv\Scripts\python.exe -m app.main"
