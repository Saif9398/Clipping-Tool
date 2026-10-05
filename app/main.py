"""FastAPI app: JSON API + SSE progress + static UI + clip serving.

Run:  python -m app.main   (or: uvicorn app.main:app)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

_RESTART_EXIT_CODE = 75
_SERVER_WORKER_ENV = "CLIPPING_TOOL_SERVER_WORKER"


def _run_supervised_server() -> int:
    """Keep the web worker restartable when launched with ``python -m app.main``."""
    env = os.environ.copy()
    env[_SERVER_WORKER_ENV] = "1"
    command = [sys.executable, "-m", "app.main"]
    project_root = Path(__file__).resolve().parent.parent
    while True:
        result = subprocess.run(command, cwd=project_root, env=env, check=False)
        if result.returncode != _RESTART_EXIT_CODE:
            return result.returncode


# Do this before importing FastAPI/pipeline modules so the lightweight parent
# process does not load models or job state. The worker below owns the app.
if __name__ == "__main__" and os.getenv(_SERVER_WORKER_ENV) != "1":
    raise SystemExit(_run_supervised_server())

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (FileResponse, JSONResponse, RedirectResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import auth, config, diskspace, jobs, retention, version
from app.hardware import detect
from app.pipeline import mp_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Clipping Tool")
STATIC = Path(__file__).parent / "static"
_shutdown_lock = threading.Lock()
_shutdown_requested = False
_restart_requested = False

log.info("code version: %s (restart this process after every code change — "
         "a running server keeps serving the old code)", version.commit())
mp_models.ensure_models()  # before accepting jobs: no worker should ever race a missing model
diskspace.sweep_orphan_parts()  # stale *.part temps from hard kills / OOM
retention.cleanup_old(config.RETENTION_DAYS)  # before load: pruned jobs never load
jobs.load_existing_jobs()
_disk = diskspace.usage_summary()
log.info("disk: %.1f GB free on work volume (work=%.1f GB, output=%.1f GB, "
         "models=%.1f GB; warn<%.0f GB, min<%.0f GB)",
         _disk["free_gb_work_volume"], _disk["work_dir_gb"], _disk["output_dir_gb"],
         _disk["models_dir_gb"], _disk["warn_free_gb"], _disk["min_free_gb"])
if _disk["free_gb_work_volume"] < config.WARN_FREE_GB:
    log.warning("LOW DISK SPACE at startup: %.1f GB free — jobs may be blocked "
                "(MIN_FREE_GB=%.0f)", _disk["free_gb_work_volume"], config.MIN_FREE_GB)
if config.MAX_CONCURRENT_JOBS > 1 and detect().has_cuda:
    log.warning("MAX_CONCURRENT_JOBS=%d on a CUDA machine: whisper (~3 GB VRAM) "
                "and NVENC render sessions can now run at the same time — the "
                "single-job VRAM safety guarantee does not hold. Keep it at 1 "
                "unless you have measured headroom.", config.MAX_CONCURRENT_JOBS)
if not auth.enabled():
    log.warning("ADMIN_PASSWORD not set — login gate DISABLED (fine for local dev, "
                "set it in .env before exposing this server)")


# ---- Auth gate ----

_PUBLIC_PATHS = ("/login", "/api/login", "/favicon.ico")


class AuthGate:
    """Pure ASGI middleware (BaseHTTPMiddleware has SSE/streaming edge cases).
    Gates every route AND both static mounts — /clips holds the rendered videos."""

    def __init__(self, app_):
        self.app = app_

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not auth.enabled():
            return await self.app(scope, receive, send)
        path = scope["path"]
        if path in _PUBLIC_PATHS or path.startswith("/static/"):
            return await self.app(scope, receive, send)
        if auth.verify_token(Request(scope).cookies.get(auth.COOKIE_NAME)):
            return await self.app(scope, receive, send)
        if path.startswith("/api"):
            resp = JSONResponse({"detail": "authentication required"}, status_code=401)
        else:
            resp = RedirectResponse("/login", status_code=302)
        await resp(scope, receive, send)


app.add_middleware(AuthGate)


def _set_session_cookie(resp) -> None:
    resp.set_cookie(
        auth.COOKIE_NAME, auth.make_token(),
        max_age=int(config.SESSION_DAYS * 86400), path="/",
        httponly=True, samesite="lax", secure=config.COOKIE_SECURE,
    )


class CreateJobBody(BaseModel):
    url: str


class ApproveBody(BaseModel):
    ranks: list[int]


class LoginBody(BaseModel):
    username: str
    password: str


class SettingsUpdate(BaseModel):
    # None = leave unchanged; "" = explicit clear; non-empty = set.
    scoring_provider: str | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None


@app.get("/")
def index(request: Request):
    resp = FileResponse(STATIC / "index.html")
    # Sliding expiry: each UI visit re-issues the cookie, so the session dies
    # only after SESSION_DAYS of *inactivity*.
    if auth.enabled() and auth.verify_token(request.cookies.get(auth.COOKIE_NAME)):
        _set_session_cookie(resp)
    return resp


@app.get("/login")
def login_page(request: Request):
    if not auth.enabled() or auth.verify_token(request.cookies.get(auth.COOKIE_NAME)):
        return RedirectResponse("/", status_code=302)
    return FileResponse(STATIC / "login.html")


@app.post("/api/login")
def login(body: LoginBody):
    if auth.enabled() and not auth.check_credentials(body.username, body.password):
        raise HTTPException(401, "wrong username or password")
    resp = JSONResponse({"ok": True})
    if auth.enabled():
        _set_session_cookie(resp)
    return resp


@app.post("/api/logout")
def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(auth.COOKIE_NAME, path="/")
    return resp


def _schedule_process_shutdown(delay_s: float = 0.8) -> None:
    """End the single-process Uvicorn server after its HTTP response is sent."""
    def stop_process():
        time.sleep(delay_s)
        log.warning("server stop requested from the web UI")
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(
        target=stop_process, name="clipping-tool-server-stop", daemon=True
    ).start()


def _schedule_process_restart(delay_s: float = 0.8) -> None:
    """Exit with the supervisor's restart code after the HTTP response is sent."""
    def restart_process():
        time.sleep(delay_s)
        log.warning("server restart requested from the web UI")
        os._exit(_RESTART_EXIT_CODE)

    threading.Thread(
        target=restart_process, name="clipping-tool-server-restart", daemon=True
    ).start()


@app.post("/api/server/stop", status_code=202)
def stop_server():
    """Stop Clipping Tool itself, after cancelling any active video work.

    AuthGate protects this endpoint whenever ADMIN_PASSWORD is configured.
    Before exposing Clipping Tool on a domain, authentication must be enabled.
    """
    global _shutdown_requested
    with _shutdown_lock:
        if _shutdown_requested or _restart_requested:
            return {"ok": True, "already_stopping": True, "cancelled_jobs": []}
        _shutdown_requested = True

    cancelled = jobs.cancel_active_jobs_for_shutdown()
    _schedule_process_shutdown()
    return {
        "ok": True,
        "already_stopping": False,
        "cancelled_jobs": cancelled,
    }


@app.post("/api/server/restart", status_code=202)
def restart_server():
    """Cancel active work, then replace Clipping Tool with a fresh server process."""
    global _restart_requested
    with _shutdown_lock:
        if _shutdown_requested or _restart_requested:
            return {"ok": True, "already_restarting": True, "cancelled_jobs": []}
        _restart_requested = True

    cancelled = jobs.cancel_active_jobs_for_shutdown("server restarted by user")
    _schedule_process_restart()
    return {
        "ok": True,
        "already_restarting": False,
        "cancelled_jobs": cancelled,
    }


@app.get("/api/hardware")
def hardware():
    hw = detect()
    return {
        "cpu_threads": hw.cpu_threads, "ram_gb": hw.ram_gb, "has_cuda": hw.has_cuda,
        "encoder": hw.encoder, "whisper_model": hw.whisper_model,
        "render_workers": hw.render_workers, "scoring_provider": config.SCORING_PROVIDER,
        "auth_enabled": auth.enabled(),
        "disk": diskspace.usage_summary(),
        "code_version": version.commit(),
    }


def _settings_view() -> dict:
    # Key presence only — the actual key value is never returned once saved.
    return {
        "scoring_provider": config.SCORING_PROVIDER,
        "has_openai_key": bool(config.OPENAI_API_KEY),
        "has_gemini_key": bool(config.GEMINI_API_KEY),
    }


@app.get("/api/settings")
def get_settings():
    return _settings_view()


@app.post("/api/settings")
def update_settings(body: SettingsUpdate):
    if body.scoring_provider is not None and body.scoring_provider not in ("local", "openai", "gemini"):
        raise HTTPException(400, "scoring_provider must be local, openai, or gemini")
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if updates:
        config.update_runtime_settings(**updates)
    return _settings_view()


@app.post("/api/jobs")
def create_job(body: CreateJobBody):
    url = body.url.strip()
    if not url.lower().startswith(("http://", "https://")):
        raise HTTPException(400, "not a valid URL")
    job = jobs.create_job(url)
    return {"job_id": job["id"]}


@app.get("/api/jobs")
def list_jobs():
    return jobs.list_jobs()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return {**job, "progress": jobs.read_progress(job_id)}


@app.post("/api/jobs/{job_id}/approve")
def approve(job_id: str, body: ApproveBody):
    if not jobs.get_job(job_id):
        raise HTTPException(404, "no such job")
    try:
        jobs.approve(job_id, body.ranks)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.post("/api/jobs/{job_id}/retry")
def retry(job_id: str):
    try:
        jobs.retry(job_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str):
    try:
        jobs.cancel(job_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    try:
        return jobs.delete_job(job_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/maintenance/clear-caches")
def clear_caches():
    return jobs.clear_caches()


@app.get("/api/jobs/{job_id}/events")
async def events(job_id: str):
    if not jobs.get_job(job_id):
        raise HTTPException(404, "no such job")

    async def stream():
        last = None
        while True:
            job = jobs.get_job(job_id)
            snapshot = json.dumps(
                {**job, "progress": jobs.read_progress(job_id)}, ensure_ascii=False)
            if snapshot != last:
                last = snapshot
                yield f"data: {snapshot}\n\n"
            if job["state"] in jobs.TERMINAL_STATES:
                break
            await asyncio.sleep(1.0)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


app.mount("/clips", StaticFiles(directory=str(config.OUTPUT_DIR)), name="clips")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
