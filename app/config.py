"""Central configuration. Everything machine-specific lives in .env — no hardcoded absolute paths."""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    v = os.getenv(name, "").strip()
    return v if v else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, ""))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except ValueError:
        return default


# Directories (relative to project root unless overridden)
WORK_DIR = Path(_env("WORK_DIR", str(PROJECT_ROOT / "work")))
OUTPUT_DIR = Path(_env("OUTPUT_DIR", str(PROJECT_ROOT / "output")))
MODELS_DIR = Path(_env("MODELS_DIR", str(PROJECT_ROOT / "models")))

# Server
HOST = _env("HOST", "127.0.0.1")
PORT = _env_int("PORT", 8000)

# Scoring
SCORING_PROVIDER = _env("SCORING_PROVIDER", "local").lower()  # local | openai | gemini
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = _env("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-2.0-flash")

# Whisper ("auto" lets hardware.py decide)
WHISPER_MODEL = _env("WHISPER_MODEL", "auto")
WHISPER_COMPUTE = _env("WHISPER_COMPUTE", "auto")

# Clip constraints (seconds)
CLIP_MIN_S = 75.0
CLIP_MAX_S = 100.0
# Candidate count scales with source duration: ~CLIPS_PER_MINUTE per minute
# (0.4 -> 30min:12, 70min:28, 120min:48), clamped to [MIN, MAX]_CANDIDATES.
# This is a CAP on the ranked list, never a quota — weak material yields fewer.
CLIPS_PER_MINUTE = _env_float("CLIPS_PER_MINUTE", 0.4)
MIN_CANDIDATES = _env_int("MIN_CANDIDATES", 4)   # floor for very short videos
MAX_CANDIDATES = _env_int("MAX_CANDIDATES", 60)  # absolute sanity ceiling

# Output video
OUT_W = 1080
OUT_H = 1920

# Constant-quality target for ALL encoders (qsv -global_quality, x264 -crf,
# nvenc -cq). Lower = sharper + bigger files. 17 ~ visually transparent for
# talking-head content; picked by A/B against 16 and 20 on real footage.
VIDEO_QUALITY = _env_int("VIDEO_QUALITY", 17)

# Framing: multiplies the face-fraction targets in camera.py, which computes
#   crop_h = face_h / (face_frac * ZOOM_FACTOR)
# so crop size is INVERSE to this knob: lower = larger source crop = looser
# framing AND LESS upscaling to 1920 = sharper. (0.9 samples ~11% more source
# pixels per output pixel than 1.0; unsharp/cas cannot recover detail that was
# never sampled, which is why a 1.0 render can look softer at a higher bitrate.)
# Split-crop overlap is NOT this knob's job — camera.split_rects_iou +
# speaker.SPLIT_MAX_CROP_IOU gate that independently.
ZOOM_FACTOR = _env_float("ZOOM_FACTOR", 0.9)

# Hard ceiling on how far a crop may be magnified to fill the output frame
# (out_h / crop_h). Face framing targets a constant face FRACTION, which on a
# distant face demands a crop far smaller than the source can carry — measured
# 5x-9.6x on real 1080p footage, i.e. irrecoverable mush. Above this cap the
# crop stops shrinking, so the face is framed smaller instead of blurrier.
# Raise for tighter punch-ins at the cost of sharpness; lower for the reverse.
MAX_UPSCALE = _env_float("MAX_UPSCALE", 4.0)

# "4K Pro" look filter strength: 1.0 = full (sharpen+contrast+vibrance), 0 = off.
LOOK_FILTER = _env_float("LOOK_FILTER", 1.0)
# Expert escape hatch: a raw ffmpeg filter string here replaces the generated chain.
LOOK_FILTER_CHAIN = _env("LOOK_FILTER_CHAIN", "")

# Download
MAX_SOURCE_HEIGHT = _env_int("MAX_SOURCE_HEIGHT", 1080)

# Parallelism ("0" = auto from hardware.py)
RENDER_WORKERS = _env_int("RENDER_WORKERS", 0)

# Stage-2 analysis workers (transcribe / scenes / faces run as separate
# processes). "0" = auto: 3 normally, but 2 under 10 GB RAM so transcribe runs
# in parallel with at most ONE video-decoding task at a time (scenes and faces
# both decode every frame — two at once thrashes a low-RAM box). An explicit
# value overrides. NOTE: the real guard against heavy-source hangs is the
# hardware source-height cap (a CPU box never downloads 4K), not this knob.
ANALYSIS_WORKERS = _env_int("ANALYSIS_WORKERS", 0)

# Hard cap on concurrent h264_nvenc encode sessions. Most NVIDIA GPUs (consumer
# cards especially, without the nvidia-patch driver mod) refuse to open more
# than a handful of simultaneous NVENC sessions; exceeding it doesn't queue,
# it fails the encoder open outright. hardware.py's single-instance smoke test
# can't detect this, so render_workers is clamped to this value whenever nvenc
# is selected. Raise it only after confirming the real limit on the target GPU.
NVENC_MAX_SESSIONS = _env_int("NVENC_MAX_SESSIONS", 3)

# How many jobs may run their heavy pipeline (analysis or render) at once.
# Extra submissions wait in "queued"/"render_queued" until a slot frees.
MAX_CONCURRENT_JOBS = _env_int("MAX_CONCURRENT_JOBS", 1)

# Wall-clock ceilings so a wedged worker can never hang a stage forever — it is
# marked failed with a clear timeout error instead. Per-CLIP for render, per
# whole Stage-2 for analysis. Generous by default (CPU-only transcription is
# slow); raise if legitimate long jobs trip them. Applied as a total budget
# scaled by how many render batches the worker count implies.
RENDER_CLIP_TIMEOUT_S = _env_int("RENDER_CLIP_TIMEOUT_S", 1200)   # 20 min / clip
ANALYSIS_TIMEOUT_S = _env_int("ANALYSIS_TIMEOUT_S", 3600)         # 60 min / Stage 2

# Disk-space guardrails. Below WARN_FREE_GB: log a warning. Below MIN_FREE_GB:
# emergency-delete the oldest finished/interrupted work/<job> caches (never
# output/ clips, never running or awaiting-approval jobs) and, if still below,
# fail the job up front with a clear message instead of a cryptic mid-render
# ffmpeg death. MIN_FREE_GB=0 disables auto-deletion (warn/fail only).
MIN_FREE_GB = _env_float("MIN_FREE_GB", 5.0)
WARN_FREE_GB = _env_float("WARN_FREE_GB", 10.0)

# Auto-delete work/<job> caches (terminal jobs only) and output/<slug> dirs
# older than this many days. 0 = keep forever. Runs at server startup and via
# `python -m app.retention`.
RETENTION_DAYS = _env_int("RETENTION_DAYS", 0)

# Admin login gate (app/auth.py). Auth is DISABLED until a password is set —
# local dev needs no login; the deployed server sets one. SHA256 variant (hex
# digest of the password) wins over plaintext if both are set.
ADMIN_USERNAME = _env("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "").strip()
ADMIN_PASSWORD_SHA256 = os.getenv("ADMIN_PASSWORD_SHA256", "").strip()
SESSION_DAYS = _env_float("SESSION_DAYS", 7.0)
SESSION_SECRET = os.getenv("SESSION_SECRET", "").strip()  # unset = auto (work/.session_secret)
COOKIE_SECURE = _env_int("COOKIE_SECURE", 0) == 1  # set 1 when serving over HTTPS

for _d in (WORK_DIR, OUTPUT_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---- Runtime-editable settings (Settings UI in app/main.py) ----
# A JSON overlay in work/ (gitignored) lets scoring_provider / API keys be
# changed from the UI and take effect immediately for the next job — no
# restart. Only keys present in the file override .env; anything never
# touched via the UI keeps following .env as before.

def _settings_file() -> Path:
    return WORK_DIR / "settings.json"


def _load_runtime_settings() -> None:
    global SCORING_PROVIDER, OPENAI_API_KEY, GEMINI_API_KEY
    try:
        data = json.loads(_settings_file().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if "scoring_provider" in data:
        SCORING_PROVIDER = data["scoring_provider"]
    if "openai_api_key" in data:
        OPENAI_API_KEY = data["openai_api_key"]
    if "gemini_api_key" in data:
        GEMINI_API_KEY = data["gemini_api_key"]


def update_runtime_settings(**kwargs: str) -> None:
    """Persist the given keys (scoring_provider/openai_api_key/gemini_api_key)
    and apply them in-memory immediately. "" explicitly clears a key (wins
    over .env); keys not passed are left untouched."""
    try:
        data = json.loads(_settings_file().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    data.update(kwargs)
    _settings_file().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    _load_runtime_settings()


_load_runtime_settings()
