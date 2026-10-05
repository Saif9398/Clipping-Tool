"""Vision model files (mediapipe Tasks + YuNet): paths + lazy download so a
fresh server setup self-heals.

Downloads must be safe under concurrent callers: Stage 2's faces task and
Stage 5's render workers (up to render_workers processes) can all call
model_path() for the same missing file at once. Each call downloads to a
unique per-process temp name and atomically replaces the target — no two
processes ever share a temp file, so there is no rename race. `ensure_models()`
is the belt-and-suspenders fix: call it once, synchronously, before any
parallel work starts, so in the common case no worker ever hits the download
path at all.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
import urllib.request
from pathlib import Path

from app import config

log = logging.getLogger(__name__)

_MODELS = {
    "blaze_face_short_range.tflite": (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
    # small-face detector (wide two-shots are below BlazeFace's floor)
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
}


def model_path(name: str) -> Path:
    path = config.MODELS_DIR / name
    if not path.exists():
        _download(name, path)
    # Confirm the file is actually openable before handing the path back: on
    # Windows, a concurrent caller's os.replace onto this same destination can
    # leave it transiently un-openable for a few milliseconds even after
    # path.exists() is true. Callers (mediapipe, this module's own tests)
    # should never observe that window.
    _wait_until_readable(path)
    return path


def _download(name: str, path: Path) -> None:
    # Unique-per-call temp name (tempfile's own uniqueness) in the same dir as
    # the target, so os.replace is an atomic same-filesystem rename. Concurrent
    # callers each get their own temp file — no shared-temp race, and
    # whichever finishes first "wins" the replace; the rest just overwrite it
    # with identical content, which is harmless.
    fd, tmp_name = tempfile.mkstemp(
        dir=str(config.MODELS_DIR), prefix=f".{name}.", suffix=".part")
    tmp = Path(tmp_name)
    try:
        os.close(fd)
        urllib.request.urlretrieve(_MODELS[name], tmp)
        _atomic_replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)  # no-op once _atomic_replace has moved it


def _atomic_replace(tmp: Path, path: Path) -> None:
    """os.replace is atomic, but on Windows two concurrent replaces onto the
    SAME destination can raise a transient sharing-violation PermissionError
    even though the race is harmless (both sides have identical content).
    Retry briefly; if the destination exists by the time we give up, some
    caller's replace won and we're done regardless of whether it was ours."""
    last_exc: Exception | None = None
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:
            last_exc = e
            if path.exists():
                return
            time.sleep(0.05 * (attempt + 1))
    if path.exists():
        return
    raise last_exc


def _wait_until_readable(path: Path, attempts: int = 15, base_delay: float = 0.03) -> None:
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            with open(path, "rb"):
                return
        except OSError as e:
            last_exc = e
            time.sleep(base_delay * (attempt + 1))
    raise last_exc


def ensure_models() -> None:
    """Download every required model file up front. Call once, synchronously,
    before any parallel render/analysis workers spawn (server startup, and
    defensively at the top of each job) so workers never race on a missing
    model file."""
    for name in _MODELS:
        if not (config.MODELS_DIR / name).exists():
            log.info("downloading model %s (first run on this machine)", name)
        model_path(name)
