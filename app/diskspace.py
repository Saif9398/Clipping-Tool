"""Disk-space guardrails: monitoring, pre-flight checks, emergency cleanup.

Why this exists: on a small server disk (small container volumes especially), every
job accumulates a multi-GB work/<job>/source.mp4 with RETENTION_DAYS=0 keeping
it forever. When the disk fills, ffmpeg dies mid-encode with errors that look
nothing like "disk full". These checks fail fast with an explicit message
instead, and — policy chosen by the operator — reclaim space by deleting the
oldest FINISHED job caches first (never output/ clips, never running or
awaiting-approval jobs; see retention._work_dir_deletable, the single shared
deletability rule).

Interaction with age-based retention: the two compose. RETENTION_DAYS handles
routine pruning at startup; this module acts only under disk pressure, so
RETENTION_DAYS=0 now means "keep forever unless the disk runs low".
MIN_FREE_GB=0 disables auto-deletion (checks then warn/fail only).
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path

from app import config

log = logging.getLogger(__name__)

_cleanup_lock = threading.Lock()  # one emergency pass at a time (MAX_CONCURRENT_JOBS>1)


class DiskSpaceError(RuntimeError):
    pass


def free_gb(path: Path) -> float:
    """Free space (GB) on the volume holding `path`. disk_usage raises on a
    nonexistent path on both Windows and Linux, and WORK_DIR is created lazily,
    so make sure the directory exists first."""
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1024**3


def dir_size_gb(path: Path) -> float:
    """Recursive size in GB. follow_symlinks=False everywhere (container
    /workspace symlink setups can loop); unreadable entries are skipped
    (Windows ACLs) rather than failing the whole scan."""
    return _dir_size_bytes(path) / 1024**3


def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        with os.scandir(path) as it:
            for entry in it:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        total += _dir_size_bytes(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                except OSError:
                    continue
    except OSError:
        pass
    return total


def usage_summary() -> dict:
    """Per-volume free space + per-directory sizes. WORK_DIR and OUTPUT_DIR are
    measured on their OWN volumes — in a container they may be different mounts."""
    return {
        "free_gb_work_volume": round(free_gb(config.WORK_DIR), 1),
        "free_gb_output_volume": round(free_gb(config.OUTPUT_DIR), 1),
        "work_dir_gb": round(dir_size_gb(config.WORK_DIR), 2),
        "output_dir_gb": round(dir_size_gb(config.OUTPUT_DIR), 2),
        "models_dir_gb": round(dir_size_gb(config.MODELS_DIR), 2),
        "min_free_gb": config.MIN_FREE_GB,
        "warn_free_gb": config.WARN_FREE_GB,
    }


def check(context: str = "job") -> None:
    """Pre-flight gate called before Stage 1 (download) and Stage 5 (render).
    Below WARN_FREE_GB: log a warning. Below MIN_FREE_GB: emergency-clean old
    job caches; if STILL below, raise DiskSpaceError so the job fails fast
    with an actionable message instead of a mid-encode ffmpeg death."""
    if config.MIN_FREE_GB <= 0 and config.WARN_FREE_GB <= 0:
        return
    free = free_gb(config.WORK_DIR)
    if config.MIN_FREE_GB > 0 and free < config.MIN_FREE_GB:
        freed = emergency_cleanup()
        free = free_gb(config.WORK_DIR)
        if free < config.MIN_FREE_GB:
            raise DiskSpaceError(
                f"only {free:.1f} GB free on the work volume (minimum "
                f"{config.MIN_FREE_GB:.0f} GB) and emergency cleanup could only "
                f"delete {len(freed)} old job cache(s) — free disk space, lower "
                f"RETENTION_DAYS, or delete old dirs under {config.WORK_DIR} "
                f"before retrying ({context})")
        log.warning("disk pressure: emergency cleanup deleted %d old job cache(s), "
                    "%.1f GB now free", len(freed), free)
    if config.WARN_FREE_GB > 0 and free < config.WARN_FREE_GB:
        log.warning("low disk space: %.1f GB free on the work volume "
                    "(warn threshold %.0f GB) — %s may fail if the disk fills",
                    free, config.WARN_FREE_GB, context)


def emergency_cleanup() -> list[Path]:
    """Delete oldest deletable work/<job> caches until free space recovers.
    Never touches OUTPUT_DIR (finished clips are the product). Deletability is
    retention's shared predicate: terminal-state, orphaned, or crash-stale dirs
    only — running/awaiting-approval jobs are never candidates."""
    from app import retention

    deleted: list[Path] = []
    if config.MIN_FREE_GB <= 0:
        return deleted
    with _cleanup_lock:
        if not config.WORK_DIR.is_dir():
            return deleted
        candidates = sorted(
            (d for d in config.WORK_DIR.iterdir() if d.is_dir()),
            key=retention._newest_mtime,
        )
        for d in candidates:
            if free_gb(config.WORK_DIR) >= config.MIN_FREE_GB:
                break
            if not retention._work_dir_deletable(d):
                continue
            shutil.rmtree(d, ignore_errors=True)
            deleted.append(d)
            log.warning("emergency cleanup: deleted old job cache %s", d.name)
    return deleted


def sweep_orphan_parts(models_dir: Path | None = None) -> list[Path]:
    """Remove leftover .part temp files in models/ from hard kills (OOM etc.).
    The mkstemp+os.replace download is atomic, so final model files can never
    be truncated — only these temps can orphan, and they eat disk silently."""
    models_dir = models_dir if models_dir is not None else config.MODELS_DIR
    removed = []
    if models_dir.is_dir():
        for p in models_dir.glob("*.part"):
            try:
                p.unlink()
                removed.append(p)
                log.info("removed orphaned partial download %s", p.name)
            except OSError:
                pass
    return removed
