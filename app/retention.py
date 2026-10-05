"""Config-driven disk retention: prune old work/ caches and output/ clip dirs.

Runs at server startup (app.main) and manually via `python -m app.retention`.
Safety rule: a work/<job> dir is only deleted when its job.json says the job is
in a terminal state (or job.json is missing/unreadable = orphaned dir) — a job
that is running or awaiting approval is never touched, however old.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

from app import config
from app.jobs import TERMINAL_STATES

log = logging.getLogger(__name__)


def _newest_mtime(d: Path) -> float:
    """Newest mtime among the dir and its direct entries (children's mtimes
    bump when files inside them change, so a shallow scan tracks activity)."""
    newest = 0.0
    try:
        newest = d.stat().st_mtime
        for p in d.iterdir():
            newest = max(newest, p.stat().st_mtime)
    except OSError:
        pass
    return newest


def _work_dir_deletable(d: Path) -> bool:
    """Single deletability rule shared by age-based retention and disk-pressure
    emergency cleanup (app.diskspace) so the two can never disagree.

    A live in-memory job in a non-terminal state is NEVER deletable, whatever
    the disk says — this closes the window where a user clicks retry while a
    cleanup pass is scanning. On-disk states: terminal = deletable; unreadable
    job.json = orphaned, deletable; awaiting_approval = NEVER deletable by
    either cleanup path (idleness is its normal condition — a human hasn't
    decided yet, and deleting it would silently discard their pending work); other
    non-terminal (active pipeline) states = deletable only when the dir has had
    no file activity for STALE_AFTER_S — an active state with a silent dir
    means the process crashed, and would otherwise pin the multi-GB source
    video forever."""
    from app import jobs as jobs_mod

    mem = jobs_mod._jobs.get(d.name)
    if mem is not None and mem.get("state") not in TERMINAL_STATES:
        return False
    jf = d / "job.json"
    try:
        state = json.loads(jf.read_text(encoding="utf-8")).get("state")
    except (OSError, json.JSONDecodeError):
        return True  # orphaned dir (no readable job.json) — age alone decides
    if state in TERMINAL_STATES:
        return True
    if state == "awaiting_approval":
        return False
    return _newest_mtime(d) < time.time() - jobs_mod.STALE_AFTER_S


def cleanup_old(days: int, work_dir: Path | None = None,
                output_dir: Path | None = None) -> list[Path]:
    """Delete work/<job> and output/<slug> dirs older than `days`. Returns the
    deleted paths. days <= 0 disables cleanup entirely."""
    if days <= 0:
        return []
    work_dir = work_dir if work_dir is not None else config.WORK_DIR
    output_dir = output_dir if output_dir is not None else config.OUTPUT_DIR
    cutoff = time.time() - days * 86400
    deleted: list[Path] = []

    for d in sorted(work_dir.iterdir() if work_dir.is_dir() else []):
        if d.is_dir() and _newest_mtime(d) < cutoff and _work_dir_deletable(d):
            shutil.rmtree(d, ignore_errors=True)
            deleted.append(d)
            log.info("retention: deleted work cache %s (> %d days old)", d.name, days)

    for d in sorted(output_dir.iterdir() if output_dir.is_dir() else []):
        if d.is_dir() and _newest_mtime(d) < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            deleted.append(d)
            log.info("retention: deleted output dir %s (> %d days old)", d.name, days)

    return deleted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    gone = cleanup_old(config.RETENTION_DAYS)
    if config.RETENTION_DAYS <= 0:
        print("RETENTION_DAYS=0 — retention disabled, nothing deleted")
    else:
        print(f"deleted {len(gone)} dir(s) older than {config.RETENTION_DAYS} days")
        for p in gone:
            print(f"  {p}")
