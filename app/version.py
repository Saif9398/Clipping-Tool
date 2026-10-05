"""Identify the running code version for local diagnostics and output manifests.
Restart the server after source changes so running jobs use the current code.
"""
from __future__ import annotations

import functools
import os
import subprocess

from app import config


@functools.lru_cache(maxsize=1)
def commit() -> str:
    baked = os.getenv("CLIPPING_TOOL_VERSION", "").strip()
    if baked:
        return baked[:40]
    try:
        r = subprocess.run(
            ["git", "-C", str(config.PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"
