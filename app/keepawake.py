"""Prevent Windows sleep from pausing long pipeline runs (no-op elsewhere).

Wrap long-running work in `with keep_awake():` — allows the display to sleep
but keeps the system awake until the block exits.
"""
from __future__ import annotations

import contextlib
import logging
import os

log = logging.getLogger(__name__)

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


@contextlib.contextmanager
def keep_awake():
    set_ok = False
    if os.name == "nt":
        try:
            import ctypes

            set_ok = bool(ctypes.windll.kernel32.SetThreadExecutionState(
                _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED))
        except Exception as e:
            log.warning("keep-awake unavailable: %s", e)
    try:
        yield
    finally:
        if set_ok:
            import ctypes

            ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
