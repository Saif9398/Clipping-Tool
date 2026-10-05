"""Provider factory with automatic fallback to the local heuristic.

SCORING_PROVIDER=local|openai|gemini in .env selects the provider; a missing key
or a failed API call never blocks the pipeline — it falls back to `local` with a
logged warning. Adding a provider = one new subclass file + one entry here.
"""
from __future__ import annotations

import logging

from app import config
from app.pipeline.scoring.base import ScoringProvider
from app.pipeline.scoring.local import LocalScoringProvider

log = logging.getLogger(__name__)


def get_provider(energy_stats: dict[int, float] | None = None) -> ScoringProvider:
    name = config.SCORING_PROVIDER
    try:
        if name == "openai":
            from app.pipeline.scoring.openai_provider import OpenAIScoringProvider

            return OpenAIScoringProvider()
        if name == "gemini":
            from app.pipeline.scoring.gemini_provider import GeminiScoringProvider

            return GeminiScoringProvider()
        if name != "local":
            log.warning("unknown SCORING_PROVIDER=%r, using local heuristic", name)
    except Exception as e:
        log.warning("scoring provider %r unavailable (%s) — falling back to local heuristic", name, e)
    return LocalScoringProvider(energy_stats)
