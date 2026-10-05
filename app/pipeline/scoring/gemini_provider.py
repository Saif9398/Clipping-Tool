"""Gemini scoring provider (free tier friendly). Needs GEMINI_API_KEY in .env."""
from __future__ import annotations

import logging

from app import config
from app.pipeline.scoring import llm_shared
from app.pipeline.scoring.base import ScoringProvider

log = logging.getLogger(__name__)


class GeminiScoringProvider(ScoringProvider):
    name = "gemini"

    def __init__(self):
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set")
        from google import genai

        self.client = genai.Client(api_key=config.GEMINI_API_KEY)

    def score_candidates(self, candidates: list[dict]) -> dict[int, dict]:
        from google.genai import types

        results: dict[int, dict] = {}
        for batch in llm_shared.batches(candidates):
            resp = self.client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=llm_shared.build_user_prompt(batch),
                config=types.GenerateContentConfig(
                    system_instruction=llm_shared.SYSTEM_PROMPT,
                    temperature=0.3,
                    response_mime_type="application/json",
                ),
            )
            results.update(llm_shared.parse_response(resp.text, batch))
        log.info("gemini scored %d/%d candidates", len(results), len(candidates))
        return results
