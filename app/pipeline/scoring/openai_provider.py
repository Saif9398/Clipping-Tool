"""OpenAI scoring provider. Needs OPENAI_API_KEY in .env."""
from __future__ import annotations

import logging

from app import config
from app.pipeline.scoring import llm_shared
from app.pipeline.scoring.base import ScoringProvider

log = logging.getLogger(__name__)


class OpenAIScoringProvider(ScoringProvider):
    name = "openai"

    def __init__(self):
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        from openai import OpenAI

        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    def score_candidates(self, candidates: list[dict]) -> dict[int, dict]:
        results: dict[int, dict] = {}
        for batch in llm_shared.batches(candidates):
            resp = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": llm_shared.SYSTEM_PROMPT},
                    {"role": "user", "content": llm_shared.build_user_prompt(batch)},
                ],
            )
            results.update(llm_shared.parse_response(resp.choices[0].message.content, batch))
        log.info("openai scored %d/%d candidates", len(results), len(candidates))
        return results
