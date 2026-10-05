"""Scoring provider interface. All providers consume the same candidate dicts and
return the same result schema, so ranking/dedupe downstream never cares which
provider produced a score.

Candidate in:  {"id": int, "start": float, "end": float, "text": str}
Result out:    {id: {"score": 0-100 float, "title": str, "hook": str, "reason": str}}
"""
from __future__ import annotations

from abc import ABC, abstractmethod

RUBRIC = """Score each candidate clip 0-100 for its potential to go viral as a
TikTok/Shorts vertical clip, judging against these axes:
- HOOK (most important): do the first 1-2 sentences instantly grab attention —
  a bold claim, surprising fact, open question, or high-stakes setup?
- EMOTIONAL PEAK: laughter, anger, awe, vulnerability, excitement.
- CONTROVERSY / STRONG OPINION: a take people will argue with in the comments.
- REVELATION: insider knowledge, a secret, a myth busted, "nobody tells you this".
- CONFLICT / TENSION: disagreement, debate, a challenge, stakes.
- QUOTABILITY: a line people would put in a caption or repeat.
- STORY ARC: a complete mini-story with setup and payoff inside the clip.
- PRACTICAL VALUE: concrete advice the viewer can use immediately.

A clip scoring 80+ must have a killer hook AND at least two other strong axes.
50-79 = solid but not exceptional. Below 50 = filler, meandering, or context-dependent.
Penalize clips that start mid-thought, reference earlier context the viewer lacks,
or take more than ~10 seconds to get interesting."""


class ScoringProvider(ABC):
    name: str = "base"

    @abstractmethod
    def score_candidates(self, candidates: list[dict]) -> dict[int, dict]:
        """May raise — callers handle fallback to the local provider."""


def clamp_score(v) -> float:
    try:
        return max(0.0, min(100.0, float(v)))
    except (TypeError, ValueError):
        return 0.0
