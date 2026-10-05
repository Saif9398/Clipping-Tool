"""Local heuristic scoring — the zero-dependency, zero-key fallback provider.

Blends lexical virality signals with audio-energy dynamics. Weaker judgment than
an LLM but deterministic and free; it also pre-ranks candidates so LLM providers
only pay for the plausible top slice.
"""
from __future__ import annotations

import re

from app.pipeline.scoring.base import ScoringProvider

_HOOK_PATTERNS = [
    r"\bnobody (tells|talks|knows)", r"\bthe (biggest|worst|best) (mistake|thing|lie)",
    r"\bhere'?s (the thing|what|why)", r"\bthe truth (is|about)", r"\bi('| a)m going to tell you",
    r"^(why|how|what) ", r"\byou (need|have) to", r"\bstop (doing|saying|believing)",
    r"\bnever\b", r"\bno one\b", r"\bsecret\b", r"\bmost people (don'?t|think|believe)",
    r"\b\$?\d[\d,]*(k| thousand| million| billion| percent|%)", r"\bchanged my life\b",
    r"\bi was wrong\b", r"\bunpopular opinion\b", r"\blisten\b[,.]",
]
_EMOTION_WORDS = (
    "amazing insane crazy unbelievable shocking terrifying heartbreaking hilarious "
    "furious love hate scared cried laughing obsessed devastating incredible wild "
    "blown embarrassing painful proud regret afraid angry"
).split()
_OPINION_MARKERS = (
    "honestly frankly overrated underrated wrong garbage nonsense bullshit scam "
    "myth lie completely totally absolutely everyone nobody worst best ever"
).split()
_STORY_MARKERS = (
    "one day so i and then suddenly at that moment long story remember when told me "
    "walked in turned out ended up realized"
).split()
_PRACTICAL_MARKERS = (
    "how to you should step tip trick strategy framework rule habit start stop "
    "instead here's what do this avoid"
).split()


class LocalScoringProvider(ScoringProvider):
    name = "local"

    def __init__(self, energy_stats: dict[int, float] | None = None):
        # candidate id -> 0..1 audio-energy-dynamics score (see candidates.py)
        self.energy_stats = energy_stats or {}

    def score_candidates(self, candidates: list[dict]) -> dict[int, dict]:
        return {c["id"]: self._score_one(c) for c in candidates}

    def _score_one(self, c: dict) -> dict:
        text = c["text"]
        lower = text.lower()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        first_two = " ".join(sentences[:2]).lower()

        hook = sum(1 for p in _HOOK_PATTERNS if re.search(p, first_two))
        hook_score = min(1.0, hook / 2.0)

        words = lower.split()
        n = max(1, len(words))
        emotion = min(1.0, sum(words.count(w) for w in _EMOTION_WORDS) / (n / 60))
        opinion = min(1.0, sum(words.count(w) for w in _OPINION_MARKERS) / (n / 80))
        story = min(1.0, sum(lower.count(m) for m in _STORY_MARKERS) / 3.0)
        practical = min(1.0, sum(lower.count(m) for m in _PRACTICAL_MARKERS) / 3.0)
        questions = min(1.0, text.count("?") / 3.0)
        # quotability: presence of short punchy sentences
        short_punchy = sum(1 for s in sentences if 4 <= len(s.split()) <= 12)
        quotable = min(1.0, short_punchy / 4.0)
        energy = self.energy_stats.get(c["id"], 0.3)

        score = 100 * (
            0.30 * hook_score + 0.14 * emotion + 0.13 * opinion + 0.10 * story
            + 0.09 * practical + 0.06 * questions + 0.10 * quotable + 0.08 * energy
        )
        # floor/jitter-free tie-breaking is handled by ranking on (score, start)
        axes = {
            "hook": hook_score, "emotion": emotion, "opinion": opinion, "story": story,
            "practical": practical, "quotable": quotable, "energy": energy,
        }
        top = sorted(axes, key=axes.get, reverse=True)[:2]
        hook_line = sentences[0] if sentences else text[:90]
        return {
            "score": round(min(99.0, score), 1),
            "title": _make_title(hook_line),
            "hook": hook_line[:140],
            "reason": f"strong {top[0]} + {top[1]} signals (heuristic)",
        }


def _make_title(hook_line: str) -> str:
    t = re.sub(r"^\W+|\W+$", "", hook_line)
    words = t.split()
    if len(words) > 9:
        t = " ".join(words[:9]) + "…"
    return t.capitalize() if t else "Untitled moment"
