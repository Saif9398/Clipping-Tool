"""Shared prompt construction and response parsing for LLM scoring providers."""
from __future__ import annotations

import json
import re

from app.pipeline.scoring.base import RUBRIC, clamp_score

BATCH_SIZE = 10

SYSTEM_PROMPT = (
    "You are a short-form video editor who has cut thousands of viral TikTok/Shorts "
    "clips from podcasts. You judge clips ruthlessly on data, not politeness.\n\n" + RUBRIC
)


def build_user_prompt(batch: list[dict]) -> str:
    lines = [
        "Score these candidate clips (each is a transcript excerpt of a 75-100s clip).",
        'Reply with ONLY a JSON array, one object per candidate:',
        '[{"id": <int>, "score": <0-100>, "title": "<catchy 4-9 word title>", '
        '"hook": "<the exact opening line, may be lightly trimmed>", '
        '"reason": "<one sentence: why this would or would not go viral>"}]',
        "",
    ]
    for c in batch:
        lines.append(f'--- CANDIDATE id={c["id"]} ({c["end"] - c["start"]:.0f}s) ---')
        lines.append(c["text"].strip())
        lines.append("")
    return "\n".join(lines)


def parse_response(raw: str, batch: list[dict]) -> dict[int, dict]:
    """Tolerant JSON extraction: handles code fences and stray prose around the array."""
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON array in LLM response: {raw[:200]!r}")
    items = json.loads(m.group(0))
    valid_ids = {c["id"] for c in batch}
    out = {}
    for it in items:
        try:
            cid = int(it["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if cid not in valid_ids:
            continue
        out[cid] = {
            "score": clamp_score(it.get("score")),
            "title": str(it.get("title") or "")[:80] or "Untitled moment",
            "hook": str(it.get("hook") or "")[:140],
            "reason": str(it.get("reason") or "")[:200],
        }
    if not out:
        raise ValueError("LLM response parsed but contained no valid candidate ids")
    return out


def batches(candidates: list[dict]) -> list[list[dict]]:
    return [candidates[i:i + BATCH_SIZE] for i in range(0, len(candidates), BATCH_SIZE)]
