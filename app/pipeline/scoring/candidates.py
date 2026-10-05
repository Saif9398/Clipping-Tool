"""Stage 3: build candidate 75-100s windows from the transcript, score them
(local heuristic prefilter -> selected LLM provider on the top slice), dedupe
overlaps, and return the ranked list shown to the user for approval.

A candidate must be a complete thought: windows start at a segment (sentence)
start and end at a segment end with a natural pause, never mid-sentence.
"""
from __future__ import annotations

import json
import logging
import wave
from pathlib import Path

import numpy as np

from app import config
from app.pipeline.scoring import get_provider
from app.pipeline.scoring.local import LocalScoringProvider

log = logging.getLogger(__name__)

SEED_STEP_S = 25.0        # try a new window start roughly every 25s of content
LLM_TOP_N = 30            # minimum LLM slice; scales up with the clip-count target
OVERLAP_IOU = 0.30
TERMINAL = (".", "!", "?", '."', '!"', '?"', ".'", "…")


def target_clip_count(duration_s: float) -> int:
    """Duration-scaled candidate target: ~CLIPS_PER_MINUTE per source minute,
    clamped to [MIN_CANDIDATES, MAX_CANDIDATES]. A cap, never a quota."""
    raw = round(duration_s / 60.0 * config.CLIPS_PER_MINUTE)
    return int(np.clip(raw, config.MIN_CANDIDATES, config.MAX_CANDIDATES))


def build_candidates(work_dir: Path) -> list[dict]:
    transcript = json.loads((work_dir / "transcript.json").read_text(encoding="utf-8"))
    segments = [s for s in transcript["segments"] if s["words"]]
    if not segments:
        return []

    duration = float(transcript.get("duration") or 0.0) or segments[-1]["end"]
    target = target_clip_count(duration)
    log.info("source %.1f min -> target %d candidates", duration / 60.0, target)

    windows = _build_windows(segments)
    log.info("built %d raw candidate windows", len(windows))
    if not windows:
        return []

    energy = _energy_scores(work_dir / "audio16k.wav", windows)
    local = LocalScoringProvider(energy)
    local_scores = local.score_candidates(windows)

    # Pre-rank with the heuristic; only the top slice goes to the (paid) LLM.
    # The slice must comfortably exceed the target or big videos would rank
    # mostly local-scored candidates.
    ranked = sorted(windows, key=lambda c: local_scores[c["id"]]["score"], reverse=True)
    top = ranked[:max(LLM_TOP_N, int(np.ceil(1.5 * target)))]

    provider = get_provider(energy)
    scores = dict(local_scores)
    if provider.name != "local":
        try:
            llm_scores = provider.score_candidates(top)
            scores.update(llm_scores)
        except Exception as e:
            log.warning("LLM scoring failed (%s) — using local heuristic scores", e)

    for c in windows:
        c.update(scores[c["id"]])
        c["provider"] = provider.name if c["id"] in scores and provider.name != "local" else "local"

    kept = _dedupe(sorted(windows, key=lambda c: (-c["score"], c["start"])))
    kept = kept[:target]
    # Drop genuinely weak tails but never below a usable handful
    strong = [c for c in kept if c["score"] >= 35]
    result = strong if len(strong) >= min(len(kept), config.MIN_CANDIDATES) else kept
    for rank, c in enumerate(result, 1):
        c["rank"] = rank
    (work_dir / "candidates.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _build_windows(segments: list[dict]) -> list[dict]:
    windows, cid = [], 0
    seeds, last_seed_t = [], -1e9
    for i, seg in enumerate(segments):
        if seg["start"] - last_seed_t >= SEED_STEP_S:
            seeds.append(i)
            last_seed_t = seg["start"]

    for i in seeds:
        start = segments[i]["start"]
        best_end, fallback_end = None, None
        for j in range(i, len(segments)):
            end = segments[j]["end"]
            dur = end - start
            if dur > config.CLIP_MAX_S:
                break
            if dur >= config.CLIP_MIN_S:
                fallback_end = j
                gap = (segments[j + 1]["start"] - end) if j + 1 < len(segments) else 999
                if segments[j]["text"].rstrip().endswith(TERMINAL) and gap >= 0.3:
                    best_end = j  # keep extending: latest natural end inside the window wins
        j = best_end if best_end is not None else fallback_end
        if j is None:
            continue
        chunk = segments[i:j + 1]
        windows.append({
            "id": cid,
            "start": round(start, 3),
            "end": round(chunk[-1]["end"], 3),
            "text": " ".join(s["text"] for s in chunk),
        })
        cid += 1
    return windows


def _energy_scores(wav_path: Path, windows: list[dict]) -> dict[int, float]:
    """0..1 score for audio dynamics (peaks over baseline = laughter/excitement proxy)."""
    try:
        with wave.open(str(wav_path), "rb") as w:
            sr = w.getframerate()
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        data = data.astype(np.float32) / 32768.0
        hop = sr // 10  # 100ms RMS windows
        n = len(data) // hop
        rms = np.sqrt(np.mean(data[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-9)
        base = np.percentile(rms[rms > 0.001], 50) if np.any(rms > 0.001) else 0.01
        out = {}
        for c in windows:
            a, b = int(c["start"] * 10), min(n, int(c["end"] * 10))
            if b <= a:
                out[c["id"]] = 0.3
                continue
            seg = rms[a:b]
            peak_ratio = float(np.percentile(seg, 95) / (base + 1e-9))
            out[c["id"]] = float(np.clip((peak_ratio - 1.0) / 3.0, 0.0, 1.0))
        return out
    except Exception as e:
        log.warning("energy analysis failed (%s) — neutral energy scores", e)
        return {c["id"]: 0.3 for c in windows}


def _dedupe(ranked: list[dict]) -> list[dict]:
    kept = []
    for c in ranked:
        ok = True
        for k in kept:
            inter = max(0.0, min(c["end"], k["end"]) - max(c["start"], k["start"]))
            union = (c["end"] - c["start"]) + (k["end"] - k["start"]) - inter
            if union > 0 and inter / union > OVERLAP_IOU:
                ok = False
                break
        if ok:
            kept.append(c)
    return kept
