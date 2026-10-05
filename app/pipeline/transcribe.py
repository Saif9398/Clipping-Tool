"""Stage 2a: faster-whisper transcription with word-level timestamps.

Runs in its own process (see jobs.py). Output: work_dir/transcript.json
  {"language": str, "segments": [{"start", "end", "text", "words": [{"w", "s", "e"}]}]}
"""
from __future__ import annotations

import json
from pathlib import Path


def transcribe(work_dir: Path, model_size: str, compute_type: str, cpu_threads: int,
               device: str = "cpu", progress_cb=None) -> dict:
    from faster_whisper import WhisperModel

    from app import config

    wav = work_dir / "audio16k.wav"
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        download_root=str(config.MODELS_DIR),
    )
    segments_iter, info = model.transcribe(
        str(wav),
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        beam_size=1,  # greedy: ~2x faster on CPU, negligible quality loss for scoring/captions
        condition_on_previous_text=False,  # avoids hallucination loops on long podcasts
    )

    segments = []
    for seg in segments_iter:
        words = [
            {"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3)}
            for w in (seg.words or []) if w.word.strip()
        ]
        segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "words": words,
        })
        if progress_cb and info.duration:
            progress_cb(min(1.0, seg.end / info.duration))

    result = {"language": info.language, "duration": info.duration, "segments": segments}
    (work_dir / "transcript.json").write_text(
        json.dumps(result, ensure_ascii=False), encoding="utf-8"
    )
    return result
