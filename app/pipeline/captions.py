"""Stage 5 captions: word-level-synced ASS subtitles, burned in by the renderer.

Style: bold uppercase sans, heavy outline + soft shadow, grouped 2-4 words per
line, the currently spoken word highlighted (yellow + slight scale-up). Placed
in the lower-middle safe area — above TikTok's UI overlay zone.
"""
from __future__ import annotations

import json
from pathlib import Path

from app import config

MAX_WORDS_PER_LINE = 4
MAX_CHARS_PER_LINE = 18
GAP_BREAK_S = 0.6

HEADER = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {config.OUT_W}
PlayResY: {config.OUT_H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Arial,78,&H00FFFFFF,&H00FFFFFF,&H00101010,&H96000000,-1,0,0,0,100,100,1,0,1,6,2,2,60,60,{config.OUT_H - 1420},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
# MarginV above places baseline ~500px from bottom -> clear of TikTok UI, under split seam.

HIGHLIGHT = r"{\c&H00E5FF&\fscx108\fscy108}"   # warm yellow, slightly larger
RESET = r"{\c&HFFFFFF&\fscx100\fscy100}"


def build_ass(transcript: dict, t0: float, t1: float, out_path: Path) -> Path:
    words = []
    for seg in transcript["segments"]:
        for wd in seg["words"]:
            if wd["s"] >= t0 - 0.2 and wd["e"] <= t1 + 0.2:
                words.append({
                    "w": wd["w"].upper(),
                    "s": max(0.0, wd["s"] - t0),
                    "e": max(0.0, min(wd["e"], t1) - t0),
                })
    lines = _group_lines(words)
    timed = []  # (start, end, text) — clamped to non-overlap below
    for line in lines:
        for k, active in enumerate(line):
            start = active["s"]
            end = line[k + 1]["s"] if k + 1 < len(line) else line[-1]["e"]
            parts = []
            for j, wd in enumerate(line):
                parts.append((HIGHLIGHT if j == k else RESET) + _esc(wd["w"]))
            timed.append([start, end, " ".join(parts)])

    # Whisper word timings can overlap across line boundaries; two overlapping
    # Dialogue events render as stacked doubled text. Enforce globally: sort by
    # start, clamp each end to the next start, drop degenerates.
    timed.sort(key=lambda e: e[0])
    events = []
    for idx, (start, end, text) in enumerate(timed):
        if idx + 1 < len(timed):
            end = min(end, timed[idx + 1][0])
        if end - start < 0.02:
            if idx + 1 < len(timed) and timed[idx + 1][0] - start < 0.02:
                continue  # zero-width slot fully swallowed by the next event
            end = start + 0.02
        events.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Cap,,0,0,0,,{text}")
    out_path.write_text(HEADER + "\n".join(events) + "\n", encoding="utf-8-sig")
    return out_path


def _group_lines(words: list[dict]) -> list[list[dict]]:
    lines, cur, chars = [], [], 0
    for wd in words:
        gap_break = cur and (wd["s"] - cur[-1]["e"]) > GAP_BREAK_S
        full = len(cur) >= MAX_WORDS_PER_LINE or (cur and chars + len(wd["w"]) > MAX_CHARS_PER_LINE)
        punct_break = cur and cur[-1]["w"].rstrip('"').endswith((".", "!", "?", ","))
        if cur and (gap_break or full or (punct_break and len(cur) >= 2)):
            lines.append(cur)
            cur, chars = [], 0
        cur.append(wd)
        chars += len(wd["w"]) + 1
    if cur:
        lines.append(cur)
    return lines


def _esc(text: str) -> str:
    return text.replace("{", "(").replace("}", ")").replace("\\", "/")


def _ts(sec: float) -> str:
    cs = int(round(sec * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def load_transcript(work_dir: Path) -> dict:
    return json.loads((work_dir / "transcript.json").read_text(encoding="utf-8"))
