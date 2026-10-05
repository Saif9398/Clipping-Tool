"""Stage 5 audio: slice the clip's dialogue, denoise (afftdn), loudness-normalize
(loudnorm to -14 LUFS — platform standard), output 48 kHz stereo WAV for the mux.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def process_clip_audio(source: Path, t0: float, t1: float, out_wav: Path) -> Path:
    af = (
        "afftdn=nf=-25,"
        "loudnorm=I=-14:TP=-1.5:LRA=11,"
        "aresample=48000"
    )
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}", "-i", str(source),
        "-vn", "-af", af, "-ac", "2", "-c:a", "pcm_s16le", str(out_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_wav
