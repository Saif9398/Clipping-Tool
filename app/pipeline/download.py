"""Stage 1: download the source video with yt-dlp and extract a 16 kHz mono WAV for analysis."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


def slugify(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    s = re.sub(r"[\s_-]+", "-", s)
    return s[:max_len].strip("-") or "video"


def download(url: str, work_dir: Path, max_height: int = 1080, progress_cb=None,
             should_cancel=None) -> dict:
    """Download `url` into work_dir/source.mp4. Returns metadata dict (also saved
    to meta.json). `should_cancel`, if given, is polled during the download; when
    it returns True the download aborts promptly (user hit Stop)."""
    import yt_dlp

    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / "source.mp4"

    def hook(d):
        if should_cancel is not None and should_cancel():
            # yt-dlp catches this in its own loop and aborts the download.
            raise yt_dlp.utils.DownloadCancelled("cancelled by user")
        if progress_cb and d.get("status") == "downloading" and d.get("total_bytes"):
            progress_cb(d.get("downloaded_bytes", 0) / d["total_bytes"])

    opts = {
        "format": (
            f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]"
            f"/bestvideo[height<={max_height}]+bestaudio"
            f"/best[height<={max_height}]/best"
        ),
        # yt-dlp's default sort ranks codecs (av01 > vp9 > avc1) above bitrate,
        # which picked a 601k AV1 stream over a 1308k H.264 one — crops inherit
        # that starved detail budget. Within the height cap, take the
        # highest-bitrate stream instead; on this CPU avc1 also decodes faster.
        "format_sort": ["res", "tbr"],
        "merge_output_format": "mp4",
        "outtmpl": str(work_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
        "retries": 5,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": 4,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    if not out_path.exists():
        cands = [p for p in work_dir.glob("source.*") if p.suffix in (".mp4", ".mkv", ".webm")]
        if not cands:
            raise RuntimeError("yt-dlp finished but no source video file found")
        src = cands[0]
        if src.suffix != ".mp4":
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-c", "copy", str(out_path)],
                check=True, capture_output=True,
            )
            src.unlink()
        else:
            src.rename(out_path)

    probe = _probe(out_path)
    meta = {
        "url": url,
        "title": info.get("title") or "video",
        "slug": slugify(info.get("title") or "video"),
        "duration": float(info.get("duration") or probe["duration"]),
        "fps": probe["fps"],
        "width": probe["width"],
        "height": probe["height"],
        "uploader": info.get("uploader") or "",
    }
    (work_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 16 kHz mono WAV drives transcription + VAD/energy analysis
    wav = work_dir / "audio16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(out_path),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
        check=True, capture_output=True,
    )
    return meta


def _probe(path: Path) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,duration",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(r.stdout)
    st = data["streams"][0]
    num, den = st["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    duration = float(st.get("duration") or data["format"]["duration"])
    return {"width": st["width"], "height": st["height"], "fps": fps, "duration": duration}
