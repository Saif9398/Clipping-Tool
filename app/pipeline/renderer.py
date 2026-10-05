"""Stage 5 renderer: one approved clip -> finished 1080x1920 MP4.

Per clip (this function runs in a worker process, one process per clip):
  1. speaker.analyze_clip()      — FaceMesh mouth-activity + speech mask -> states
  2. camera.plan()               — per-frame crop plan (single / split)
  3. audio.process_clip_audio()  — denoise + loudnorm WAV
  4. captions.build_ass()        — word-synced ASS
  5. frame loop -> ffmpeg stdin  — subtitles burned + audio muxed in ONE encode
  6. ffprobe verification        — frame count must match the plan exactly

The loop is frame-count-driven: exactly n_frames frames are always written
(a failed read repeats the last good frame), so dropped frames are impossible
by construction; the ffprobe assert catches anything else.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from app import config
from app.pipeline import audio as audio_mod
from app.pipeline import camera, captions, speaker

log = logging.getLogger(__name__)

FADE_V = 0.25
FADE_A = 0.15
DIVIDER_PX = 4


def _unique_path(directory: Path, stem: str, ext: str) -> Path:
    """stem.ext if free, else stem_2.ext, stem_3.ext, ... — never overwrite an
    existing finished file (re-running the same URL used to clobber clips).
    The exists-check races only if two jobs with an identical output slug render
    simultaneously; impossible with MAX_CONCURRENT_JOBS=1."""
    p = directory / f"{stem}{ext}"
    i = 2
    while p.exists():
        p = directory / f"{stem}_{i}{ext}"
        i += 1
    return p


def render_clip(work_dir: str, out_dir: str, clip: dict, encoder: str,
                encoder_args: list[str], filter_threads: int = 1) -> dict:
    """Worker-process entrypoint (picklable args only). Returns result dict."""
    import cv2

    started = time.perf_counter()
    # Each render also has a concurrent ffmpeg filter graph. Letting OpenCV see
    # all host CPUs makes every worker spawn dozens of threads inside a
    # quota-limited Pod; one producer thread per worker keeps aggregate
    # throughput predictable while ffmpeg uses its explicit budget below.
    cv2.setNumThreads(1)
    filter_threads = max(1, int(filter_threads))

    work = Path(work_dir)
    # resolve() because ffmpeg runs with cwd=clip_dir below — a relative out_dir
    # would silently point somewhere else inside the ffmpeg command
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0, t1 = float(clip["start"]), float(clip["end"])
    transcript = captions.load_transcript(work)

    analysis = speaker.analyze_clip(work, t0, t1, transcript)
    n, fps = analysis.n_frames, analysis.fps

    cap = cv2.VideoCapture(str(work / "source.mp4"))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scenes = json.loads((work / "scenes.json").read_text(encoding="utf-8"))["boundaries"]
    cut_frames = [int(round((b - t0) * fps)) for b in scenes if t0 < b < t1]

    plans = camera.plan(analysis, src_w, src_h, cut_frames)

    clip_tag = f"{clip['rank']:02d}_{_slug(clip['title'])}"
    clip_dir = work / f"clip_{clip['rank']:02d}"
    clip_dir.mkdir(exist_ok=True)
    wav = audio_mod.process_clip_audio(work / "source.mp4", t0, t1, clip_dir / "audio.wav")
    ass = captions.build_ass(transcript, t0, t1, clip_dir / "captions.ass")
    out_path = _unique_path(out, clip_tag, ".mp4")

    dur = n / fps
    look = _look_filter_chain()
    filter_v = (
        f"{look}ass=captions.ass,"
        f"fade=t=in:st=0:d={FADE_V},fade=t=out:st={max(0.0, dur - FADE_V):.3f}:d={FADE_V}"
    )
    # apad+atrim pins audio to exactly the video duration, so -frames:v alone
    # controls length (a -shortest race could otherwise clip the last frames)
    filter_a = (
        f"afade=t=in:d={FADE_A},afade=t=out:st={max(0.0, dur - FADE_A):.3f}:d={FADE_A},"
        f"apad,atrim=0:{dur:.6f}"
    )
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-filter_complex_threads", str(filter_threads),
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{config.OUT_W}x{config.OUT_H}", "-r", f"{fps:.6f}", "-i", "pipe:0",
        "-i", str(wav.name),
        "-filter_complex", f"[0:v]{filter_v}[v];[1:a]{filter_a}[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", encoder, *encoder_args, "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart", "-frames:v", str(n),
        str(out_path),
    ]
    # cwd=clip_dir so the ass filter sees a relative filename (avoids Windows
    # drive-colon escaping hell inside ffmpeg filter strings)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(clip_dir))

    f_start = int(round(t0 * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    last_frame = np.zeros((src_h, src_w, 3), dtype=np.uint8)
    write_err: Exception | None = None
    try:
        for i in range(n):
            ok, frame = cap.read()
            if ok:
                last_frame = frame
            else:
                frame = last_frame  # never drop a frame
            try:
                proc.stdin.write(_compose(frame, plans[i]).tobytes())
            except (BrokenPipeError, OSError, ValueError) as e:
                # ffmpeg already exited (e.g. it failed to open the encoder —
                # common when too many concurrent hardware-encoder sessions
                # exceed the GPU's session cap). Stop feeding a dead pipe
                # immediately; the real cause is in stderr, fetched below.
                # Do NOT re-raise here: closing an already-broken stdin in the
                # `finally` block below would itself raise (flush of a stream
                # that failed mid-flush), and that secondary ValueError would
                # replace this one in the traceback and hide ffmpeg's actual
                # error message.
                write_err = e
                break
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass  # already broken/closed — the real error surfaces below
        # CRITICAL for Linux: hand communicate() a None stdin. CPython's POSIX
        # _communicate() flushes proc.stdin itself, guarded only for
        # BrokenPipeError — flushing the pipe we just closed raises
        # ValueError("flush of closed file") from INSIDE communicate(), on
        # every render including successful ones. (Windows' communicate()
        # only close()s, which is a no-op here — the bug never fires locally.)
        # communicate() checks `if self.stdin:` first, so None skips its
        # stdin handling entirely; stderr draining and returncode are intact.
        proc.stdin = None
        _, err = proc.communicate(timeout=600)
        if proc.returncode != 0 or write_err is not None:
            # A dead encode leaves a partial MP4 at out_path; _unique_path
            # never overwrites, so without this the corrupt file would sit in
            # output/<slug>/ forever looking downloadable.
            out_path.unlink(missing_ok=True)
            msg = err.decode(errors="replace")[-800:] or str(write_err)
            if "no space left" in msg.lower():
                msg = "server disk is full (No space left on device) | " + msg
            raise RuntimeError(f"ffmpeg encode failed: {msg}")

    try:
        _verify(out_path, n)
    except Exception:
        out_path.unlink(missing_ok=True)  # failed verification = corrupt output
        raise
    from app.pipeline import qa

    qa_result = qa.run_qa(out_path, clip_dir, plans, fps,
                          qa_dir=out / "qa" / f"clip_{clip['rank']:02d}")
    # Success: the per-clip intermediates (audio.wav ~18 MB + captions.ass)
    # have served their purpose — QA above was the last reader. Kept on
    # failure (we never get here) for debugging.
    shutil.rmtree(clip_dir, ignore_errors=True)
    log.info("rendered %s (%d frames) qa=%s", out_path.name, n,
             "passed" if qa_result["passed"] else "FLAGGED")
    return {"rank": clip["rank"], "file": out_path.name, "frames": n,
            "duration": round(dur, 2), "elapsed_s": round(time.perf_counter() - started, 1),
            "filter_threads": filter_threads, "qa": qa_result}


def _look_filter_chain() -> str:
    """CapCut-style '4K Pro' enhancement, applied BEFORE the caption burn so text
    stays halo-free. Returns "" or a chain ending in ',' for direct prepending."""
    if config.LOOK_FILTER_CHAIN.strip():
        return config.LOOK_FILTER_CHAIN.strip().rstrip(",") + ","
    s = max(0.0, config.LOOK_FILTER)
    if s <= 0.0:
        return ""
    return (
        f"unsharp=5:5:{0.8 * s:.2f}:5:5:0.0,"
        f"cas={min(1.0, 0.35 * s):.3f},"
        f"eq=contrast={1.0 + 0.08 * s:.3f}:saturation={1.0 + 0.05 * s:.3f},"
        f"vibrance=intensity={0.18 * s:.3f},"
    )


def _compose(frame: np.ndarray, plan: camera.FramePlan) -> np.ndarray:
    import cv2

    W, H = config.OUT_W, config.OUT_H
    if plan.mode == "split" and len(plan.rects) == 2:
        half_h = H // 2
        top = _crop_scale(frame, plan.rects[0], W, half_h)
        bottom = _crop_scale(frame, plan.rects[1], W, half_h)
        canvas = np.vstack([top, bottom])
        y = half_h
        cv2.line(canvas, (0, y), (W, y), (16, 16, 16), DIVIDER_PX)
        return np.ascontiguousarray(canvas)
    return _crop_scale(frame, plan.rects[0], W, H)


def _crop_scale(frame: np.ndarray, rect: tuple, out_w: int, out_h: int) -> np.ndarray:
    import cv2

    x, y, w, h = rect
    # subpixel crop center keeps camera motion buttery; int size is fine (see camera.py)
    patch = cv2.getRectSubPix(frame, (max(2, int(round(w))), max(2, int(round(h)))),
                              (x + w / 2, y + h / 2))
    # Upscales are the norm here (a 9:16 crop of a 1080p source is <=607 px
    # wide before scaling to 1080): Lanczos keeps edges crisp where bilinear
    # visibly softens at the 1.8-3x factors face punch-ins produce.
    interp = cv2.INTER_AREA if patch.shape[1] > out_w else cv2.INTER_LANCZOS4
    return cv2.resize(patch, (out_w, out_h), interpolation=interp)


def _verify(path: Path, expected_frames: int) -> None:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,width,height", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    st = json.loads(r.stdout)["streams"][0]
    got = int(st["nb_read_frames"])
    if got != expected_frames:
        raise RuntimeError(f"{path.name}: frame count {got} != expected {expected_frames}")
    if st["width"] != config.OUT_W or st["height"] != config.OUT_H:
        raise RuntimeError(f"{path.name}: wrong resolution {st['width']}x{st['height']}")


def _slug(text: str) -> str:
    from app.pipeline.download import slugify

    return slugify(text, max_len=40)
