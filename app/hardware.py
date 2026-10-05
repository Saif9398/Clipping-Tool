"""Runtime hardware detection: pick whisper model/compute, worker counts, and ffmpeg encoder.

CPU-only is the first-class path; GPU (CUDA) and Intel QuickSync are used when present.
"""
from __future__ import annotations

import functools
import logging
import math
import os
import subprocess
from dataclasses import dataclass

from app import config

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Hardware:
    cpu_threads: int
    ram_gb: float
    has_cuda: bool
    encoder: str            # ffmpeg video encoder name
    encoder_args: list[str]
    whisper_model: str
    whisper_compute: str
    whisper_device: str     # "cuda" | "cpu" — must pair with compute (float16 needs cuda)
    whisper_threads: int
    stage2_workers: int     # transcribe / scenes / faces run as separate processes
    render_workers: int
    max_source_height: int  # effective download cap; CPU-only can't decode 4K AV1


def _ram_gb() -> float:
    try:
        if os.name == "nt":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            m = MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 1024**3
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except Exception:
        return 8.0


def _parse_cpu_max(raw: str) -> int | None:
    """Parse cgroup-v2 ``cpu.max`` and return its effective CPU quota."""
    try:
        quota, period = raw.split()[:2]
        if quota == "max":
            return None
        return max(1, math.ceil(int(quota) / int(period)))
    except (ValueError, ZeroDivisionError, IndexError):
        return None


def _cpu_threads() -> int:
    """CPU count visible to this process, capped by Linux container quotas.

    ``os.cpu_count()`` may report host resources above the container quota.
    Cap the effective count before selecting CTranslate2/OpenCV concurrency.
    """
    counts = [os.cpu_count() or 1]
    if hasattr(os, "sched_getaffinity"):
        try:
            counts.append(len(os.sched_getaffinity(0)))
        except OSError:
            pass

    if os.name != "nt":
        try:
            quota = _parse_cpu_max(
                open("/sys/fs/cgroup/cpu.max", encoding="utf-8").read().strip()
            )
            if quota:
                counts.append(quota)
        except OSError:
            # cgroup v1 fallback
            try:
                q = int(open(
                    "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", encoding="utf-8"
                ).read())
                p = int(open(
                    "/sys/fs/cgroup/cpu/cpu.cfs_period_us", encoding="utf-8"
                ).read())
                if q > 0 and p > 0:
                    counts.append(max(1, math.ceil(q / p)))
            except (OSError, ValueError):
                pass
    return max(1, min(counts))


def _has_cuda() -> bool:
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, timeout=10, check=True)
        return True
    except Exception:
        return False


def _encoder_works(name: str, extra: list[str]) -> bool:
    """Actually encode 8 test frames — listing in `-encoders` is not proof the driver works."""
    cmd = [
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=256x256:r=8:d=1",
        "-c:v", name, *extra, "-f", "null", "-",
    ]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False


def _nvenc_args(quality: int) -> list[str]:
    """nvenc args at the best quality this GPU actually accepts.

    The base list is required for -cq to mean anything at all. The extras are
    pure quality wins on Turing and newer but HARD-ERROR on older silicon
    (-b_ref_mode needs Turing+; -tune hq needs a recent SDK), and a rejected
    flag fails the encoder open for every clip rather than degrading — so the
    richer list is smoke-tested first and only kept if it really encodes.
    """
    base = [
        "-rc", "vbr", "-cq", str(max(1, quality - 3)), "-b:v", "0",
        "-profile:v", "high", "-spatial-aq", "1",
    ]
    preferred = ["-preset", "p6", "-tune", "hq", *base,
                 "-rc-lookahead", "20", "-b_ref_mode", "middle"]
    if _encoder_works("h264_nvenc", preferred):
        return preferred
    return ["-preset", "p5", *base]


@functools.lru_cache(maxsize=1)
def detect() -> Hardware:
    threads = _cpu_threads()
    ram = _ram_gb()
    cuda = _has_cuda()

    # Quality notes. ONE VIDEO_QUALITY knob drives all three encoders so tuning
    # is a .env edit — but the three scales are NOT interchangeable, so the knob
    # is MAPPED per encoder rather than passed through raw:
    #  - qsv  -global_quality (ICQ): honored as given; the measured-good local
    #    path (17 -> ~7 Mbps on real podcast footage; 23 gave 2-4 Mbps, mush).
    #  - x264 -crf: same ballpark as qsv ICQ, BUT preset matters as much as the
    #    number. veryfast throws away the RD tools that keep upscaled faces
    #    crisp; medium is the floor for output anyone watches. This is the
    #    fallback the GPU server lands on when nvenc/qsv are unavailable.
    #  - nvenc -cq: roughly 3 points "weaker" than x264 crf at the same number,
    #    so it gets a -3 offset plus the quality flags (p6/hq/lookahead/b-frames)
    #    it needs to compete. And -cq is SILENTLY IGNORED unless rate control is
    #    vbr with -b:v 0 — without those nvenc emits its ~2 Mbps default and
    #    1080x1920 turns to mush, so that pair is load-bearing, not decoration.
    q = str(config.VIDEO_QUALITY)
    if cuda and _encoder_works("h264_nvenc", ["-preset", "p4"]):
        encoder, enc_args = "h264_nvenc", _nvenc_args(config.VIDEO_QUALITY)
    elif _encoder_works("h264_qsv", ["-global_quality", "23"]):
        encoder, enc_args = "h264_qsv", ["-global_quality", q, "-look_ahead", "0"]
    else:
        encoder, enc_args = "libx264", ["-preset", "medium", "-crf", q]

    model = config.WHISPER_MODEL
    if model == "auto":
        if cuda:
            model = "large-v3" if ram >= 12 else "medium"
        elif ram >= 15 and threads >= 12:
            model = "medium"
        else:
            model = "small"
    compute = config.WHISPER_COMPUTE
    if compute == "auto":
        compute = "float16" if cuda else "int8"

    render_workers = config.RENDER_WORKERS or max(1, threads // 4)
    if not config.RENDER_WORKERS and ram < 10:
        # Auto mode on a low-RAM box: one clip at a time. Each render worker
        # loads cv2 + mediapipe and holds full frames; two in parallel on
        # <10 GB is what OOM-killed the server mid-render. An explicit
        # RENDER_WORKERS=N in .env still overrides this (the user opts in).
        render_workers = 1
    if encoder == "h264_nvenc":
        # A single successful smoke-test encode above only proves nvenc works
        # AT ALL, not that it survives N concurrent sessions — most GPUs cap
        # simultaneous NVENC sessions well below core-count-derived worker
        # counts, and exceeding it fails every session past the cap rather
        # than queuing. Clamp so Stage 5 never launches more than the driver
        # can actually open at once.
        render_workers = min(render_workers, max(1, config.NVENC_MAX_SESSIONS))

    # Stage 2 runs transcribe + scenes + faces as separate processes. On a
    # low-RAM box even Whisper plus one full-frame decoder can exhaust Windows'
    # commit limit (OpenCV then fails to allocate a single 1080p frame). Auto
    # mode is deliberately sequential under 10 GB; jobs.py also gives every
    # heavy stage a fresh process so native-library memory is fully reclaimed.
    # A deployment server with enough RAM keeps the 3-way parallel fast path.
    # Explicit ANALYSIS_WORKERS still wins.
    stage2_workers = config.ANALYSIS_WORKERS or (1 if ram < 10 else 3)

    # Effective source-resolution cap. Software-decoding 4K AV1/VP9 (YouTube
    # offers no 4K H.264) is infeasible on a CPU-only box — one 17-min 4K AV1
    # clip metered ~3 fps decode, ~132 min per analysis pass, and hung Stage 2.
    # A CUDA box (NVDEC decodes AV1) keeps the configured value; without it we
    # cap at 1080 regardless of .env. Analysis needs no more than 1080 anyway
    # (scenedetect + faces both downscale); only the render wants more, and only
    # where it can be decoded. Overriding an explicit .env is intentional here —
    # a 4K download this machine can't process is a footgun, not a feature.
    max_source_height = config.MAX_SOURCE_HEIGHT if cuda else min(config.MAX_SOURCE_HEIGHT, 1080)
    if max_source_height != config.MAX_SOURCE_HEIGHT:
        _log.warning("source height capped %d -> %d (no GPU decoder; 4K AV1/VP9 "
                     "is not CPU-decodable at usable speed here)",
                     config.MAX_SOURCE_HEIGHT, max_source_height)

    return Hardware(
        cpu_threads=threads,
        ram_gb=round(ram, 1),
        has_cuda=cuda,
        encoder=encoder,
        encoder_args=enc_args,
        whisper_model=model,
        whisper_compute=compute,
        whisper_device="cuda" if cuda else "cpu",
        whisper_threads=max(2, threads - 2),
        stage2_workers=stage2_workers,
        render_workers=render_workers,
        max_source_height=max_source_height,
    )
