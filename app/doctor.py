"""Read-only server diagnostic: run `python -m app.doctor` on any machine
(Windows laptop or Linux server) to answer, without starting a job:

  - which git commit is actually running, and is the tree clean / behind origin
  - disk: free space per volume, what work/, output/, models/ are consuming,
    and a per-job breakdown of work/ (what to delete when the disk fills)
  - model files: present, non-empty, openable; orphaned .part temps
  - hardware detection result: encoder, worker counts, NVENC cap in effect
  - retention / disk-guardrail settings in effect
  - job.json health: unreadable (corrupt) job files that hide jobs from the UI

Prints a report; changes nothing.
"""
from __future__ import annotations

import json
import subprocess
import sys

from app import config


def _source_line(job_dir) -> str:
    """codec/WxH/bitrate of a job's downloaded source, or "" if there isn't one.
    Crops are cut from this and upscaled to 1080x1920, so a starved or
    low-resolution source is a sharpness ceiling nothing downstream can lift."""
    src = job_dir / "source.mp4"
    if not src.exists():
        return ""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name,width,height", "-show_entries", "format=bit_rate",
             "-of", "json", str(src)],
            capture_output=True, text=True, timeout=30, check=True,
        )
        data = json.loads(r.stdout)
        st = data["streams"][0]
        mbps = float(data.get("format", {}).get("bit_rate") or 0) / 1e6
        return (f"{st.get('codec_name', '?')} {st.get('width')}x{st.get('height')} "
                f"{mbps:.2f} Mbps")
    except Exception as e:
        return f"(unprobeable: {e})"


def _git(*args: str) -> str:
    try:
        r = subprocess.run(["git", "-C", str(config.PROJECT_ROOT), *args],
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else f"(git error: {r.stderr.strip()})"
    except Exception as e:  # git missing entirely
        return f"(unavailable: {e})"


def main() -> int:
    from app import diskspace
    from app.hardware import detect
    from app.pipeline.mp_models import _MODELS

    problems: list[str] = []
    print("=== Clipping Tool doctor ===\n")

    # --- deployed code ---
    head = _git("rev-parse", "--short", "HEAD")
    dirty = _git("status", "--porcelain")
    print(f"git HEAD: {head}  ({_git('log', '-1', '--format=%s')})")
    print(f"working tree: {'CLEAN' if not dirty else 'MODIFIED — uncommitted changes present'}")
    behind = _git("rev-list", "--count", "HEAD..origin/main")
    if behind.isdigit() and int(behind) > 0:
        problems.append(f"code is {behind} commit(s) behind origin/main — run: git pull")
    print(f"behind origin/main: {behind} commit(s)  [git fetch first for a live answer]\n")

    # --- disk ---
    d = diskspace.usage_summary()
    print(f"work volume free : {d['free_gb_work_volume']:.1f} GB"
          f"   (warn < {d['warn_free_gb']:.0f}, block < {d['min_free_gb']:.0f})")
    print(f"output volume free: {d['free_gb_output_volume']:.1f} GB")
    print(f"work/   : {d['work_dir_gb']:.2f} GB   output/ : {d['output_dir_gb']:.2f} GB"
          f"   models/ : {d['models_dir_gb']:.2f} GB")
    if d["free_gb_work_volume"] < config.MIN_FREE_GB:
        problems.append(f"free space {d['free_gb_work_volume']:.1f} GB is below "
                        f"MIN_FREE_GB={config.MIN_FREE_GB:.0f} — jobs will be blocked")
    elif d["free_gb_work_volume"] < config.WARN_FREE_GB:
        problems.append(f"free space {d['free_gb_work_volume']:.1f} GB is below the "
                        f"warn threshold ({config.WARN_FREE_GB:.0f} GB)")

    print("\nper-job work/ breakdown:")
    bad_json = []
    if config.WORK_DIR.is_dir():
        rows = []
        for jd in sorted(config.WORK_DIR.iterdir()):
            if not jd.is_dir():
                continue
            state = "?"
            jf = jd / "job.json"
            if jf.exists():
                try:
                    state = json.loads(jf.read_text(encoding="utf-8")).get("state", "?")
                except (OSError, json.JSONDecodeError):
                    state = "CORRUPT job.json"
                    bad_json.append(jd.name)
            else:
                state = "no job.json (orphan)"
            rows.append((diskspace.dir_size_gb(jd), jd.name, state, _source_line(jd)))
        for size, name, state, src in sorted(rows, reverse=True):
            print(f"  {size:7.2f} GB  {name}  [{state}]")
            if src:
                print(f"               source: {src}")
        if not rows:
            print("  (empty)")
    else:
        print("  (work dir does not exist yet)")
    if bad_json:
        problems.append(f"corrupt job.json in: {', '.join(bad_json)} — these jobs are "
                        "invisible in the UI; their dirs are treated as orphans by cleanup")

    # --- models ---
    print("\nmodel files:")
    for name in _MODELS:
        p = config.MODELS_DIR / name
        if not p.exists():
            print(f"  MISSING  {name}")
            problems.append(f"model missing: {name} (ensure_models() downloads it at startup)")
            continue
        size = p.stat().st_size
        try:
            with open(p, "rb"):
                readable = True
        except OSError:
            readable = False
        status = "ok" if size > 0 and readable else "CORRUPT/UNREADABLE"
        print(f"  {status:8} {name}  ({size / 1e6:.1f} MB)")
        if status != "ok":
            problems.append(f"model {name} is empty or unreadable — delete it and restart "
                            "(it will re-download)")
    parts = list(config.MODELS_DIR.glob("*.part")) if config.MODELS_DIR.is_dir() else []
    if parts:
        print(f"  orphaned .part temps: {len(parts)} (server startup removes these)")

    # --- hardware / settings ---
    hw = detect()
    print(f"\nhardware: encoder={hw.encoder} render_workers={hw.render_workers} "
          f"stage2_workers={hw.stage2_workers} "
          f"cuda={hw.has_cuda} whisper={hw.whisper_model}/{hw.whisper_compute} "
          f"threads={hw.cpu_threads} ram={hw.ram_gb} GB")
    print(f"  encoder args: {' '.join(hw.encoder_args)}")
    if hw.encoder == "h264_nvenc":
        print(f"  NVENC session cap in effect: NVENC_MAX_SESSIONS={config.NVENC_MAX_SESSIONS}")
    # The knobs that decide how sharp a clip looks, in one place — the encoder
    # is only half of it; ZOOM_FACTOR sets how far the crop is upscaled.
    print(f"quality: VIDEO_QUALITY={config.VIDEO_QUALITY} ZOOM_FACTOR={config.ZOOM_FACTOR} "
          f"MAX_UPSCALE={config.MAX_UPSCALE} "
          f"LOOK_FILTER={config.LOOK_FILTER}"
          f"{' (overridden by LOOK_FILTER_CHAIN)' if config.LOOK_FILTER_CHAIN.strip() else ''} "
          f"MAX_SOURCE_HEIGHT={hw.max_source_height}"
          f"{f' (configured {config.MAX_SOURCE_HEIGHT}, capped: no GPU decoder)' if hw.max_source_height != config.MAX_SOURCE_HEIGHT else ''}")
    print(f"settings: RETENTION_DAYS={config.RETENTION_DAYS} MIN_FREE_GB={config.MIN_FREE_GB} "
          f"WARN_FREE_GB={config.WARN_FREE_GB} MAX_CONCURRENT_JOBS={config.MAX_CONCURRENT_JOBS}")
    if config.MAX_CONCURRENT_JOBS > 1 and hw.has_cuda:
        problems.append("MAX_CONCURRENT_JOBS>1 on a CUDA machine — whisper VRAM and NVENC "
                        "sessions can collide; keep it at 1")

    # --- verdict ---
    print("\n=== verdict ===")
    if problems:
        for p in problems:
            print(f"  PROBLEM: {p}")
        return 1
    print("  no problems detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
