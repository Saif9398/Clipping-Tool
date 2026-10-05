"""Job orchestration: stage gates + real process-level parallelism.

Stage 1  download                     (job thread)
Stage 2  transcribe | scenes | faces  (parallel on capable hosts; isolated on low RAM)
Stage 3  scoring -> candidates        (job thread; network I/O)
Stage 4  human approval               (API gate — nothing renders until then)
Stage 5  render approved clips        (N worker PROCESSES, concurrent)
Stage 6  manifest + done

Per-stage wall times are recorded in job.json["timings"] so the Phase H test can
prove Stage 2/5 actually ran concurrently (sum of parts >> stage wall time).
"""
from __future__ import annotations

import errno
import json
import logging
import math
import os
import re
import threading
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path

from app import config
from app.hardware import detect

log = logging.getLogger(__name__)

_jobs: dict[str, dict] = {}
_locks: dict[str, threading.Lock] = {}
_owned: set[str] = set()  # jobs whose worker threads live in THIS process
_cancel: dict[str, threading.Event] = {}   # job_id -> set when the user hits Stop
_active_pool: dict[str, ProcessPoolExecutor] = {}  # job_id -> its live worker pool

# Caps how many jobs run their heavy pipeline (analysis or render) at once in
# this process. Per-job threads still spawn immediately — extras block here in
# "queued"/"render_queued" state until a slot frees.
_pipeline_sem = threading.BoundedSemaphore(max(1, config.MAX_CONCURRENT_JOBS))

TERMINAL_STATES = ("done", "done_with_errors", "error", "cancelled")
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class JobCancelled(Exception):
    """Raised inside a job thread when the user requested Stop — turned into a
    clean 'cancelled' terminal state instead of an error."""
# A job not owned by us (CLI run, previous server) is "interrupted" only after
# this long with zero file activity in its work dir. Stage transitions can be
# far apart on slow hardware, but progress files/clip artifacts update often.
STALE_AFTER_S = 900


# ---- Stage 2 worker-process entrypoints (top-level: Windows spawn-picklable) ----
# Each writes work_dir/progress_<name>.txt (0..1) so the parent/UI can show
# real progress across the process boundary without IPC.

def _progress_writer(work_dir: str, name: str):
    path = Path(work_dir) / f"progress_{name}.txt"
    state = {"last": -1.0}

    def cb(frac: float):
        if frac - state["last"] >= 0.01:
            state["last"] = frac
            try:
                path.write_text(f"{frac:.3f}", encoding="utf-8")
            except OSError:
                pass

    return cb


def _task_transcribe(work_dir: str, model: str, compute: str, threads: int,
                     device: str = "cpu") -> float:
    if (Path(work_dir) / "transcript.json").exists():  # retry: already computed
        return 0.0
    t = time.perf_counter()
    from app.pipeline.transcribe import transcribe

    transcribe(Path(work_dir), model, compute, threads, device=device,
               progress_cb=_progress_writer(work_dir, "transcribe"))
    return time.perf_counter() - t


def _task_scenes(work_dir: str) -> float:
    if (Path(work_dir) / "scenes.json").exists():
        return 0.0
    t = time.perf_counter()
    from app.pipeline.scenes import detect_scenes

    detect_scenes(Path(work_dir), progress_cb=_progress_writer(work_dir, "scenes"))
    return time.perf_counter() - t


def _task_faces(work_dir: str) -> float:
    if (Path(work_dir) / "faces_coarse.json").exists():
        return 0.0
    t = time.perf_counter()
    from app.pipeline.faces import detect_faces_coarse

    detect_faces_coarse(Path(work_dir), progress_cb=_progress_writer(work_dir, "faces"))
    return time.perf_counter() - t


def read_progress(job_id: str) -> dict[str, float]:
    out = {}
    for f in (config.WORK_DIR / job_id).glob("progress_*.txt"):
        try:
            out[f.stem.removeprefix("progress_")] = float(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return out


# ---- Job lifecycle ----

def create_job(url: str) -> dict:
    job_id = uuid.uuid4().hex[:10]
    job = {
        "id": job_id, "url": url, "state": "queued", "created": time.time(),
        "stages": {}, "timings": {}, "meta": {}, "candidates": [],
        "approved": [], "clips": [], "error": None,
    }
    _jobs[job_id] = job
    _locks[job_id] = threading.Lock()
    _owned.add(job_id)
    _save(job)
    threading.Thread(target=_run_analysis, args=(job_id,), daemon=True).start()
    return job


def get_job(job_id: str) -> dict | None:
    if job_id not in _owned:
        job = _load_from_disk(job_id)
        if job:
            _jobs[job_id] = job
            _locks.setdefault(job_id, threading.Lock())
    return _jobs.get(job_id)


def list_jobs() -> list[dict]:
    for jf in config.WORK_DIR.glob("*/job.json"):
        jid = jf.parent.name
        if jid not in _owned:
            job = _load_from_disk(jid)
            if job:
                _jobs[jid] = job
                _locks.setdefault(jid, threading.Lock())
    return sorted(_jobs.values(), key=lambda j: j["created"], reverse=True)


def approve(job_id: str, ranks: list[int]) -> dict:
    job = get_job(job_id)
    if job is None:
        raise ValueError("no such job")
    if job["state"] != "awaiting_approval":
        raise ValueError(f"job is in state {job['state']!r}, not awaiting approval")
    valid = {c["rank"] for c in job["candidates"]}
    picked = [r for r in ranks if r in valid]
    if not picked:
        raise ValueError("no valid candidate ranks given")
    _owned.add(job_id)  # the render thread will live in this process
    _update(job_id, approved=picked, state="render_queued")
    threading.Thread(target=_run_render, args=(job_id,), daemon=True).start()
    return job


def retry(job_id: str) -> dict:
    """Re-run a failed job in its EXISTING work dir — completed artifacts
    (download, transcript, face tracks, scenes) are detected and skipped."""
    job = get_job(job_id)
    if job is None:
        raise ValueError("no such job")
    if job["state"] not in TERMINAL_STATES:
        raise ValueError(f"job is in state {job['state']!r}, not a failed/finished state")
    _owned.add(job_id)
    _cancel.pop(job_id, None)  # a retried job starts with a clean cancel slate
    # Old progress/stage values otherwise make a retry look as if the failed
    # worker is still running.  Analysis artifacts themselves stay intact, so
    # the worker entrypoints above can reuse every completed stage.
    for progress_file in _work_dir(job_id).glob("progress_*.txt"):
        progress_file.unlink(missing_ok=True)
    _update(
        job_id,
        state="queued",
        error=None,
        trace=None,
        stages={},
        timings={},
        candidates=[],
        approved=[],
        clips=[],
        flagged=[],
    )
    threading.Thread(target=_run_analysis, args=(job_id,), daemon=True).start()
    return job


def cancel(job_id: str) -> dict:
    """Request Stop. Works at any stage: sets a per-job event the pipeline polls
    at every stage boundary and inside the render/analysis drain loops, and
    force-kills the job's live worker pool so a long stage stops promptly rather
    than at the next boundary. The job thread finalizes it to 'cancelled'."""
    job = get_job(job_id)
    if job is None:
        raise ValueError("no such job")
    if job["state"] in TERMINAL_STATES:
        raise ValueError(f"job already {job['state']}")
    _cancel.setdefault(job_id, threading.Event()).set()
    pool = _active_pool.get(job_id)
    if pool is not None:
        _terminate_pool(pool)  # stop the running stage now
    _update(job_id, state="cancelling")
    log.info("cancel requested for job %s", job_id)
    return job


def cancel_active_jobs_for_shutdown(reason: str = "server stopped by user") -> list[str]:
    """Stop every job owned by this server before the process exits.

    Persist a terminal state immediately: if the OS ends the process before a
    job thread reaches its normal JobCancelled handler, the next server start
    must not resurrect a forever-"cancelling" job in the UI.
    """
    stopped = []
    for job_id, job in list(_jobs.items()):
        if job_id not in _owned or job.get("state") in TERMINAL_STATES:
            continue
        try:
            cancel(job_id)
            _update(job_id, state="cancelled", error=reason)
            stopped.append(job_id)
        except ValueError:
            continue
    return stopped


def _is_cancelled(job_id: str) -> bool:
    ev = _cancel.get(job_id)
    return ev is not None and ev.is_set()


def _raise_if_cancelled(job_id: str) -> None:
    if _is_cancelled(job_id):
        raise JobCancelled()


def delete_job(job_id: str) -> dict:
    """Remove a finished job entirely: its work/ cache AND its output/ clips AND
    its in-memory record. Refuses while the job is still active (stop it first)."""
    import shutil

    job = get_job(job_id)
    if job is not None and job["state"] not in TERMINAL_STATES:
        raise ValueError(f"job is {job['state']} — stop it before deleting")
    freed = 0.0
    from app import diskspace

    wd = config.WORK_DIR / job_id
    if wd.is_dir():
        freed += diskspace.dir_size_gb(wd)
        shutil.rmtree(wd, ignore_errors=True)
    slug = ((job or {}).get("meta") or {}).get("slug")
    if slug:
        od = config.OUTPUT_DIR / slug
        if od.is_dir():
            freed += diskspace.dir_size_gb(od)
            shutil.rmtree(od, ignore_errors=True)
    for reg in (_jobs, _locks, _cancel, _active_pool):
        reg.pop(job_id, None)
    _owned.discard(job_id)
    log.info("deleted job %s (freed %.2f GB)", job_id, freed)
    return {"deleted": job_id, "freed_gb": round(freed, 2)}


def clear_caches() -> dict:
    """Reclaim disk by deleting the work/ cache (multi-GB source.mp4 +
    intermediates) of every FINISHED job, while KEEPING finished output/ clips.
    Running / awaiting-approval jobs are left untouched."""
    import shutil

    from app import diskspace

    freed, n = 0.0, 0
    for jf in list(config.WORK_DIR.glob("*/job.json")):
        jid = jf.parent.name
        job = get_job(jid)
        if job is None or job["state"] not in TERMINAL_STATES:
            continue
        freed += diskspace.dir_size_gb(jf.parent)
        shutil.rmtree(jf.parent, ignore_errors=True)
        n += 1
        # keep the in-memory record so History still lists it (as cache-cleared)
    log.info("clear_caches: removed %d work cache(s), freed %.2f GB", n, freed)
    return {"cleared": n, "freed_gb": round(freed, 2)}


def load_existing_jobs() -> None:
    """Populate the registry from disk on server start (disk stays the source of
    truth for non-owned jobs — see get_job/list_jobs — so a CLI run in another
    process shows live state here instead of a false 'interrupted' error).

    Also FINALIZE jobs stranded mid-flight by a killed process (crash / OOM /
    closed terminal): _load_from_disk relabels them 'error: interrupted' in
    memory, but the on-disk job.json still says 'rendering'/'analyzing'. Persist
    the error so the row shows a clear failure (and is retryable) instead of a
    forever-spinning stage the moment nobody is left to update it."""
    list_jobs()
    for jid, job in list(_jobs.items()):
        if job.get("state") == "error" and "interrupted" in (job.get("error") or ""):
            path = config.WORK_DIR / jid / "job.json"
            try:
                on_disk = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if on_disk.get("state") in TERMINAL_STATES:
                continue
            on_disk["state"] = "error"
            on_disk["error"] = job["error"]
            on_disk["updated"] = time.time()
            tmp = path.with_name("job.json.tmp")
            tmp.write_text(json.dumps(on_disk, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, path)
            log.info("reaped stranded job %s (%s) -> error", jid, on_disk.get("state"))


def _load_from_disk(job_id: str) -> dict | None:
    jf = config.WORK_DIR / job_id / "job.json"
    try:
        job = json.loads(jf.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, KeyError):
        return None
    if job.get("error"):
        job["error"] = _humanize_error_message(str(job["error"]))
    if job.get("state") not in TERMINAL_STATES and _last_activity(job_id) < time.time() - STALE_AFTER_S:
        job["state"] = "error"
        job["error"] = (f"interrupted — no file activity for over {STALE_AFTER_S // 60} min "
                        "(the process running this job likely stopped)")
    return job


def _last_activity(job_id: str) -> float:
    newest = 0.0
    try:
        for p in (config.WORK_DIR / job_id).iterdir():
            newest = max(newest, p.stat().st_mtime)
    except OSError:
        pass
    return newest


# ---- Internals ----

def _work_dir(job_id: str) -> Path:
    d = config.WORK_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(job: dict) -> None:
    # Atomic write (same pattern as mp_models): a plain write_text truncated by
    # ENOSPC or a crash leaves invalid JSON that makes the job invisible and
    # marks its dir "orphaned" to cleanup; and SSE/API readers polling job.json
    # must never observe a half-written file.
    wd = _work_dir(job["id"])
    tmp = wd / "job.json.tmp"
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, wd / "job.json")


def _update(job_id: str, **kw) -> dict:
    job = _jobs[job_id]
    with _locks[job_id]:
        job.update(kw)
        job["updated"] = time.time()
        _save(job)
    return job


def _stage(job_id: str, name: str, status: str, seconds: float | None = None) -> None:
    job = _jobs[job_id]
    with _locks[job_id]:
        job["stages"][name] = status
        if seconds is not None:
            job["timings"][name] = round(seconds, 1)
        job["updated"] = time.time()
        _save(job)


def _drain(futs: dict, deadline: float):
    """Yield (item, result, exc) as each future finishes. When `deadline` wall-
    clock seconds elapse, yield (item, None, TimeoutError(...)) for every future
    still unfinished and stop — so a wedged/killed worker can never hang a stage
    forever (the caller marks those items failed and force-kills the pool)."""
    done = set()
    try:
        for fut in as_completed(list(futs), timeout=deadline):
            done.add(fut)
            try:
                yield futs[fut], fut.result(), None
            except Exception as e:  # worker raised or died (BrokenProcessPool)
                yield futs[fut], None, e
    except FuturesTimeout:
        for fut, item in futs.items():
            if fut not in done:
                fut.cancel()
                yield item, None, TimeoutError(
                    f"timed out after {deadline:.0f}s — worker hung or was killed "
                    "(e.g. out of memory); aborted so the job does not stall")


def _terminate_pool(pool: ProcessPoolExecutor) -> None:
    """Force-kill worker processes so the executor's shutdown can't block on a
    task that will never return. Best-effort; the private-attr access is the
    only way to reach the OS processes and is deliberate."""
    try:
        for p in list(getattr(pool, "_processes", {}).values()):
            p.terminate()
    except Exception:
        pass


def _run_stage2_isolated(job_id: str, tasks: list[tuple[str, object, tuple]]) -> None:
    """Run each analysis task in its own one-worker process.

    Native allocations made by Whisper, OpenCV and MediaPipe are then returned
    to Windows when that worker exits.  Reusing one worker sequentially is not
    sufficient on low-memory machines because native libraries can retain
    arenas/caches after a task returns.
    """
    for name, func, args in tasks:
        _raise_if_cancelled(job_id)
        _stage(job_id, name, "running")
        with ProcessPoolExecutor(max_workers=1) as pool:
            _active_pool[job_id] = pool
            try:
                futures = {pool.submit(func, *args): name}
                for _, secs, exc in _drain(futures, config.ANALYSIS_TIMEOUT_S):
                    _raise_if_cancelled(job_id)
                    if exc is not None:
                        _terminate_pool(pool)
                        raise RuntimeError(f"analysis stage '{name}' {exc}")
                    _stage(job_id, name, "done", secs)
            finally:
                if _is_cancelled(job_id):
                    _terminate_pool(pool)
                _active_pool.pop(job_id, None)


def _run_stage2_parallel(
    job_id: str,
    tasks: list[tuple[str, object, tuple]],
    workers: int,
) -> None:
    """Run independent analysis tasks concurrently on hosts with enough RAM."""
    futures = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        _active_pool[job_id] = pool
        try:
            for name, func, args in tasks:
                futures[pool.submit(func, *args)] = name
            for name in futures.values():
                _stage(job_id, name, "running")
            for name, secs, exc in _drain(futures, config.ANALYSIS_TIMEOUT_S):
                _raise_if_cancelled(job_id)
                if exc is not None:
                    _terminate_pool(pool)
                    raise RuntimeError(f"analysis stage '{name}' {exc}")
                _stage(job_id, name, "done", secs)
        finally:
            if _is_cancelled(job_id):
                _terminate_pool(pool)
            _active_pool.pop(job_id, None)


def _run_analysis(job_id: str) -> None:
    from app.keepawake import keep_awake

    # Semaphore OUTSIDE keep_awake: a job waiting for a slot must not keep the
    # machine awake. No deadlock with renders — the slot is released before the
    # approval gate, and _run_render acquires its own.
    with _pipeline_sem:
        with keep_awake():
            _run_analysis_impl(job_id)


def _run_analysis_impl(job_id: str) -> None:
    job = _jobs[job_id]
    wd = _work_dir(job_id)
    hw = detect()
    try:
        _raise_if_cancelled(job_id)  # cancelled while queued, before any work
        # Defensive: main.py already does this at server startup, but the CLI
        # path (app.cli) never runs main.py's startup code, and this is a
        # cheap no-op once models exist. Must complete before Stage 2 spawns
        # its worker pool — those processes must never race a missing model.
        from app.pipeline import mp_models

        mp_models.ensure_models()

        # Fail fast if the disk is (nearly) full — the download alone is
        # multi-GB, and a full disk otherwise surfaces as a cryptic failure
        # deep inside whisper/ffmpeg. May emergency-clean old job caches.
        from app import diskspace

        diskspace.check("download + analysis")

        # Stage 1: download (skipped on retry when artifacts already exist)
        _update(job_id, state="downloading")
        _stage(job_id, "download", "running")
        t = time.perf_counter()
        cached = all((wd / f).exists() for f in ("source.mp4", "meta.json", "audio16k.wav"))
        if cached:
            meta = json.loads((wd / "meta.json").read_text(encoding="utf-8"))
            # A cached source taller than what this hardware can decode (e.g. a
            # 4K file from before the cap, or a config change) would re-hang
            # Stage 2 on Retry. Drop the stale artifacts and re-download at the
            # capped height instead of silently reusing the un-decodable file.
            if int(meta.get("height") or 0) > hw.max_source_height:
                log.info("job %s: cached source %dp exceeds decodable cap %dp — "
                         "re-downloading", job_id, meta.get("height"), hw.max_source_height)
                for f in ("source.mp4", "audio16k.wav"):
                    (wd / f).unlink(missing_ok=True)
                cached = False
        if not cached:
            from app.pipeline.download import download

            meta = download(job["url"], wd, hw.max_source_height,
                            should_cancel=lambda: _is_cancelled(job_id))
        _update(job_id, meta=meta)
        _stage(job_id, "download", "done", time.perf_counter() - t)

        # Stage 2: parallel on capable hosts.  On low-RAM hosts detect() returns
        # one worker; each task then gets a fresh process so Whisper's native
        # memory is fully released before OpenCV/MediaPipe start.
        _raise_if_cancelled(job_id)
        _update(job_id, state="analyzing")
        t2 = time.perf_counter()
        tasks = [
            (
                "transcribe",
                _task_transcribe,
                (
                    str(wd),
                    hw.whisper_model,
                    hw.whisper_compute,
                    hw.whisper_threads,
                    hw.whisper_device,
                ),
            ),
            ("scenes", _task_scenes, (str(wd),)),
            ("faces", _task_faces, (str(wd),)),
        ]
        if hw.stage2_workers == 1:
            log.info("stage2: low-memory isolated mode (fresh process per task)")
            _run_stage2_isolated(job_id, tasks)
        else:
            log.info("stage2: parallel mode (%d workers)", hw.stage2_workers)
            _run_stage2_parallel(job_id, tasks, hw.stage2_workers)
        _stage(job_id, "stage2_wall", "done", time.perf_counter() - t2)

        # Stage 3: scoring
        _raise_if_cancelled(job_id)
        _update(job_id, state="scoring")
        _stage(job_id, "scoring", "running")
        t3 = time.perf_counter()
        from app.pipeline.scoring.candidates import build_candidates

        cands = build_candidates(wd)
        _stage(job_id, "scoring", "done", time.perf_counter() - t3)
        public = [
            {k: c[k] for k in ("rank", "start", "end", "title", "hook", "reason",
                               "score", "provider")}
            for c in cands
        ]
        for c in public:
            c["duration"] = round(c["end"] - c["start"], 1)
        _update(job_id, candidates=public, state="awaiting_approval")
    except JobCancelled:
        log.info("analysis cancelled for job %s", job_id)
        _update(job_id, state="cancelled", error="cancelled by user")
    except Exception as e:
        if _is_cancelled(job_id):  # e.g. yt-dlp DownloadCancelled from Stop
            log.info("analysis cancelled (mid-op) for job %s", job_id)
            _update(job_id, state="cancelled", error="cancelled by user")
        else:
            log.exception("analysis failed for job %s", job_id)
            _update(job_id, state="error", error=_describe_error(e),
                    trace=traceback.format_exc()[-2000:])
    finally:
        _cancel.pop(job_id, None)


def _run_render(job_id: str) -> None:
    from app.keepawake import keep_awake

    with _pipeline_sem:
        _update(job_id, state="rendering")
        with keep_awake():
            _run_render_impl(job_id)


def _run_render_impl(job_id: str) -> None:
    job = _jobs[job_id]
    wd = _work_dir(job_id)
    hw = detect()
    try:
        _raise_if_cancelled(job_id)
        from app import diskspace

        diskspace.check("render")

        out_dir = config.OUTPUT_DIR / job["meta"]["slug"]
        picked = [c for c in job["candidates"] if c["rank"] in job["approved"]]
        t5 = time.perf_counter()
        results, errors = [], []
        from app.pipeline.renderer import render_clip

        workers = min(hw.render_workers, len(picked)) or 1
        # One OpenCV producer thread runs alongside each ffmpeg filter graph.
        # Divide the cgroup-aware CPU quota across both halves of every worker.
        filter_threads = max(1, (hw.cpu_threads - workers) // workers)
        log.info("stage5: %d render worker(s) for %d clip(s), encoder=%s, "
                 "filter_threads=%d "
                 "(hw.render_workers=%d, NVENC_MAX_SESSIONS=%d)",
                 workers, len(picked), hw.encoder, filter_threads, hw.render_workers,
                 config.NVENC_MAX_SESSIONS)
        # Total Stage-5 budget: per-clip ceiling x how many sequential batches
        # the worker count implies, plus slack for pool spin-up.
        deadline = config.RENDER_CLIP_TIMEOUT_S * math.ceil(len(picked) / workers) + 120
        hung = False
        with ProcessPoolExecutor(max_workers=workers) as pool:
            _active_pool[job_id] = pool
            try:
                futs = {
                    pool.submit(render_clip, str(wd), str(out_dir), c,
                                hw.encoder, hw.encoder_args, filter_threads): c
                    for c in picked
                }
                for c in picked:
                    _stage(job_id, f"render_{c['rank']:02d}", "running")
                for c, res, exc in _drain(futs, deadline):
                    _raise_if_cancelled(job_id)  # Stop hit mid-render: pool killed
                    if exc is None:
                        res["url"] = f"/clips/{job['meta']['slug']}/{res['file']}"
                        results.append(res)
                        qa_ok = res.get("qa", {}).get("passed", True)
                        _stage(job_id, f"render_{c['rank']:02d}",
                               "done" if qa_ok else "review", res.get("elapsed_s"))
                        _update(job_id, clips=sorted(results, key=lambda r: r["rank"]))
                    else:
                        log.error("clip %s failed: %s", c["rank"], exc)
                        errors.append(f"clip {c['rank']}: {exc}")
                        _stage(job_id, f"render_{c['rank']:02d}", "failed")
                        hung = hung or isinstance(exc, TimeoutError)
                if hung:
                    _terminate_pool(pool)  # kill wedged workers so shutdown returns
            finally:
                if _is_cancelled(job_id):
                    _terminate_pool(pool)  # ensure shutdown() can't block on a worker
                _active_pool.pop(job_id, None)
        _stage(job_id, "stage5_wall", "done", time.perf_counter() - t5)

        flagged = [r["rank"] for r in results if not r.get("qa", {}).get("passed", True)]
        from app import version

        manifest = {
            "source": job["meta"], "clips": sorted(results, key=lambda r: r["rank"]),
            "errors": errors, "flagged_for_review": sorted(flagged),
            "render_workers": workers, "code_version": version.commit(),
            # The settings that actually produced these pixels. Without them a
            # "why does this look softer than last week's?" question can only be
            # answered by guessing which .env/commit was live at render time.
            "render_settings": {
                "encoder": hw.encoder, "encoder_args": hw.encoder_args,
                "filter_threads_per_worker": filter_threads,
                "video_quality": config.VIDEO_QUALITY,
                "zoom_factor": config.ZOOM_FACTOR,
                "max_upscale": config.MAX_UPSCALE,
                "look_filter": config.LOOK_FILTER,
                "look_filter_chain": config.LOOK_FILTER_CHAIN,
                "max_source_height": config.MAX_SOURCE_HEIGHT,
            },
        }
        from app.pipeline.renderer import _unique_path

        out_dir.mkdir(parents=True, exist_ok=True)
        _unique_path(out_dir, "manifest", ".json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        state = "done" if results and not errors else ("done_with_errors" if results else "error")
        _update(job_id, state=state, flagged=sorted(flagged), error="; ".join(errors) or None)
    except JobCancelled:
        log.info("render cancelled for job %s", job_id)
        _update(job_id, state="cancelled", error="cancelled by user")
    except Exception as e:
        if _is_cancelled(job_id):
            log.info("render cancelled (mid-op) for job %s", job_id)
            _update(job_id, state="cancelled", error="cancelled by user")
        else:
            log.exception("render failed for job %s", job_id)
            _update(job_id, state="error", error=_describe_error(e),
                    trace=traceback.format_exc()[-2000:])
    finally:
        _cancel.pop(job_id, None)


def _humanize_error_message(detail: str, error_type: str = "") -> str:
    """Strip terminal formatting and replace common failures with next steps."""
    detail = _ANSI_ESCAPE_RE.sub("", detail).strip()
    text = detail.lower()
    if error_type.lower() == "memoryerror" or any(
        marker in text
        for marker in (
            "outofmemory",
            "out of memory",
            "insufficient memory",
            "failed to allocate",
        )
    ):
        return (
            "Video analysis ran out of RAM. Clipping Tool will reuse completed work; "
            "close memory-heavy apps, then click Retry."
        )
    if (
        "winerror 10013" in text
        or "forbidden by its access permissions" in text
        or (
            "permission denied" in text
            and ("socket" in text or "connection" in text)
        )
    ):
        return (
            "YouTube network access is blocked on this server. "
            "Restart Clipping Tool with internet/network permission, then click Retry."
        )
    if "unable to download" in text or "downloaderror" in error_type.lower():
        return (
            "YouTube download failed. Check that the link is public and available, "
            "then click Retry."
        )
    return f"{error_type}: {detail}" if error_type else detail


def _describe_error(e: Exception) -> str:
    """Human-facing job error. ENOSPC hides inside library/subprocess errors
    that look nothing like 'disk full' — name it when we see it."""
    detail = str(e)
    msg = _humanize_error_message(detail, type(e).__name__)
    text = detail.lower()
    if (isinstance(e, OSError) and e.errno == errno.ENOSPC) or "no space left" in text:
        msg = ("server disk is full — free space, lower RETENTION_DAYS, or "
               f"delete old dirs under {config.WORK_DIR} | {msg}")
    return msg
