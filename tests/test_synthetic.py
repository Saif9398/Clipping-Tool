"""Synthetic pipeline verification (no paid APIs or GPU required).

The web startup checks require the public vision models to be cached first.
Run ensure_models() from app.pipeline.mp_models before running this suite.

Builds a fake 20s source video + transcript, monkeypatches speaker analysis with
a crafted two-person timeline (single -> split -> single), then runs the REAL
camera planner + renderer + audio + captions and asserts the output contract:
exact frame count, exact 1080x1920, audio present.

Run:  python -m pytest tests/ -v (after installing requirements-dev.txt)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.hardware import _parse_cpu_max  # noqa: E402
from app.pipeline import renderer, speaker  # noqa: E402
from app.pipeline.scoring.candidates import _build_windows, _dedupe, target_clip_count  # noqa: E402

WORK = config.WORK_DIR / "_synthetic_test"
OUT = config.OUTPUT_DIR / "_synthetic_test"
DUR = 20.0
FPS = 30.0


def _sleep_forever(_seconds: float = 60.0):
    """Top-level (picklable) worker that outlives any test deadline — stands in
    for a wedged render worker so the timeout safeguard can be exercised."""
    import time as _t
    _t.sleep(_seconds)
    return "should-never-return"


def test_cgroup_cpu_quota_parser():
    assert _parse_cpu_max("600000 100000") == 6
    assert _parse_cpu_max("150000 100000") == 2
    assert _parse_cpu_max("max 100000") is None
    assert _parse_cpu_max("invalid") is None


def make_source():
    WORK.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate={FPS:.0f}:duration={DUR}",
        "-f", "lavfi", "-i", f"sine=frequency=300:duration={DUR}",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
        str(WORK / "source.mp4"),
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", str(WORK / "source.mp4"),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(WORK / "audio16k.wav"),
    ], check=True)
    words, t = [], 0.5
    while t < DUR - 0.6:
        words.append({"w": f"word{len(words)}", "s": round(t, 2), "e": round(t + 0.25, 2)})
        t += 0.35
    transcript = {"language": "en", "duration": DUR, "segments": [
        {"start": 0.5, "end": DUR - 0.5, "text": " ".join(w["w"] for w in words), "words": words},
    ]}
    (WORK / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")
    (WORK / "scenes.json").write_text(json.dumps({"boundaries": [10.0]}), encoding="utf-8")


def fake_analysis(work_dir, t0, t1, transcript):
    n = int(round((t1 - t0) * FPS))
    fr = np.arange(n) / FPS
    # two "faces": left person drifting slightly, right person static
    boxes = {
        0: np.stack([0.30 + 0.01 * np.sin(fr), np.full(n, 0.45),
                     np.full(n, 0.12), np.full(n, 0.22)], axis=1).astype(np.float32),
        1: np.stack([np.full(n, 0.72), np.full(n, 0.48),
                     np.full(n, 0.11), np.full(n, 0.20)], axis=1).astype(np.float32),
    }
    visible = {0: np.ones(n, bool), 1: np.ones(n, bool)}
    f = lambda s: int(s * FPS)
    states = [
        {"f0": 0, "f1": f(8), "mode": "single", "tids": [0]},
        {"f0": f(8), "f1": f(14), "mode": "split", "tids": [0, 1]},
        {"f0": f(14), "f1": n, "mode": "single", "tids": [1]},
    ]
    return speaker.ClipAnalysis(n_frames=n, fps=FPS, boxes=boxes, visible=visible, states=states)


def test_windowing():
    words = [{"w": "x", "s": float(i), "e": i + 0.5} for i in range(0, 200)]
    segs = [{"start": float(i * 10), "end": float(i * 10 + 9.5),
             "text": "Sentence number %d ends properly." % i,
             "words": words[i * 10:(i + 1) * 10]} for i in range(20)]
    wins = _build_windows(segs)
    assert wins, "no windows built"
    for w in wins:
        d = w["end"] - w["start"]
        assert config.CLIP_MIN_S <= d <= config.CLIP_MAX_S, f"window {d}s out of range"
    ranked = [dict(w, score=100 - i) for i, w in enumerate(wins)]
    kept = _dedupe(ranked)
    for i, a in enumerate(kept):
        for b in kept[i + 1:]:
            inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
            union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
            assert inter / union <= 0.30 + 1e-9
    print(f"OK windowing: {len(wins)} windows, {len(kept)} after dedupe, all within 75-100s")


def test_render():
    make_source()
    orig = speaker.analyze_clip
    speaker.analyze_clip = fake_analysis
    try:
        clip = {"rank": 1, "start": 1.0, "end": 19.0, "title": "Synthetic Test Clip"}
        from app.hardware import detect
        hw = detect()
        res = renderer.render_clip(str(WORK), str(OUT), clip, hw.encoder, hw.encoder_args)
    finally:
        speaker.analyze_clip = orig
    expected = int(round(18.0 * FPS))
    assert res["frames"] == expected, f"{res['frames']} != {expected}"
    out_file = OUT / res["file"]
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height",
         "-of", "json", str(out_file)], capture_output=True, text=True, check=True)
    streams = json.loads(probe.stdout)["streams"]
    kinds = {s["codec_type"] for s in streams}
    assert "audio" in kinds and "video" in kinds
    v = next(s for s in streams if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (config.OUT_W, config.OUT_H)
    assert not (WORK / "clip_01").exists(), \
        "per-clip intermediates (audio.wav/captions.ass) not cleaned after success"
    print(f"OK render: {out_file.name} — {res['frames']} frames, 1080x1920, audio muxed (encoder={hw.encoder})")


def test_no_split_for_duplicate_tracks():
    """Two co-located tracks (same physical face detected twice) must NEVER
    produce a split state — regression test for the duplicated-person bug."""
    n, fps = 300, 30.0
    speech = np.ones(n, bool)
    act = {0: np.full(n, 0.05), 1: np.full(n, 0.045)}  # both clearly engaged
    vis = {0: np.ones(n, bool), 1: np.ones(n, bool)}
    same_spot = np.tile(np.array([0.5, 0.4, 0.14, 0.28], np.float32), (n, 1))
    offset = same_spot.copy()
    offset[:, 0] += 0.06  # centers 0.06 apart << 1.3 x width 0.14
    states = speaker._state_machine(act, vis, {0: same_spot, 1: offset}, n, fps)
    assert all(st["mode"] == "single" for st in states), f"false split: {states}"

    # control: genuinely distinct faces at the same activity levels DO split
    far = same_spot.copy()
    far[:, 0] = 0.85
    states2 = speaker._state_machine(act, vis, {0: same_spot, 1: far}, n, fps)
    assert any(st["mode"] == "split" for st in states2), "distinct faces failed to split"
    print("OK split distinctness: duplicates never split, distinct faces still do")


def test_dialogue_split_and_active_speaker_focus():
    """Rapid two-person dialogue uses a stable safe split, then returns to the
    clear active speaker. A third visible speaker can also take focus."""
    n, fps = 360, 30.0
    left = np.tile(np.array([0.20, 0.40, 0.12, 0.22], np.float32), (n, 1))
    right = np.tile(np.array([0.80, 0.40, 0.12, 0.22], np.float32), (n, 1))
    vis = {0: np.ones(n, bool), 1: np.ones(n, bool)}
    act0, act1 = np.zeros(n), np.zeros(n)
    for start, tid in ((0, 0), (45, 1), (90, 0), (135, 1)):
        (act0 if tid == 0 else act1)[start:start + 45] = 0.05
    act0[180:] = 0.05
    states = speaker._state_machine(
        {0: act0, 1: act1}, vis, {0: left, 1: right}, n, fps
    )
    state_at = lambda i: next(st for st in states if st["f0"] <= i < st["f1"])
    assert state_at(90)["mode"] == "split", f"rapid dialogue did not split: {states}"
    assert state_at(270)["mode"] == "single" and state_at(270)["tids"] == [0], \
        f"clear speaker did not regain full frame: {states}"

    third = np.tile(np.array([0.50, 0.40, 0.12, 0.22], np.float32), (n, 1))
    a0, a1, a2 = np.zeros(n), np.zeros(n), np.zeros(n)
    a0[:60], a1[:60], a2[120:] = 0.05, 0.04, 0.06
    states = speaker._state_machine(
        {0: a0, 1: a1, 2: a2},
        {0: np.ones(n, bool), 1: np.ones(n, bool), 2: np.ones(n, bool)},
        {0: left, 1: right, 2: third},
        n,
        fps,
    )
    state_at = lambda i: next(st for st in states if st["f0"] <= i < st["f1"])
    assert state_at(240)["mode"] == "single" and state_at(240)["tids"] == [2], \
        f"strongest visible speaker did not take focus: {states}"
    print("OK dialogue: safe back-and-forth splits; clear/third speaker gets full focus")


def test_render_timeout_drain():
    """A wedged worker must not hang a stage: _drain yields a TimeoutError for
    the unfinished item once the deadline passes, promptly (not after the
    worker's own 60s sleep), and _terminate_pool cleans it up."""
    import time as _t
    from concurrent.futures import ProcessPoolExecutor
    from app import jobs

    t0 = _t.perf_counter()
    with ProcessPoolExecutor(max_workers=1) as pool:
        futs = {pool.submit(_sleep_forever, 60.0): {"rank": 7}}
        outcomes = list(jobs._drain(futs, deadline=2.0))
        jobs._terminate_pool(pool)
    elapsed = _t.perf_counter() - t0
    assert len(outcomes) == 1, outcomes
    item, res, exc = outcomes[0]
    assert item == {"rank": 7} and res is None
    assert isinstance(exc, TimeoutError), f"expected TimeoutError, got {exc!r}"
    assert elapsed < 20, f"drain blocked {elapsed:.0f}s — did not honor deadline"
    print(f"OK render timeout: hung worker -> TimeoutError in {elapsed:.1f}s, not 60s")


def test_stranded_job_reaped_on_load():
    """A job left 'rendering' by a killed process is finalized to a clear error
    on startup (persisted to disk), not left spinning forever."""
    import time as _t
    from app import jobs

    jid = "_stranded_test"
    d = config.WORK_DIR / jid
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({
        "id": jid, "state": "rendering", "created": 0.0, "updated": 0.0,
        "meta": {"title": "Stuck"}, "candidates": [], "approved": [1],
        "stages": {"render_01": "running"},
    }), encoding="utf-8")
    old = _t.time() - (jobs.STALE_AFTER_S + 600)  # well past the stale window
    import os as _os
    _os.utime(d / "job.json", (old, old))
    _os.utime(d, (old, old))
    try:
        jobs.load_existing_jobs()
        on_disk = json.loads((d / "job.json").read_text(encoding="utf-8"))
        assert on_disk["state"] == "error", f"not reaped: {on_disk['state']}"
        assert "interrupted" in (on_disk["error"] or ""), on_disk.get("error")
        # and it is retryable (terminal state)
        assert on_disk["state"] in jobs.TERMINAL_STATES
    finally:
        jobs._jobs.pop(jid, None)
        jobs._locks.pop(jid, None)
        shutil.rmtree(d, ignore_errors=True)
    print("OK stranded reap: killed-mid-render job persisted as error on load")


def test_cancel_stops_running_pool():
    """Stop during a running stage: cancel() sets the flag, force-kills the live
    worker pool, and the drain/consume loop surfaces JobCancelled (-> the job
    thread finalizes 'cancelled', not 'error')."""
    import threading
    from concurrent.futures import ProcessPoolExecutor
    from app import jobs

    jid = "_cancel_test"
    jobs._jobs[jid] = {"id": jid, "state": "rendering", "created": 0.0,
                       "meta": {}, "candidates": [], "approved": [], "stages": {},
                       "timings": {}}
    jobs._locks[jid] = threading.Lock()
    jobs._owned.add(jid)
    try:
        with ProcessPoolExecutor(max_workers=1) as pool:
            jobs._active_pool[jid] = pool
            futs = {pool.submit(_sleep_forever, 60.0): {"rank": 1}}
            jobs.cancel(jid)  # user hits Stop
            assert jobs._is_cancelled(jid)
            assert jobs._jobs[jid]["state"] == "cancelling"
            raised = False
            try:
                for item, res, exc in jobs._drain(futs, deadline=30.0):
                    jobs._raise_if_cancelled(jid)
            except jobs.JobCancelled:
                raised = True
            finally:
                jobs._active_pool.pop(jid, None)
            assert raised, "Stop did not surface as JobCancelled"
    finally:
        shutil.rmtree(config.WORK_DIR / jid, ignore_errors=True)
        for reg in (jobs._jobs, jobs._locks, jobs._cancel, jobs._active_pool):
            reg.pop(jid, None)
        jobs._owned.discard(jid)
    print("OK cancel: Stop force-kills the running pool and raises JobCancelled")


def test_delete_and_clear_caches():
    """delete_job removes a finished job's cache + clips + memory (and refuses an
    active job); clear_caches removes terminal work caches but keeps running
    jobs and finished output clips."""
    from app import jobs

    orig_w, orig_o = config.WORK_DIR, config.OUTPUT_DIR
    base = config.WORK_DIR / "_maint_test"
    w, o = base / "work", base / "out"
    shutil.rmtree(base, ignore_errors=True)
    w.mkdir(parents=True); o.mkdir(parents=True)
    config.WORK_DIR, config.OUTPUT_DIR = w, o

    def mk(jid, state, slug=None):
        d = w / jid
        d.mkdir(parents=True)
        (d / "source.mp4").write_text("x" * 2000, encoding="utf-8")
        j = {"id": jid, "state": state, "created": 0.0}
        if slug:
            j["meta"] = {"slug": slug}
        (d / "job.json").write_text(json.dumps(j), encoding="utf-8")
        jobs._jobs[jid] = j
        jobs._owned.add(jid)
        return d

    made = ["d_done", "d_run", "d_del"]
    try:
        mk("d_done", "done")
        mk("d_run", "analyzing")           # active: must be kept
        dd = mk("d_del", "done", "delslug")
        od = o / "delslug"; od.mkdir(); (od / "clip.mp4").write_text("y", encoding="utf-8")

        # delete_job refuses an active job
        jobs._jobs["d_run"]["state"] = "analyzing"
        try:
            jobs.delete_job("d_run"); raise AssertionError("should refuse active job")
        except ValueError:
            pass

        # delete_job removes cache + clips + memory
        jobs.delete_job("d_del")
        assert not dd.exists() and not od.exists(), "delete left files behind"
        assert "d_del" not in jobs._jobs

        # clear_caches: terminal caches gone, running kept, no output touched
        r = jobs.clear_caches()
        assert not (w / "d_done").exists(), "terminal cache not cleared"
        assert (w / "d_run").exists(), "running cache wrongly cleared"
        assert r["cleared"] >= 1
    finally:
        config.WORK_DIR, config.OUTPUT_DIR = orig_w, orig_o
        for jid in made:
            jobs._jobs.pop(jid, None); jobs._owned.discard(jid)
        shutil.rmtree(base, ignore_errors=True)
    print("OK delete/clear: per-job delete + cache clear keep running jobs & output clips")


def test_low_ram_caps_render_workers():
    """Auto mode on a <10 GB box renders one clip at a time (OOM guard); an
    explicit RENDER_WORKERS still wins."""
    import unittest.mock as mock
    from app import hardware

    orig = (hardware._ram_gb, hardware._has_cuda, hardware._encoder_works, config.RENDER_WORKERS)
    try:
        hardware._has_cuda = lambda: False
        hardware._encoder_works = lambda n, e: n == "h264_qsv"
        hardware._ram_gb = lambda: 7.7
        config.RENDER_WORKERS = 0  # auto
        with mock.patch.object(hardware, "_cpu_threads", return_value=8):
            assert hardware.detect.__wrapped__().render_workers == 1
        config.RENDER_WORKERS = 3  # explicit opt-in overrides the cap
        with mock.patch.object(hardware, "_cpu_threads", return_value=8):
            assert hardware.detect.__wrapped__().render_workers == 3
    finally:
        hardware._ram_gb, hardware._has_cuda, hardware._encoder_works, config.RENDER_WORKERS = orig
    print("OK low-ram cap: auto->1 render worker under 10GB, explicit setting respected")


def test_drop_static_nonparticipant_faces():
    """A printed face on set (wall poster/TV still) is detected but never moves
    or talks — it must be dropped so it can't consume a track slot, while a
    real still-sitting speaker (tiny head motion + mouth variation) is kept."""
    from app.pipeline.speaker import Track, _drop_static_faces, SAMPLE_FPS

    n = int(4 * SAMPLE_FPS)
    poster = Track(tid=0)
    poster.ts = [i / SAMPLE_FPS for i in range(n)]
    poster.boxes = [[0.60, 0.05, 0.025, 0.03] for _ in range(n)]  # dead still + tiny
    poster.mouth = [0.02 + 0.01 * (i % 2) for i in range(n)]      # printed lips jitter

    person = Track(tid=1)
    person.ts = [i / SAMPLE_FPS for i in range(n)]
    # a mostly-still real speaker: small head motion, full-size face
    person.boxes = [[0.4 + 0.02 * np.sin(i / 5), 0.42, 0.08, 0.16] for i in range(n)]
    person.mouth = [0.03 + 0.02 * (i % 3) for i in range(n)]

    kept = _drop_static_faces([poster, person])
    ids = {t.tid for t in kept}
    assert ids == {1}, f"expected only the real person kept, got {ids}"

    # a big face that happens to be dead-still (frozen close-up) is NOT furniture
    frozen = Track(tid=2)
    frozen.ts = [i / SAMPLE_FPS for i in range(n)]
    frozen.boxes = [[0.5, 0.4, 0.09, 0.18] for _ in range(n)]  # still but full-size
    frozen.mouth = [0.02] * n
    assert {t.tid for t in _drop_static_faces([frozen, person])} == {2, 1}, \
        "a full-size still face must be kept (only small+static = furniture)"

    # safety: never return empty even if the only track looks static
    assert _drop_static_faces([poster]) == [poster], "must never drop the last track"
    print("OK static-face drop: small+static poster removed, real/large faces kept")


def test_no_split_when_crops_overlap():
    """Two REAL people sitting close together: distinct by the face-distance
    rule, but their projected split crops (~3x face height wide) would frame
    mostly the same region. The state machine must refuse the split (steady
    single — no split<->single flapping, each flip being a camera cut) —
    plan-time prevention of the QA flag 'crop rects overlap N% — same source
    region'. A clearly separated pair at identical activity must still split."""
    from app.pipeline.camera import split_rects_iou

    n, fps = 300, 30.0
    act = {0: np.full(n, 0.05), 1: np.full(n, 0.045)}  # both clearly engaged
    vis = {0: np.ones(n, bool), 1: np.ones(n, bool)}
    face = np.tile(np.array([0.38, 0.4, 0.10, 0.30], np.float32), (n, 1))
    close = face.copy()
    close[:, 0] = 0.515  # centers 0.135 apart > 1.3 x width 0.10 -> old rule splits
    assert 0.135 > 1.3 * 0.10, "fixture must pass the face-distance rule"
    assert split_rects_iou(face[0], close[0], 1920, 1080) > 0.5, \
        "fixture must actually produce heavily overlapping split crops"
    states = speaker._state_machine(act, vis, {0: face, 1: close}, n, fps, 1920, 1080)
    assert all(st["mode"] == "single" for st in states), \
        f"split allowed despite overlapping crops: {states}"
    assert len(states) <= 2, f"state flapping instead of steady single: {states}"

    far = face.copy()
    far[:, 0] = 0.85  # genuinely separated pair
    assert split_rects_iou(face[0], far[0], 1920, 1080) <= speaker.SPLIT_MAX_CROP_IOU
    states2 = speaker._state_machine(act, vis, {0: face, 1: far}, n, fps, 1920, 1080)
    assert any(st["mode"] == "split" for st in states2), "separated pair failed to split"
    print("OK crop-overlap gate: close pair stays steady single, separated pair still splits")


def test_target_scaling():
    """Clip-count target scales with duration (~0.4/min), floors for short
    videos, ceilings for absurd ones, and follows CLIPS_PER_MINUTE from .env."""
    assert target_clip_count(30 * 60) == 12
    assert target_clip_count(70 * 60) == 28
    assert target_clip_count(120 * 60) == 48
    assert target_clip_count(6 * 60) == config.MIN_CANDIDATES  # short-video floor
    assert target_clip_count(10 * 3600) == config.MAX_CANDIDATES  # sanity ceiling
    orig = config.CLIPS_PER_MINUTE
    try:
        config.CLIPS_PER_MINUTE = 0.8
        assert target_clip_count(30 * 60) == 24, "multiplier is not tunable"
    finally:
        config.CLIPS_PER_MINUTE = orig
    print("OK target scaling: 30min=12, 70min=28, 120min=48, floor+ceiling+env knob")


def test_windowing_capacity_for_large_targets():
    """A 120-min transcript must yield enough non-overlapping windows to fill
    the top of the scaled range (>=48) with the current seed/dedupe settings."""
    words = [{"w": "x", "s": float(i * 2), "e": i * 2 + 1.0} for i in range(3600)]
    segs = [{"start": float(i * 10), "end": float(i * 10 + 9.5),
             "text": "Sentence number %d ends properly." % i,
             "words": words[i * 5:(i + 1) * 5]} for i in range(720)]
    wins = _build_windows(segs)
    ranked = [dict(w, score=100 - 0.01 * i) for i, w in enumerate(wins)]
    kept = _dedupe(ranked)
    assert len(kept) >= 48, f"windowing capacity too low for 120min: {len(kept)}"
    print(f"OK capacity: 120min transcript -> {len(wins)} windows, {len(kept)} after dedupe (>=48)")


def test_gap_interpolation_holds():
    """A short detection gap is bridged linearly; a long gap HOLDS the last
    known box instead of panning toward the next sighting (which framed empty
    background) — regression test for the lost-speaker bug."""
    fps = 30.0
    tr = speaker.Track(tid=0)

    def add(t, x):
        tr.ts.append(round(t, 3))
        tr.boxes.append([x, 0.4, 0.12, 0.22])
        tr.mouth.append(0.02)

    for t in np.arange(0.0, 2.01, 0.1):
        add(t, 0.30)
    for t in np.arange(2.6, 3.01, 0.1):   # 0.6s gap: bridge
        add(t, 0.32)
    for t in np.arange(6.0, 8.01, 0.1):   # 3.0s gap: hold
        add(t, 0.80)
    n = int(8.0 * fps)
    boxes, _vis = speaker._interpolate([tr], n, fps)
    f = lambda s: int(s * fps)
    mid_short = boxes[0][f(2.3), 0]
    assert 0.30 - 1e-4 <= mid_short <= 0.32 + 1e-4, f"short gap not bridged: {mid_short}"
    gap_x = boxes[0][f(3.2):f(5.9), 0]
    assert np.allclose(gap_x, 0.32, atol=1e-3), f"long gap panned instead of holding: {gap_x[::20]}"
    print("OK gap interp: short gaps bridge, long gaps hold last known box")


def test_single_track_grace():
    """Lone-speaker state machine: brief invisibility holds the speaker,
    only sustained absence (> grace) goes to the wide fallback."""
    n, fps = 450, 30.0
    f = lambda s: int(s * fps)
    act = {0: np.full(n, 0.05)}
    vis = np.ones(n, bool)
    vis[f(3):int(3.8 * fps)] = False   # 0.8s dropout: within grace
    vis[f(8):f(12)] = False            # 4s absence: beyond grace
    boxes = {0: np.tile(np.array([0.7, 0.4, 0.12, 0.22], np.float32), (n, 1))}
    states = speaker._state_machine(act, {0: vis}, boxes, n, fps)

    def tids_at(i):
        return next(st["tids"] for st in states if st["f0"] <= i < st["f1"])

    assert tids_at(f(3.4)) == [0], f"short dropout lost the speaker: {states}"
    assert tids_at(f(10)) == [], f"sustained absence did not go wide: {states}"
    assert tids_at(f(13)) == [0], f"speaker not re-acquired: {states}"
    print("OK grace: 0.8s dropout held, 4s absence went wide, then re-acquired")


def test_center_fallback_is_smooth():
    """When the face is genuinely absent, the camera must EASE to the wide
    center crop under the velocity clamp — never hard-snap."""
    from app.pipeline import camera

    n, fps = 450, 30.0
    src_w, src_h = 1280, 720
    f = lambda s: int(s * fps)
    boxes = {0: np.tile(np.array([0.75, 0.4, 0.12, 0.22], np.float32), (n, 1))}
    states = [
        {"f0": 0, "f1": f(5), "mode": "single", "tids": [0]},
        {"f0": f(5), "f1": f(11), "mode": "single", "tids": []},
        {"f0": f(11), "f1": n, "mode": "single", "tids": [0]},
    ]
    analysis = speaker.ClipAnalysis(n_frames=n, fps=fps, boxes=boxes,
                                    visible={0: np.ones(n, bool)}, states=states)
    plans = camera.plan(analysis, src_w, src_h, [])
    ctr = [p.rects[0][0] + p.rects[0][2] / 2 for p in plans]
    vmax = camera.VMAX_FRAC * src_w
    for i in range(f(5) - 1, f(11) - 1):
        d = abs(ctr[i + 1] - ctr[i])
        assert d <= vmax + 1e-3, f"snap at frame {i}: center jumped {d:.1f}px (vmax {vmax:.1f})"
    assert abs(ctr[f(10.5)] - src_w / 2) < 0.1 * src_w, "never reached the wide center fallback"
    print("OK fallback: absence eases to center under velocity clamp, no snap")


def test_coarse_face_safety_lock():
    """A stale/missing speaker box must not frame empty background. The camera
    uses the independent coarse detector, including immediately after a cut."""
    from app.pipeline import camera

    n, fps = 90, 30.0
    src_w, src_h = 1280, 720
    stale = np.tile(np.array([0.25, 0.40, 0.05, 0.10], np.float32), (n, 1))
    coarse = [[[0.75, 0.40, 0.18, 0.32, 0.9]] for _ in range(n)]
    states = [{"f0": 0, "f1": n, "mode": "single", "tids": [0]}]

    missing = speaker.ClipAnalysis(
        n_frames=n,
        fps=fps,
        boxes={0: stale},
        visible={0: np.zeros(n, bool)},
        states=states,
        coarse_faces=coarse,
    )
    plans = camera.plan(missing, src_w, src_h, [])
    center = plans[0].rects[0][0] + plans[0].rects[0][2] / 2
    assert center > 0.65 * src_w, f"missing track ignored coarse face: center={center}"

    cut_stale = speaker.ClipAnalysis(
        n_frames=n,
        fps=fps,
        boxes={0: stale},
        visible={0: np.ones(n, bool)},
        states=states,
        coarse_faces=coarse,
    )
    plans = camera.plan(cut_stale, src_w, src_h, [30])
    before = plans[29].rects[0][0] + plans[29].rects[0][2] / 2
    after = plans[30].rects[0][0] + plans[30].rects[0][2] / 2
    assert before < 0.35 * src_w, f"pre-cut tracker unexpectedly moved: {before}"
    assert after > 0.65 * src_w, f"cut did not reacquire lone coarse face: {after}"
    print("OK coarse safety: missing/stale speaker crop reacquires the real face")


def test_multi_track_blackout_goes_wide():
    """Two-person conversation, then BOTH tracks vanish for 10s (total
    detection blackout): the state machine must go wide (tids=[]), not hold a
    stale close-up framing empty background."""
    n, fps = 600, 30.0
    f = lambda s: int(s * fps)
    act = {0: np.full(n, 0.05), 1: np.full(n, 0.04)}
    for t in act.values():
        t[f(5):f(15)] = 0.0  # activity is visibility-gated in the real pipeline
    vis0, vis1 = np.ones(n, bool), np.ones(n, bool)
    vis0[f(5):f(15)] = False
    vis1[f(5):f(15)] = False
    boxes = {0: np.tile(np.array([0.30, 0.4, 0.12, 0.22], np.float32), (n, 1)),
             1: np.tile(np.array([0.75, 0.4, 0.12, 0.22], np.float32), (n, 1))}
    states = speaker._state_machine(act, {0: vis0, 1: vis1}, boxes, n, fps)
    tids_at = lambda i: next(st["tids"] for st in states if st["f0"] <= i < st["f1"])
    assert tids_at(f(10)) == [], f"blackout held a stale close-up: {states}"
    assert tids_at(f(2)) != [] and tids_at(f(18)) != [], f"faces lost outside blackout: {states}"
    print("OK blackout: sustained all-track absence goes wide, not stale close-up")


def test_blackout_hands_off_to_visible_track():
    """The conversation pair vanishes (camera angle change) but a third track
    IS visible — frame that track instead of background."""
    n, fps = 600, 30.0
    f = lambda s: int(s * fps)
    act = {0: np.full(n, 0.05), 1: np.full(n, 0.04), 2: np.zeros(n)}
    act[0][f(5):f(15)] = 0.0
    act[1][f(5):f(15)] = 0.0
    act[2][f(5):f(15)] = 0.03
    vis0, vis1, vis2 = np.ones(n, bool), np.ones(n, bool), np.zeros(n, bool)
    vis0[f(5):f(15)] = False
    vis1[f(5):f(15)] = False
    vis2[f(5):f(15)] = True
    boxes = {0: np.tile(np.array([0.30, 0.4, 0.12, 0.22], np.float32), (n, 1)),
             1: np.tile(np.array([0.75, 0.4, 0.12, 0.22], np.float32), (n, 1)),
             2: np.tile(np.array([0.50, 0.4, 0.06, 0.11], np.float32), (n, 1))}
    states = speaker._state_machine(act, {0: vis0, 1: vis1, 2: vis2}, boxes, n, fps)
    tids_at = lambda i: next(st["tids"] for st in states if st["f0"] <= i < st["f1"])
    assert tids_at(f(10)) == [2], f"did not hand off to the visible track: {states}"
    print("OK handoff: conversation pair invisible -> frames the visible track")


def test_fragment_merge():
    """Two fragments of one person split by a dropout re-join into one track;
    a genuinely different position stays separate."""
    def mk(tid, t0, x, n=10):
        tr = speaker.Track(tid=tid)
        for k in range(n):
            tr.ts.append(round(t0 + 0.1 * k, 3))
            tr.boxes.append([x, 0.4, 0.12, 0.22])
            tr.mouth.append(0.02)
        return tr

    a = mk(0, 0.0, 0.50)    # ends at 0.9s
    b = mk(1, 1.9, 0.52)    # 1s gap, same spot -> merge into a
    c = mk(2, 1.9, 0.90)    # far away -> stays its own track
    merged = speaker._merge_fragments([a, b, c])
    assert len(merged) == 2, f"expected 2 tracks after merge, got {len(merged)}"
    assert max(len(tr.ts) for tr in merged) == 20, "fragments were not joined"
    print("OK fragment merge: dropout fragments re-join, distinct faces stay separate")


def test_caption_no_overlap():
    """Whisper word timings that overlap across line boundaries must not produce
    overlapping ASS events (doubled on-screen text)."""
    from app.pipeline import captions, qa

    words = []
    t = 0.0
    for i in range(24):
        # every 4th word overlaps the previous word's range by 0.2s
        s = t - 0.2 if i % 4 == 0 and i else t
        words.append({"w": f"word{i}.", "s": round(s, 2), "e": round(s + 0.4, 2)})
        t = s + 0.35
    transcript = {"segments": [{"start": 0, "end": t, "text": "x", "words": words}]}
    WORK.mkdir(parents=True, exist_ok=True)
    ass = captions.build_ass(transcript, 0.0, t + 1, WORK / "overlap_test.ass")
    flags = qa._check_ass_overlaps(ass)
    assert not flags, f"caption overlaps survived: {flags}"
    print("OK captions: overlapping word timings -> zero overlapping ASS events")


def test_disk_emergency_cleanup():
    """Disk pressure deletes oldest deletable work caches ONLY: terminal jobs
    and crash-stale 'rendering' dirs go; live/awaiting jobs and anything with a
    non-terminal in-memory state survive; output/ is never scanned. If nothing
    deletable remains and space is still low, check() raises DiskSpaceError."""
    import os
    import time as _time

    from app import diskspace, jobs

    base = WORK / "_disk_test"
    shutil.rmtree(base, ignore_errors=True)

    def mk(name, state, old=False):
        d = base / name
        d.mkdir(parents=True)
        (d / "job.json").write_text(json.dumps({"state": state}), encoding="utf-8")
        (d / "source.mp4").write_text("x", encoding="utf-8")
        if old:
            t = _time.time() - 3600  # > STALE_AFTER_S (900s)
            for p in (*d.iterdir(), d):
                os.utime(p, (t, t))
        return d

    done_old = mk("a_done_old", "done", old=True)
    done_new = mk("b_done_new", "done")
    stale_crash = mk("c_stale_crash", "rendering", old=True)   # crashed server
    live_render = mk("d_live_render", "rendering")             # recent activity
    waiting = mk("e_waiting", "awaiting_approval", old=True)
    memblocked = mk("f_memblocked", "done", old=True)          # disk says done...
    jobs._jobs["f_memblocked"] = {"state": "rendering"}        # ...memory says live

    orig = (config.WORK_DIR, config.MIN_FREE_GB, diskspace.free_gb)
    deleted_counter = {"n": 0}

    def fake_free_gb(path):
        # 1 GB free until two dirs are gone, then plenty
        return 1.0 if deleted_counter["n"] < 2 else 50.0

    real_rmtree = shutil.rmtree

    def counting_rmtree(p, **kw):
        deleted_counter["n"] += 1
        real_rmtree(p, **kw)

    config.WORK_DIR, config.MIN_FREE_GB = base, 5.0
    diskspace.free_gb = fake_free_gb
    diskspace.shutil.rmtree = counting_rmtree
    try:
        gone = diskspace.emergency_cleanup()
        assert not done_old.exists() and not stale_crash.exists(), \
            f"oldest deletable dirs not removed: {gone}"
        assert done_new.exists(), "newer cache deleted before space recovered check"
        assert live_render.exists(), "ACTIVE render dir deleted"
        assert waiting.exists(), "awaiting_approval dir deleted — data loss"
        assert memblocked.exists(), "in-memory non-terminal job's dir deleted"

        # nothing deletable left + still low -> DiskSpaceError from check()
        deleted_counter["n"] = -100  # free stays 1.0 forever
        try:
            diskspace.check("test")
            raise AssertionError("check() did not raise with no reclaimable space")
        except diskspace.DiskSpaceError as e:
            assert "free disk space" in str(e).lower() or "GB" in str(e)
    finally:
        config.WORK_DIR, config.MIN_FREE_GB, diskspace.free_gb = orig
        diskspace.shutil.rmtree = real_rmtree
        jobs._jobs.pop("f_memblocked", None)
        real_rmtree(base, ignore_errors=True)
    print("OK disk cleanup: oldest terminal+stale deleted, live/waiting/memory-live "
          "kept, DiskSpaceError when nothing reclaimable")


def test_disk_check_blocks_render():
    """A render on a full disk must fail the job up front with the explicit
    disk message — not die cryptically inside ffmpeg."""
    import threading

    from app import diskspace, jobs

    jid = "_disktest99"
    jobs._jobs[jid] = {"id": jid, "state": "render_queued", "created": 0.0,
                       "stages": {}, "timings": {}, "meta": {"slug": "x"},
                       "candidates": [], "approved": [], "clips": [], "error": None}
    jobs._locks[jid] = threading.Lock()
    orig = (diskspace.free_gb, diskspace.emergency_cleanup, config.MIN_FREE_GB)
    diskspace.free_gb = lambda path: 0.2
    diskspace.emergency_cleanup = lambda: []
    config.MIN_FREE_GB = 5.0
    try:
        jobs._run_render_impl(jid)
        job = jobs._jobs[jid]
        assert job["state"] == "error", job["state"]
        assert "GB" in (job["error"] or ""), f"no disk message: {job['error']}"
    finally:
        diskspace.free_gb, diskspace.emergency_cleanup, config.MIN_FREE_GB = orig
        shutil.rmtree(config.WORK_DIR / jid, ignore_errors=True)
        jobs._jobs.pop(jid, None)
        jobs._locks.pop(jid, None)
    print("OK disk gate: full disk fails the render job fast with an explicit message")


def test_atomic_job_save_and_part_sweep():
    """job.json writes are atomic (no .tmp survivors, always valid JSON) and
    orphaned model .part temps get swept."""
    import threading

    from app import diskspace, jobs

    jid = "_atomictest1"
    jobs._jobs[jid] = {"id": jid, "state": "queued", "created": 0.0}
    jobs._locks[jid] = threading.Lock()
    try:
        jobs._save(jobs._jobs[jid])
        wd = config.WORK_DIR / jid
        assert not (wd / "job.json.tmp").exists(), "tmp file left behind"
        assert json.loads((wd / "job.json").read_text(encoding="utf-8"))["id"] == jid
    finally:
        shutil.rmtree(config.WORK_DIR / jid, ignore_errors=True)
        jobs._jobs.pop(jid, None)
        jobs._locks.pop(jid, None)

    mdir = WORK / "_parts_test"
    shutil.rmtree(mdir, ignore_errors=True)
    mdir.mkdir(parents=True)
    (mdir / ".model.abc.part").write_text("partial", encoding="utf-8")
    (mdir / "real_model.bin").write_text("keep", encoding="utf-8")
    removed = diskspace.sweep_orphan_parts(mdir)
    assert len(removed) == 1 and not (mdir / ".model.abc.part").exists()
    assert (mdir / "real_model.bin").exists(), "non-part file swept!"
    shutil.rmtree(mdir, ignore_errors=True)
    print("OK atomic save + part sweep: no tmp survivors, valid JSON, only .part removed")


def test_render_pipe_failure_not_masked():
    """Regression test for the "flush of closed file" bug: if ffmpeg exits
    immediately (e.g. it can't open the requested encoder — the real-world
    trigger is too many concurrent h264_nvenc sessions exceeding the GPU's
    cap), the very first proc.stdin.write() raises BrokenPipeError. The old
    code let that exception hit `finally: proc.stdin.close()` unguarded,
    which re-raised as `ValueError: flush of closed file` and swallowed
    ffmpeg's real stderr. render_clip must now surface one RuntimeError
    carrying ffmpeg's actual message instead."""
    make_source()
    orig = speaker.analyze_clip
    speaker.analyze_clip = fake_analysis
    try:
        clip = {"rank": 99, "start": 1.0, "end": 19.0, "title": "Pipe Failure Test"}
        try:
            renderer.render_clip(str(WORK), str(OUT), clip, "not_a_real_encoder", [])
            raise AssertionError("expected render_clip to raise on a bogus encoder")
        except RuntimeError as e:
            msg = str(e)
            assert "flush of closed file" not in msg.lower(), \
                f"masking bug regressed: {msg}"
            assert "ffmpeg encode failed" in msg, f"unexpected error shape: {msg}"
        partials = list(OUT.glob("99_*.mp4"))
        assert not partials, f"partial MP4 left in output after failed render: {partials}"
    finally:
        speaker.analyze_clip = orig
    print("OK pipe failure: dead-encoder crash surfaces ffmpeg's real error, "
          "not a masked 'flush of closed file'")


def test_posix_communicate_flush_not_masked():
    """The 8b92270 fix guarded OUR stdin.close(), but on the Linux server the
    identical error came back: CPython's POSIX _communicate() flushes
    proc.stdin ITSELF, guarded only for BrokenPipeError (subprocess.py ~2067,
    3.12) — flushing the already-closed pipe raises the same
    ValueError('flush of closed file') from inside communicate(). Because
    close-then-communicate is the renderer's normal EOF signal, on Linux this
    fired on EVERY clip, successful encodes included. Windows' communicate()
    never flushes (close only, idempotent), which is why the laptop test suite
    stayed green. This test grafts the POSIX stdin block onto communicate() so
    the Linux behavior runs on Windows: a full happy-path render must succeed,
    and a dead encoder must still surface ffmpeg's real stderr."""
    import subprocess as sp

    class PosixFlushPopen(sp.Popen):
        def communicate(self, *args, **kwargs):
            # verbatim shape of CPython 3.12 POSIX _communicate stdin handling
            if self.stdin and not self._communication_started:
                try:
                    self.stdin.flush()
                except BrokenPipeError:
                    pass
            return super().communicate(*args, **kwargs)

    make_source()
    orig_analyze = speaker.analyze_clip
    speaker.analyze_clip = fake_analysis
    orig_popen = sp.Popen
    sp.Popen = PosixFlushPopen  # renderer resolves subprocess.Popen at call time
    try:
        from app.hardware import detect
        hw = detect()
        clip = {"rank": 98, "start": 1.0, "end": 7.0, "title": "Posix Happy Path"}
        res = renderer.render_clip(str(WORK), str(OUT), clip, hw.encoder, hw.encoder_args)
        assert res["frames"] == int(round(6.0 * FPS)), res

        clip = {"rank": 97, "start": 1.0, "end": 7.0, "title": "Posix Dead Encoder"}
        try:
            renderer.render_clip(str(WORK), str(OUT), clip, "not_a_real_encoder", [])
            raise AssertionError("expected render_clip to raise on a bogus encoder")
        except RuntimeError as e:
            assert "flush of closed file" not in str(e).lower(), \
                f"POSIX communicate() flush still masks the real error: {e}"
            assert "ffmpeg encode failed" in str(e), f"unexpected error shape: {e}"
    finally:
        sp.Popen = orig_popen
        speaker.analyze_clip = orig_analyze
    print("OK posix communicate: happy path + dead encoder both survive the "
          "stdin flush inside POSIX communicate() (Linux server behavior)")


def test_nvenc_session_cap():
    """render_workers must never exceed NVENC_MAX_SESSIONS when nvenc is
    selected, even on a many-core box that would otherwise auto-scale far
    past what the GPU's concurrent-session limit allows — regression test
    for the root cause behind uniform render failures on a fresh GPU server."""
    from app import hardware

    orig = (hardware._has_cuda, hardware._encoder_works, hardware._ram_gb,
            config.RENDER_WORKERS, config.NVENC_MAX_SESSIONS)
    try:
        config.RENDER_WORKERS = 0  # auto
        config.NVENC_MAX_SESSIONS = 3
        hardware._has_cuda = lambda: True
        hardware._encoder_works = lambda name, extra: name == "h264_nvenc"
        hardware._ram_gb = lambda: 64.0  # a 64-core GPU server has ample RAM
        import unittest.mock as mock
        with mock.patch.object(hardware, "_cpu_threads", return_value=64):
            hw = hardware.detect.__wrapped__()
        assert hw.encoder == "h264_nvenc", hw
        assert hw.render_workers == 3, \
            f"nvenc render_workers not capped: {hw.render_workers} (64-core auto would be 16)"
    finally:
        (hardware._has_cuda, hardware._encoder_works, hardware._ram_gb,
         config.RENDER_WORKERS, config.NVENC_MAX_SESSIONS) = orig
    print("OK nvenc cap: 64-core auto-scaling clamped to NVENC_MAX_SESSIONS=3")


def test_gpu_hardware_selection():
    """On a CUDA machine detect() must pair device=cuda with float16 + NVENC;
    without CUDA it must fall back to cpu/int8. Contract test only — probes are
    mocked, real GPU execution is unverifiable on this hardware."""
    from app import hardware

    orig = (hardware._has_cuda, hardware._encoder_works,
            config.WHISPER_MODEL, config.WHISPER_COMPUTE)
    try:
        config.WHISPER_MODEL, config.WHISPER_COMPUTE = "auto", "auto"
        hardware._has_cuda = lambda: True
        hardware._encoder_works = lambda name, extra: name == "h264_nvenc"
        hw = hardware.detect.__wrapped__()  # bypass lru_cache
        assert hw.whisper_device == "cuda", hw
        assert hw.whisper_compute == "float16", hw
        assert hw.encoder == "h264_nvenc", hw
        assert hw.whisper_model in ("large-v3", "medium"), hw
        # nvenc's -cq is SILENTLY IGNORED without -rc vbr and -b:v 0 (falls
        # back to ~2 Mbps default = blurry 1080x1920). Lock the full triple in.
        a = hw.encoder_args
        assert "-cq" in a and "-rc" in a and a[a.index("-rc") + 1] == "vbr" \
            and "-b:v" in a and a[a.index("-b:v") + 1] == "0", \
            f"nvenc CQ mode not actually enabled: {a}"

        hardware._has_cuda = lambda: False
        hardware._encoder_works = lambda name, extra: name == "h264_qsv"
        hw = hardware.detect.__wrapped__()
        assert hw.encoder == "h264_qsv", hw
        assert "-global_quality" in hw.encoder_args and \
            hw.encoder_args[hw.encoder_args.index("-global_quality") + 1] == str(config.VIDEO_QUALITY), \
            f"qsv quality not driven by VIDEO_QUALITY: {hw.encoder_args}"
        assert config.VIDEO_QUALITY <= 20, \
            f"VIDEO_QUALITY default regressed above the measured-blurry range: {config.VIDEO_QUALITY}"

        hardware._encoder_works = lambda name, extra: False
        hw = hardware.detect.__wrapped__()
        assert hw.whisper_device == "cpu" and hw.whisper_compute == "int8", hw
        assert hw.encoder == "libx264", hw
    finally:
        (hardware._has_cuda, hardware._encoder_works,
         config.WHISPER_MODEL, config.WHISPER_COMPUTE) = orig
    print("OK gpu selection: cuda -> cuda/float16/nvenc, no cuda -> cpu/int8/x264")


def test_source_height_cap_and_analysis_workers():
    """A CPU-only box can't decode 4K AV1 (it hung Stage 2), so detect() caps
    max_source_height to 1080 without CUDA and keeps the configured value with
    it; stage2_workers drops to 1 under 10 GB unless ANALYSIS_WORKERS overrides."""
    import unittest.mock as mock
    from app import hardware

    orig = (hardware._ram_gb, hardware._has_cuda, hardware._encoder_works,
            config.MAX_SOURCE_HEIGHT, config.ANALYSIS_WORKERS)
    try:
        hardware._encoder_works = lambda n, e: n == "h264_qsv"
        config.MAX_SOURCE_HEIGHT = 2160
        config.ANALYSIS_WORKERS = 0  # auto

        # CPU-only, low RAM: 4K request capped to 1080, analysis isolated.
        hardware._has_cuda = lambda: False
        hardware._ram_gb = lambda: 7.7
        with mock.patch.object(hardware, "_cpu_threads", return_value=8):
            hw = hardware.detect.__wrapped__()
        assert hw.max_source_height == 1080, hw.max_source_height
        assert hw.stage2_workers == 1, hw.stage2_workers

        # CUDA box: keeps the configured 2160 (NVDEC decodes AV1) and 3 workers.
        hardware._has_cuda = lambda: True
        hardware._encoder_works = lambda n, e: n == "h264_nvenc"
        hardware._ram_gb = lambda: 32.0
        with mock.patch.object(hardware, "_cpu_threads", return_value=16):
            hw = hardware.detect.__wrapped__()
        assert hw.max_source_height == 2160, hw.max_source_height
        assert hw.stage2_workers == 3, hw.stage2_workers

        # Explicit ANALYSIS_WORKERS wins even on a low-RAM CPU box.
        config.ANALYSIS_WORKERS = 3
        hardware._has_cuda = lambda: False
        hardware._encoder_works = lambda n, e: n == "h264_qsv"
        hardware._ram_gb = lambda: 7.7
        with mock.patch.object(hardware, "_cpu_threads", return_value=8):
            hw = hardware.detect.__wrapped__()
        assert hw.stage2_workers == 3, hw.stage2_workers
    finally:
        (hardware._ram_gb, hardware._has_cuda, hardware._encoder_works,
         config.MAX_SOURCE_HEIGHT, config.ANALYSIS_WORKERS) = orig
    print("OK source cap: CPU-only 4K->1080, CUDA keeps 2160; analysis isolated under 10GB, explicit wins")


def test_transcribe_receives_device():
    """The device chosen by hardware.detect() must actually reach transcribe()
    — regression test for whisper silently running on CPU on GPU machines."""
    from app import jobs
    from app.pipeline import transcribe as tmod

    seen = {}

    def fake(work_dir, model_size, compute_type, cpu_threads, device="cpu",
             progress_cb=None):
        seen["device"] = device
        return {}

    d = WORK / "device_probe"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    orig = tmod.transcribe
    tmod.transcribe = fake
    try:
        jobs._task_transcribe(str(d), "small", "float16", 4, "cuda")
    finally:
        tmod.transcribe = orig
    assert seen.get("device") == "cuda", f"device not plumbed through: {seen}"
    print("OK device plumbing: _task_transcribe passes device='cuda' to transcribe()")


def test_concurrency_cap():
    """With MAX_CONCURRENT_JOBS=1, three submitted jobs must run their pipeline
    strictly one at a time (extras block in 'queued')."""
    import threading
    import time as _time

    from app import jobs

    counter = {"cur": 0, "max": 0}
    lk = threading.Lock()

    def fake_impl(job_id):
        with lk:
            counter["cur"] += 1
            counter["max"] = max(counter["max"], counter["cur"])
        _time.sleep(0.3)
        with lk:
            counter["cur"] -= 1
        jobs._update(job_id, state="done")

    orig_impl, orig_sem = jobs._run_analysis_impl, jobs._pipeline_sem
    jobs._run_analysis_impl = fake_impl
    jobs._pipeline_sem = threading.BoundedSemaphore(1)
    created = []
    try:
        for _ in range(3):
            created.append(jobs.create_job("https://example.test/fake")["id"])
        deadline = _time.time() + 15
        while _time.time() < deadline and not all(
                jobs._jobs[j]["state"] == "done" for j in created):
            _time.sleep(0.05)
    finally:
        jobs._run_analysis_impl, jobs._pipeline_sem = orig_impl, orig_sem
        for jid in created:
            shutil.rmtree(config.WORK_DIR / jid, ignore_errors=True)
    assert all(jobs._jobs[j]["state"] == "done" for j in created), "jobs never finished"
    assert counter["max"] == 1, f"{counter['max']} pipelines ran concurrently (cap 1)"
    print("OK concurrency cap: 3 jobs, max 1 pipeline running at a time")


def test_unique_output_path():
    """Existing output files must never be overwritten — suffix instead."""
    d = WORK / "uniq"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    p1 = renderer._unique_path(d, "01_clip", ".mp4")
    assert p1.name == "01_clip.mp4"
    p1.write_text("original", encoding="utf-8")
    p2 = renderer._unique_path(d, "01_clip", ".mp4")
    assert p2.name == "01_clip_2.mp4", p2.name
    p2.write_text("second", encoding="utf-8")
    p3 = renderer._unique_path(d, "01_clip", ".mp4")
    assert p3.name == "01_clip_3.mp4", p3.name
    assert p1.read_text(encoding="utf-8") == "original", "first render was clobbered"
    print("OK unique paths: collision -> _2, _3; originals untouched")


def test_retention_cleanup():
    """Old terminal/orphaned work dirs and old output dirs are pruned; running,
    awaiting-approval, and fresh dirs survive; days=0 deletes nothing."""
    import os
    import time as _time

    from app import retention

    base, outbase = WORK / "_retention_work", OUT / "_retention_out"

    def mk(root, name, state=None, old=False):
        d = root / name
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        if state:
            (d / "job.json").write_text(json.dumps({"state": state}), encoding="utf-8")
        else:
            (d / "clip.mp4").write_text("x", encoding="utf-8")
        if old:
            t = _time.time() - 30 * 86400
            for p in (*d.iterdir(), d):
                os.utime(p, (t, t))
        return d

    old_done = mk(base, "old_done", "done", old=True)
    old_wait = mk(base, "old_wait", "awaiting_approval", old=True)
    fresh = mk(base, "fresh_done", "done")
    orphan = mk(base, "old_orphan", None, old=True)
    out_old = mk(outbase, "old_slug", None, old=True)
    out_new = mk(outbase, "new_slug", None)
    try:
        assert retention.cleanup_old(0, base, outbase) == [], "days=0 must be a no-op"
        deleted = retention.cleanup_old(7, base, outbase)
        assert not old_done.exists(), "old terminal job not deleted"
        assert not orphan.exists(), "orphaned dir not deleted"
        assert not out_old.exists(), "old output dir not deleted"
        assert old_wait.exists(), "awaiting_approval job was DELETED — data loss"
        assert fresh.exists() and out_new.exists(), "fresh dirs were deleted"
        assert len(deleted) == 3, deleted
    finally:
        shutil.rmtree(base, ignore_errors=True)
        shutil.rmtree(outbase, ignore_errors=True)
    print("OK retention: old terminal+orphan+output pruned; waiting/fresh kept; 0=off")


def test_auth_gate():
    """Login wall through the real ASGI middleware: redirect/401 when logged
    out, cookie session after login, tampered/expired token rejection, logout,
    and the no-credentials dev path staying fully open."""
    from fastapi.testclient import TestClient

    from app import auth
    from app.main import app

    orig = (config.ADMIN_USERNAME, config.ADMIN_PASSWORD, config.ADMIN_PASSWORD_SHA256)
    config.ADMIN_USERNAME, config.ADMIN_PASSWORD, config.ADMIN_PASSWORD_SHA256 = \
        "sk", "test-secret-pw", ""
    try:
        c = TestClient(app, follow_redirects=False)
        r = c.get("/")
        assert r.status_code == 302 and r.headers["location"] == "/login", r.status_code
        assert c.get("/api/jobs").status_code == 401
        assert c.get("/api/hardware").status_code == 401
        assert c.get("/clips/x/y.mp4").status_code == 302, "clips mount not gated"
        assert c.get("/login").status_code == 200, "login page not public"

        r = c.post("/api/login", json={"username": "sk", "password": "nope"})
        assert r.status_code == 401 and auth.COOKIE_NAME not in r.cookies
        assert "test-secret-pw" not in r.text, "password echoed in response"

        r = c.post("/api/login", json={"username": "sk", "password": "test-secret-pw"})
        assert r.status_code == 200 and auth.COOKIE_NAME in r.cookies
        assert c.get("/api/jobs").status_code == 200, "cookie session not accepted"
        assert c.get("/").status_code == 200

        c2 = TestClient(app, follow_redirects=False)
        c2.cookies.set(auth.COOKIE_NAME, "9999999999.deadbeef")
        assert c2.get("/api/jobs").status_code == 401, "tampered token accepted"
        c2.cookies.set(auth.COOKIE_NAME, auth.make_token(days=-1))
        assert c2.get("/api/jobs").status_code == 401, "expired token accepted"

        assert c.post("/api/logout").status_code == 200
        assert c.get("/api/jobs").status_code == 401, "session survived logout"
    finally:
        (config.ADMIN_USERNAME, config.ADMIN_PASSWORD, config.ADMIN_PASSWORD_SHA256) = orig

    # Force-disabled state explicitly — don't rely on .env having no admin
    # password set (the deployed/dev .env may legitimately have real creds).
    config.ADMIN_PASSWORD, config.ADMIN_PASSWORD_SHA256 = "", ""
    try:
        c3 = TestClient(app)
        assert c3.get("/api/jobs").status_code == 200
        assert c3.get("/").status_code == 200
    finally:
        (config.ADMIN_USERNAME, config.ADMIN_PASSWORD, config.ADMIN_PASSWORD_SHA256) = orig
    print("OK auth gate: 302/401 logged out, cookie login+logout, tamper/expiry "
          "rejected, disabled mode open")


def test_server_and_video_stop_controls():
    """Server controls are one-shot and video work has per-job stop controls."""
    import importlib
    from fastapi.testclient import TestClient

    mainmod = importlib.import_module("app.main")
    scheduled = []
    restarted = []
    original_schedule = mainmod._schedule_process_shutdown
    original_restart = mainmod._schedule_process_restart
    original_cancel_all = mainmod.jobs.cancel_active_jobs_for_shutdown
    original_requested = mainmod._shutdown_requested
    original_restart_requested = mainmod._restart_requested
    original_admin = (config.ADMIN_PASSWORD, config.ADMIN_PASSWORD_SHA256)
    try:
        config.ADMIN_PASSWORD, config.ADMIN_PASSWORD_SHA256 = "", ""
        mainmod._shutdown_requested = False
        mainmod._restart_requested = False
        mainmod._schedule_process_shutdown = lambda delay_s=0.8: scheduled.append(delay_s)
        mainmod._schedule_process_restart = lambda delay_s=0.8: restarted.append(delay_s)
        mainmod.jobs.cancel_active_jobs_for_shutdown = (
            lambda reason="server stopped by user": ["job-a", "job-b"]
        )

        client = TestClient(mainmod.app)
        response = client.post("/api/server/stop", json={})
        assert response.status_code == 202, response.text
        assert response.json() == {
            "ok": True,
            "already_stopping": False,
            "cancelled_jobs": ["job-a", "job-b"],
        }
        assert scheduled == [0.8], scheduled

        repeated = client.post("/api/server/stop", json={})
        assert repeated.status_code == 202
        assert repeated.json()["already_stopping"] is True
        assert scheduled == [0.8], "duplicate click scheduled another process kill"

        # Stop and restart are mutually exclusive once either action has begun.
        rejected_restart = client.post("/api/server/restart", json={})
        assert rejected_restart.status_code == 202
        assert rejected_restart.json()["already_restarting"] is True
        assert restarted == []

        mainmod._shutdown_requested = False
        response = client.post("/api/server/restart", json={})
        assert response.status_code == 202, response.text
        assert response.json() == {
            "ok": True,
            "already_restarting": False,
            "cancelled_jobs": ["job-a", "job-b"],
        }
        assert restarted == [0.8], restarted

        repeated = client.post("/api/server/restart", json={})
        assert repeated.status_code == 202
        assert repeated.json()["already_restarting"] is True
        assert restarted == [0.8], "duplicate click scheduled another restart"

    finally:
        mainmod._schedule_process_shutdown = original_schedule
        mainmod._schedule_process_restart = original_restart
        mainmod.jobs.cancel_active_jobs_for_shutdown = original_cancel_all
        mainmod._shutdown_requested = original_requested
        mainmod._restart_requested = original_restart_requested
        config.ADMIN_PASSWORD, config.ADMIN_PASSWORD_SHA256 = original_admin

    static_dir = Path(mainmod.__file__).parent / "static"
    html = (static_dir / "index.html").read_text(encoding="utf-8")
    js = (static_dir / "app.js").read_text(encoding="utf-8")
    assert 'id="serverRestartBtn"' in html and "Restart server" in html
    assert 'id="serverStopBtn"' in html and "Stop server" in html
    assert 'id="stopBtn"' in html and "Stop video processing" in html
    assert 'postJSON("/api/server/restart", {})' in js
    assert 'postJSON("/api/server/stop", {})' in js
    assert 'data-stop="${j.id}"' in js
    assert "if (!state.job && active) watch(active.id)" in js
    assert "PROCESSING_STATES.has(job.state)" in js
    print("OK controls: per-video stop, one-shot server stop + restart")


def test_settings_runtime_override():
    """Settings UI contract: provider + keys are applied in-memory immediately
    (no restart) and NEVER echoed back in any /api/settings response, even
    right after being saved."""
    from fastapi.testclient import TestClient

    from app import config
    from app.main import app
    from app.pipeline import scoring
    from app.pipeline.scoring import gemini_provider

    orig_file = config._settings_file
    orig_vals = (config.SCORING_PROVIDER, config.OPENAI_API_KEY, config.GEMINI_API_KEY)
    # This endpoint sits behind the same AuthGate as everything else — that's
    # already covered by test_auth_gate. Disable auth here so this test can
    # focus on the settings contract regardless of what .env has configured.
    orig_admin = (config.ADMIN_PASSWORD, config.ADMIN_PASSWORD_SHA256)
    config.ADMIN_PASSWORD, config.ADMIN_PASSWORD_SHA256 = "", ""
    temp_file = WORK / "_settings_test.json"
    temp_file.unlink(missing_ok=True)
    config._settings_file = lambda: temp_file

    orig_gemini_init = gemini_provider.GeminiScoringProvider.__init__
    gemini_provider.GeminiScoringProvider.__init__ = lambda self: None
    try:
        c = TestClient(app)
        r = c.post("/api/settings", json={"scoring_provider": "gemini",
                                          "gemini_api_key": "synthetic-test-value-not-a-credential"})
        assert r.status_code == 200
        body = r.json()
        assert "synthetic-test-value-not-a-credential" not in r.text, "key echoed in POST response"
        assert set(body) == {"scoring_provider", "has_openai_key", "has_gemini_key"}
        assert body["scoring_provider"] == "gemini" and body["has_gemini_key"] is True

        r2 = c.get("/api/settings")
        assert "synthetic-test-value-not-a-credential" not in r2.text, "key echoed in GET response"
        assert r2.json()["scoring_provider"] == "gemini"
        assert r2.json()["has_gemini_key"] is True

        # the whole point: no restart needed — the global updated in-process
        assert config.GEMINI_API_KEY == "synthetic-test-value-not-a-credential"
        assert scoring.get_provider().name == "gemini"

        r3 = c.post("/api/settings", json={"openai_api_key": ""})
        assert r3.json()["has_openai_key"] is False
        assert config.OPENAI_API_KEY == ""
    finally:
        gemini_provider.GeminiScoringProvider.__init__ = orig_gemini_init
        config._settings_file = orig_file
        config.SCORING_PROVIDER, config.OPENAI_API_KEY, config.GEMINI_API_KEY = orig_vals
        config.ADMIN_PASSWORD, config.ADMIN_PASSWORD_SHA256 = orig_admin
        temp_file.unlink(missing_ok=True)
    print("OK settings: provider+key applied live with no restart, never echoed back")


def test_model_download_no_race():
    """N concurrent callers requesting the same missing model file must not
    race — regression test for the bug where all parallel render workers
    shared one deterministic temp filename (whoever renamed first left the
    rest with "No such file or directory"). Each call now gets a unique temp
    file + atomic os.replace, so concurrent redundant downloads are harmless."""
    import threading
    import time as _time

    from app.pipeline import mp_models

    fake_name = "_fake_model_for_test.bin"
    fake_path = config.MODELS_DIR / fake_name
    fake_path.unlink(missing_ok=True)
    orig_models = dict(mp_models._MODELS)
    mp_models._MODELS[fake_name] = "http://example.invalid/fake"

    lock = threading.Lock()
    call_count = {"n": 0}

    def fake_urlretrieve(url, filename):
        with lock:
            call_count["n"] += 1
        _time.sleep(0.05)  # widen the race window past what real network jitter gave us
        Path(filename).write_bytes(b"fake-model-bytes")

    orig_urlretrieve = mp_models.urllib.request.urlretrieve
    mp_models.urllib.request.urlretrieve = fake_urlretrieve
    errors = []

    def worker():
        try:
            p = mp_models.model_path(fake_name)
            assert p.read_bytes() == b"fake-model-bytes"
        except Exception as e:  # noqa: BLE001 - captured for the assertion below
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        assert not errors, f"race errors in {len(errors)}/8 callers: {errors}"
        assert fake_path.exists() and fake_path.read_bytes() == b"fake-model-bytes"
        assert call_count["n"] >= 1, "no download happened at all"
    finally:
        mp_models.urllib.request.urlretrieve = orig_urlretrieve
        mp_models._MODELS = orig_models
        fake_path.unlink(missing_ok=True)
    print(f"OK model download: 8 concurrent callers on a missing file, "
          f"{call_count['n']} redundant downloads, zero races")


def test_ensure_models_startup_gate():
    """ensure_models() downloads whatever is missing and is a cheap no-op for
    files already on disk — the startup gate the audit asked for."""
    from app.pipeline import mp_models

    names = ["_fake_a_for_test.bin", "_fake_b_for_test.bin"]
    paths = [config.MODELS_DIR / n for n in names]
    for p in paths:
        p.unlink(missing_ok=True)
    orig_models = dict(mp_models._MODELS)
    mp_models._MODELS = {n: f"http://example.invalid/{n}" for n in names}

    calls = []

    def fake_urlretrieve(url, filename):
        calls.append(url)
        Path(filename).write_bytes(b"x")

    orig_urlretrieve = mp_models.urllib.request.urlretrieve
    mp_models.urllib.request.urlretrieve = fake_urlretrieve
    try:
        mp_models.ensure_models()
        assert all(p.exists() for p in paths), "ensure_models did not download missing files"
        assert len(calls) == 2
        mp_models.ensure_models()  # second call: both already present
        assert len(calls) == 2, "ensure_models re-downloaded files that already exist"
    finally:
        mp_models.urllib.request.urlretrieve = orig_urlretrieve
        mp_models._MODELS = orig_models
        for p in paths:
            p.unlink(missing_ok=True)
    print("OK ensure_models: downloads what's missing, no-ops on what's already there")


def test_human_friendly_network_error():
    from app import jobs

    raw = (
        "\x1b[0;31mERROR:\x1b[0m Unable to download API page: "
        "[WinError 10013] socket access forbidden by its access permissions"
    )
    message = jobs._describe_error(RuntimeError(raw))
    assert "\x1b" not in message
    assert message == (
        "YouTube network access is blocked on this server. "
        "Restart Clipping Tool with internet/network permission, then click Retry."
    )

    oom = (
        "analysis stage 'faces' OpenCV(5.0.0) error: (-4:Insufficient memory) "
        "Failed to allocate 6220800 bytes in function 'cv::OutOfMemoryError'"
    )
    assert jobs._describe_error(RuntimeError(oom)) == (
        "Video analysis ran out of RAM. Clipping Tool will reuse completed work; "
        "close memory-heavy apps, then click Retry."
    )


def test_low_memory_stage2_uses_fresh_process_per_task():
    """Sequential low-RAM mode must not reuse a native-library worker."""
    import concurrent.futures
    from app import jobs

    made_pools = []
    stage_events = []

    class ImmediatePool:
        def __init__(self, max_workers):
            assert max_workers == 1
            self._processes = {}
            made_pools.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def submit(self, func, *args):
            future = concurrent.futures.Future()
            try:
                future.set_result(func(*args))
            except Exception as exc:
                future.set_exception(exc)
            return future

    original_pool = jobs.ProcessPoolExecutor
    original_stage = jobs._stage
    try:
        jobs.ProcessPoolExecutor = ImmediatePool
        jobs._stage = lambda job_id, name, status, seconds=None: stage_events.append(
            (name, status)
        )
        tasks = [
            ("transcribe", lambda: 1.0, ()),
            ("scenes", lambda: 2.0, ()),
            ("faces", lambda: 3.0, ()),
        ]
        jobs._run_stage2_isolated("_test_low_ram", tasks)
    finally:
        jobs.ProcessPoolExecutor = original_pool
        jobs._stage = original_stage
        jobs._active_pool.pop("_test_low_ram", None)

    assert len(made_pools) == 3, "analysis tasks reused the same worker pool"
    assert stage_events == [
        ("transcribe", "running"), ("transcribe", "done"),
        ("scenes", "running"), ("scenes", "done"),
        ("faces", "running"), ("faces", "done"),
    ]


if __name__ == "__main__":
    test_cgroup_cpu_quota_parser()
    test_human_friendly_network_error()
    test_low_memory_stage2_uses_fresh_process_per_task()
    test_windowing()
    test_target_scaling()
    test_windowing_capacity_for_large_targets()
    test_no_split_for_duplicate_tracks()
    test_dialogue_split_and_active_speaker_focus()
    test_gap_interpolation_holds()
    test_single_track_grace()
    test_center_fallback_is_smooth()
    test_coarse_face_safety_lock()
    test_multi_track_blackout_goes_wide()
    test_blackout_hands_off_to_visible_track()
    test_fragment_merge()
    test_caption_no_overlap()
    test_render_timeout_drain()
    test_stranded_job_reaped_on_load()
    test_cancel_stops_running_pool()
    test_delete_and_clear_caches()
    test_low_ram_caps_render_workers()
    test_drop_static_nonparticipant_faces()
    test_no_split_when_crops_overlap()
    test_disk_emergency_cleanup()
    test_disk_check_blocks_render()
    test_atomic_job_save_and_part_sweep()
    test_render_pipe_failure_not_masked()
    test_posix_communicate_flush_not_masked()
    test_nvenc_session_cap()
    test_gpu_hardware_selection()
    test_source_height_cap_and_analysis_workers()
    test_transcribe_receives_device()
    test_concurrency_cap()
    test_unique_output_path()
    test_retention_cleanup()
    test_auth_gate()
    test_server_and_video_stop_controls()
    test_settings_runtime_override()
    test_model_download_no_race()
    test_ensure_models_startup_gate()
    test_render()
    shutil.rmtree(WORK, ignore_errors=True)
    print("ALL SYNTHETIC TESTS PASSED")
