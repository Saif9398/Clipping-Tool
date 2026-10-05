"""Per-clip active-speaker analysis (the input to the camera planner).

Approach (design reference: smart-reframe; see docs/ATTRIBUTION.md):
- FaceMesh sampled at ~10 fps over the clip only (cheap; the coarse whole-video
  pass already told us faces exist — this pass adds mouth landmarks).
- Persistent identity tracks by centroid continuity.
- "Is speaking" per track = rolling variance of mouth openness, gated by an
  actual-speech mask built from transcript word timings + audio RMS.
- State machine with hysteresis + minimum state duration:
    SINGLE(tid)  one clearly dominant speaker
    SPLIT(a, b)  two engaged speakers -> stacked close-ups (never a wide shot)
  Silence/uncertainty holds the previous state instead of flapping.

Output arrays are per OUTPUT frame (source fps) — the camera planner consumes
them directly with no further detection work.
"""
from __future__ import annotations

import logging
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_FPS = 10.0
MOUTH_WIN_S = 0.6          # rolling window for mouth-openness variance
ACT_THRESH = 0.010         # mouth-activity threshold (normalized units)
ENGAGED_LOOKBACK_S = 1.25  # recent speech from both people allows a stable split
ENGAGED_RATIO = 3.0        # reject split if one person's recent evidence is tiny
MIN_STATE_S = 1.5          # responsive enough to follow a real speaker change
                           # while still suppressing mouth-detector flicker
MIN_SPLIT_STATE_S = 1.0    # safe split may enter sooner than a speaker cut
STATIC_POS_STD = 0.004     # a track whose box-center barely moves over its life...
STATIC_MAX_W = 0.045       # ...AND is small in frame is set dressing (wall poster
                           # / promo art / TV still), not a participant. Real faces
                           # here run 0.08+ wide and jitter 0.09+; a poster sat at
                           # width 0.025, pos_std 0.001 (mouth landmarks jitter on a
                           # printed face, so mouth variance is NOT a reliable tell).
TRACK_MATCH_DIST = 0.18    # max normalized centroid distance to join a track
DISTINCT_DIST_FACTOR = 1.3  # split only if centers are > this x the wider face's width apart
SPLIT_MAX_CROP_IOU = 0.30   # and only if the two split CROPS overlap at most this much
                            # (crops are ~3x face height wide: faces can be "distinct"
                            # while crops still show the same region — QA soft-flags
                            # IoU>0.20 w/ pixel corroboration, hard-fails >0.50; 0.30
                            # leaves margin for smoothing wobble)
GAP_BRIDGE_S = 1.0         # interp bridges detection gaps up to this; longer gaps hold last box
FRAG_MERGE_GAP_S = 3.0     # track fragments of one person within this gap get re-joined
ABSENT_GRACE_S = 1.0       # single-speaker framing gives up only after this long unseen


@dataclass
class Track:
    tid: int
    ts: list[float] = field(default_factory=list)       # sample times (clip-relative)
    boxes: list[list[float]] = field(default_factory=list)  # [cx, cy, w, h] normalized
    mouth: list[float] = field(default_factory=list)    # mouth openness (normalized)

    @property
    def centroid(self) -> tuple[float, float]:
        b = self.boxes[-1]
        return b[0], b[1]


@dataclass
class ClipAnalysis:
    n_frames: int
    fps: float
    # tid -> (N,4) float array [cx,cy,w,h] normalized + (N,) visibility bools
    boxes: dict[int, np.ndarray]
    visible: dict[int, np.ndarray]
    # list of {"f0": int, "f1": int, "mode": "single"|"split", "tids": [..]}
    states: list[dict]
    # Per-frame coarse detections [cx, cy, w, h, confidence]. These are an
    # independent safety net for the camera when the high-rate speaker track
    # is temporarily missing or stale around a source shot change.
    coarse_faces: list[list[list[float]]] = field(default_factory=list)


def analyze_clip(work_dir: Path, t0: float, t1: float, transcript: dict) -> ClipAnalysis:
    import cv2

    cap = cv2.VideoCapture(str(work_dir / "source.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    f_start = int(round(t0 * fps))
    n_frames = int(round((t1 - t0) * fps))

    tracks = _collect_tracks(cap, fps, f_start, n_frames)
    cap.release()

    # Re-join fragments of one person split by a detection dropout BEFORE the
    # size filter — otherwise a fragmented real speaker can be discarded
    # entirely (every piece under the sample floor) → whole-clip center crop.
    tracks = _merge_fragments(tracks)
    # Drop set-dressing "faces" (wall posters, promo art, a TV still) that the
    # detectors pick up but that are not participants — they can otherwise eat a
    # track slot and, on this footage, a printed poster of the two hosts was one
    # of the four detections. A real person moves and/or talks; furniture does
    # neither. (This one, e.g., sat dead-still at width 0.025 near the top edge.)
    tracks = _drop_static_faces(tracks)
    # Keep the (up to) 4 most-observed tracks (2 camera angles x 2 people);
    # tiny ones are misdetections
    tracks = sorted(tracks, key=lambda tr: len(tr.ts), reverse=True)[:4]
    tracks = [tr for tr in tracks if len(tr.ts) >= 5]

    boxes, visible = _interpolate(tracks, n_frames, fps)
    speech = _speech_mask(work_dir, t0, t1, transcript, n_frames, fps)
    activity = _mouth_activity(tracks, n_frames, fps, speech, visible)
    states = _state_machine(activity, visible, boxes, n_frames, fps, src_w, src_h)
    coarse_faces = _coarse_faces_per_frame(work_dir, t0, n_frames, fps)
    return ClipAnalysis(
        n_frames=n_frames,
        fps=fps,
        boxes=boxes,
        visible=visible,
        states=states,
        coarse_faces=coarse_faces,
    )


def _coarse_faces_per_frame(work_dir: Path, t0: float, n_frames: int,
                            fps: float) -> list[list[list[float]]]:
    """Nearest coarse detections for each clip frame.

    Stage 2 already generated this file, so the render adds no detection work.
    Keeping this independent from FaceMesh is important: the two detectors fail
    differently, which makes the coarse pass a reliable framing safety net.
    """
    import json

    path = work_dir / "faces_coarse.json"
    try:
        entries = json.loads(path.read_text(encoding="utf-8")).get("entries", [])
    except (OSError, json.JSONDecodeError):
        return []
    if not entries:
        return []

    times = np.asarray([float(e["t"]) for e in entries], dtype=np.float64)
    frame_times = t0 + np.arange(n_frames, dtype=np.float64) / fps
    hi = np.searchsorted(times, frame_times, side="left")
    lo = np.clip(hi - 1, 0, len(times) - 1)
    hi = np.clip(hi, 0, len(times) - 1)
    nearest = np.where(
        np.abs(times[hi] - frame_times) < np.abs(times[lo] - frame_times),
        hi,
        lo,
    )
    return [
        [[float(v) for v in face] for face in entries[int(idx)].get("f", [])]
        for idx in nearest
    ]


def _collect_tracks(cap, fps: float, f_start: int, n_frames: int) -> list[Track]:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    from app.pipeline.mp_models import model_path

    stride = max(1, round(fps / SAMPLE_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    tracks: list[Track] = []

    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path("face_landmarker.task"))),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=4,
        # presence defaults to 0.5 — stricter than everything else in the stack
        # and the main source of mid-track dropouts on blur/head-turns
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.4,
    )
    crop_opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path("face_landmarker.task"))),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
    )
    yunet = None
    with vision.FaceLandmarker.create_from_options(opts) as mesh, \
            vision.FaceLandmarker.create_from_options(crop_opts) as crop_mesh:
        for i in range(n_frames):
            if not cap.grab():
                break
            if i % stride:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            h, w = frame.shape[:2]
            scale = 640 / w
            small = cv2.resize(frame, (640, int(h * scale)))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = mesh.detect_for_video(img, int(i / fps * 1000))
            t = i / fps
            frame_boxes: list[list[float]] = []
            samples: list[tuple[list[float], float | None]] = []
            for lms in (res.face_landmarks or []):
                pts = np.array([(p.x, p.y) for p in lms], dtype=np.float32)
                x0, y0 = pts.min(axis=0)
                x1, y1 = pts.max(axis=0)
                box = [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
                # FaceLandmarker sometimes emits two landmark sets for ONE face;
                # a duplicate would spawn a phantom track (and later a false
                # split-screen of the same person) — suppress it here.
                if any(_same_face(box, other) for other in frame_boxes):
                    continue
                frame_boxes.append(box)
                # mouth openness: inner-lip gap (13-14) normalized by face height (10-152)
                face_h = np.linalg.norm(pts[10] - pts[152]) + 1e-6
                openness = float(np.linalg.norm(pts[13] - pts[14]) / face_h)
                samples.append((box, openness))
            if not samples:
                # FaceLandmarker's internal detector cannot see small faces
                # (wide two-shots) at ANY input size — YuNet can. Without this
                # rescue a wide shot is a total blackout and the camera frames
                # empty background.
                if yunet is None:
                    yunet = _make_yunet(small.shape[1], small.shape[0])
                for box, openness in _rescue_detect(yunet, small, frame, crop_mesh):
                    if any(_same_face(box, other) for other in frame_boxes):
                        continue
                    frame_boxes.append(box)
                    samples.append((box, openness))
            for box, openness in samples:
                _assign(tracks, t, box, openness)
    return tracks


def _make_yunet(w: int, h: int):
    import cv2

    from app.pipeline.mp_models import model_path

    return cv2.FaceDetectorYN_create(
        str(model_path("face_detection_yunet_2023mar.onnx")), "", (w, h),
        score_threshold=0.6)


def _rescue_detect(yunet, small_bgr, frame_bgr, crop_mesh) -> list[tuple[list[float], float | None]]:
    """YuNet full-frame detection for faces below FaceLandmarker's floor, plus
    a landmark cascade on an upscaled full-res crop for mouth openness. A face
    whose crop yields no landmarks (e.g. profile view) keeps its YuNet box —
    real framing, just no speech evidence for that sample."""
    import cv2
    import mediapipe as mp

    _, dets = yunet.detect(small_bgr)
    if dets is None:
        return []
    sh, sw = small_bgr.shape[:2]
    fh_px, fw_px = frame_bgr.shape[:2]
    out: list[tuple[list[float], float | None]] = []
    for d in dets:
        x, y, w, h = d[:4]
        box = [(x + w / 2) / sw, (y + h / 2) / sh, w / sw, h / sh]
        m = int(max(box[2] * fw_px, box[3] * fh_px) * 1.5)
        px, py = int(box[0] * fw_px), int(box[1] * fh_px)
        crop = frame_bgr[max(0, py - m):min(fh_px, py + m),
                         max(0, px - m):min(fw_px, px + m)]
        openness = None
        if crop.size:
            if crop.shape[0] < 256:
                s = 256 / crop.shape[0]
                crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), 256))
            rgb = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_BGR2RGB)
            res = crop_mesh.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if res.face_landmarks:
                pts = np.array([(p.x, p.y) for p in res.face_landmarks[0]], dtype=np.float32)
                face_h = np.linalg.norm(pts[10] - pts[152]) + 1e-6
                openness = float(np.linalg.norm(pts[13] - pts[14]) / face_h)
        out.append((box, openness))
    return out


def _same_face(a: list[float], b: list[float]) -> bool:
    """Two detections in one frame belong to the same physical face if their
    centers are closer than half the wider box's width."""
    return float(np.hypot(a[0] - b[0], a[1] - b[1])) < 0.5 * max(a[2], b[2])


def _assign(tracks: list[Track], t: float, box: list[float],
            openness: float | None) -> None:
    best, best_d = None, TRACK_MATCH_DIST
    for tr in tracks:
        cx, cy = tr.centroid
        d = ((box[0] - cx) ** 2 + (box[1] - cy) ** 2) ** 0.5
        if d < best_d and (not tr.ts or tr.ts[-1] != t):
            best, best_d = tr, d
    if best is None:
        best = Track(tid=len(tracks))
        tracks.append(best)
    best.ts.append(t)
    best.boxes.append(box)
    # openness None = box without landmarks (profile face via rescue): freeze
    # the last value — zero variance means no false speaking evidence
    if openness is None:
        openness = best.mouth[-1] if best.mouth else 0.0
    best.mouth.append(openness)


def _drop_static_faces(tracks: list[Track]) -> list[Track]:
    """Remove non-participant faces: a detection that, over enough samples to
    judge, both barely moves (box-center std < STATIC_POS_STD) and is small in
    frame (mean width < STATIC_MAX_W) is a printed/screen face, not a person.
    Both conditions are required and both thresholds sit far below any real
    framed head, so a still-sitting speaker (who is larger AND has some micro-
    motion) is never dropped. Never returns empty (falls back to the originals
    if every track looked static)."""
    kept = []
    for tr in tracks:
        if len(tr.ts) >= int(2 * SAMPLE_FPS):  # need ~2s of samples to trust the stats
            b = np.asarray(tr.boxes)
            pos_std = float(np.hypot(b[:, 0].std(), b[:, 1].std()))
            mean_w = float(b[:, 2].mean())
            if pos_std < STATIC_POS_STD and mean_w < STATIC_MAX_W:
                log.info("speaker: dropping static non-participant face "
                         "(pos_std=%.4f mean_w=%.3f, %d samples)",
                         pos_std, mean_w, len(tr.ts))
                continue
        kept.append(tr)
    return kept or tracks


def _merge_fragments(tracks: list[Track]) -> list[Track]:
    """Re-join tracks that are one person split by a detection dropout: no time
    overlap, gap < FRAG_MERGE_GAP_S, boundary centroids within ~1.5 face widths."""
    tracks = sorted(tracks, key=lambda tr: tr.ts[0])
    merged: list[Track] = []
    for tr in tracks:
        host = None
        for m in merged:
            gap = tr.ts[0] - m.ts[-1]
            if not (0 <= gap < FRAG_MERGE_GAP_S):
                continue
            a, b = m.boxes[-1], tr.boxes[0]
            d = float(np.hypot(a[0] - b[0], a[1] - b[1]))
            if d < max(TRACK_MATCH_DIST, 1.5 * max(a[2], b[2])):
                host = m
                break
        if host is None:
            merged.append(tr)
        else:
            host.ts += tr.ts
            host.boxes += tr.boxes
            host.mouth += tr.mouth
    for tid, tr in enumerate(merged):
        tr.tid = tid
    return merged


def _interpolate(tracks: list[Track], n_frames: int, fps: float):
    """Boxes for every output frame. Linear interp bridges short detection gaps
    (<= GAP_BRIDGE_S); across longer gaps the last known box is HELD rather than
    panned toward the next sighting — a blind pan frames empty background.
    Ends are held flat either way."""
    frame_t = np.arange(n_frames) / fps
    boxes, visible = {}, {}
    for tr in tracks:
        ts = np.array(tr.ts)
        arr = np.array(tr.boxes)  # (M,4)
        out = np.empty((n_frames, 4), dtype=np.float32)
        for k in range(4):
            out[:, k] = np.interp(frame_t, ts, arr[:, k])
        if len(ts) > 1:
            idx = np.searchsorted(ts, frame_t, side="right") - 1
            prev = np.clip(idx, 0, len(ts) - 1)
            nxt = np.clip(idx + 1, 0, len(ts) - 1)
            hold = (idx >= 0) & (idx < len(ts) - 1) & (ts[nxt] - ts[prev] > GAP_BRIDGE_S)
            out[hold] = arr[prev[hold]]
        boxes[tr.tid] = out
        # visible where a real sample exists within ~0.5s
        vis = np.zeros(n_frames, dtype=bool)
        idx = np.searchsorted(ts, frame_t)
        idx_lo = np.clip(idx - 1, 0, len(ts) - 1)
        idx_hi = np.clip(idx, 0, len(ts) - 1)
        near = np.minimum(np.abs(frame_t - ts[idx_lo]), np.abs(ts[idx_hi] - frame_t))
        vis[near <= 0.5] = True
        visible[tr.tid] = vis
    return boxes, visible


def _speech_mask(work_dir: Path, t0: float, t1: float, transcript: dict,
                 n_frames: int, fps: float) -> np.ndarray:
    """Per-frame bool: is anyone actually speaking (transcript words ∪ audio RMS)."""
    mask = np.zeros(n_frames, dtype=bool)
    for seg in transcript["segments"]:
        for wd in seg["words"]:
            if wd["e"] < t0 or wd["s"] > t1:
                continue
            a = max(0, int((wd["s"] - t0) * fps))
            b = min(n_frames, int(np.ceil((wd["e"] - t0) * fps)) + 1)
            mask[a:b] = True
    try:
        with wave.open(str(work_dir / "audio16k.wav"), "rb") as w:
            sr = w.getframerate()
            w.setpos(min(int(t0 * sr), w.getnframes()))
            data = np.frombuffer(w.readframes(int((t1 - t0) * sr)), dtype=np.int16)
        data = data.astype(np.float32) / 32768.0
        hop = max(1, int(sr / fps))
        n = min(n_frames, len(data) // hop)
        rms = np.sqrt(np.mean(data[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-9)
        thresh = max(0.01, float(np.percentile(rms, 25)) * 1.8)
        mask[:n] |= rms > thresh
    except Exception as e:
        log.warning("VAD energy mask failed (%s) — transcript-only speech mask", e)
    return mask


def _mouth_activity(tracks: list[Track], n_frames: int, fps: float,
                    speech: np.ndarray, visible: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Per-frame speaking evidence per track: rolling std of mouth openness,
    interpolated to frame rate, gated by the speech mask AND track visibility
    (np.interp holds a dead track's last value forever — without the visibility
    gate a vanished track keeps 'speaking' and can trigger false splits)."""
    frame_t = np.arange(n_frames) / fps
    out = {}
    for tr in tracks:
        ts = np.array(tr.ts)
        mouth = np.array(tr.mouth)
        win = max(3, int(MOUTH_WIN_S * SAMPLE_FPS))
        act = np.zeros_like(mouth)
        for i in range(len(mouth)):
            lo = max(0, i - win + 1)
            act[i] = mouth[lo:i + 1].std()
        per_frame = np.interp(frame_t, ts, act)
        per_frame = np.where(speech & visible[tr.tid], per_frame, 0.0)
        # EMA smoothing (~0.4s time constant)
        alpha = 1.0 - np.exp(-1.0 / (0.4 * fps))
        sm = np.empty_like(per_frame)
        acc = 0.0
        for i, v in enumerate(per_frame):
            acc += alpha * (v - acc)
            sm[i] = acc
        out[tr.tid] = sm
    return out


def _causal_peak(values: np.ndarray, window_frames: int) -> np.ndarray:
    """Recent peak without looking into future frames."""
    window_frames = max(1, int(window_frames))
    out = np.empty_like(values)
    for i in range(len(values)):
        out[i] = np.max(values[max(0, i - window_frames + 1):i + 1])
    return out


def _state_machine(activity: dict[int, np.ndarray], visible: dict[int, np.ndarray],
                   boxes: dict[int, np.ndarray], n_frames: int, fps: float,
                   src_w: int = 1920, src_h: int = 1080) -> list[dict]:
    tids = list(activity.keys())
    if not tids:
        return [{"f0": 0, "f1": n_frames, "mode": "single", "tids": []}]
    if len(tids) == 1:
        # Hold framing on the lone speaker through detection dropouts; go wide
        # (tids=[]) only for absences longer than the grace window.
        tid = tids[0]
        absent = _long_invisible(visible[tid], fps)
        raw1 = [("single", [] if absent[i] else [tid]) for i in range(n_frames)]
        return _merge_runs(raw1, fps)

    # The two most active tracks overall are "the conversation"
    totals = {t: float(activity[t].sum()) for t in tids}
    a, b = sorted(tids, key=totals.get, reverse=True)[:2]
    # Stable split order: person more to the left goes on top (reading order)
    a_x = float(np.mean(boxes[a][:, 0]))
    b_x = float(np.mean(boxes[b][:, 0]))
    top, bottom = (a, b) if a_x <= b_x else (b, a)
    recent_window = max(1, int(round(ENGAGED_LOOKBACK_S * fps)))
    recent_a = _causal_peak(activity[a], recent_window)
    recent_b = _causal_peak(activity[b], recent_window)

    # Function-level import (project convention) — also avoids the
    # camera -> speaker top-level import cycle (camera imports ClipAnalysis).
    from app.pipeline.camera import split_rects_iou

    def distinct(i: int) -> bool:
        """True only when the two tracks are unambiguously different people
        AND a split of them would actually show two different regions:
        centers farther apart than DISTINCT_DIST_FACTOR x the wider face (a
        duplicated track of one face is co-located and can never pass), and
        the projected split CROPS must not overlap more than
        SPLIT_MAX_CROP_IOU — two real people sitting close together pass the
        face-distance test while their crops (~3x face height wide) still
        frame the same region, which renders as a doubled-looking split and
        flaps split<->single on the boundary (every flip is a camera cut)."""
        ba, bb = boxes[a][i], boxes[b][i]
        dist = float(np.hypot(ba[0] - bb[0], ba[1] - bb[1]))
        if dist <= DISTINCT_DIST_FACTOR * max(ba[2], bb[2]):
            return False
        return split_rects_iou(ba, bb, src_w, src_h) <= SPLIT_MAX_CROP_IOU

    # All retained tracks gone for longer than grace = genuine blackout.
    # A third visible camera-angle track must keep the frame alive.
    any_visible = np.logical_or.reduce([visible[t] for t in tids])
    absent_all = _long_invisible(any_visible, fps)

    raw = []
    prev = ("single", [a])
    for i in range(n_frames):
        both_vis = visible[a][i] and visible[b][i]
        recent_hi, recent_lo = max(recent_a[i], recent_b[i]), min(recent_a[i], recent_b[i])
        engaged = (
            recent_lo > ACT_THRESH
            and recent_hi / (recent_lo + 1e-9) < ENGAGED_RATIO
        )
        vis_now = [t for t in tids if visible[t][i]]
        best = max(vis_now, key=lambda t: activity[t][i]) if vis_now else None
        speaking = best is not None and activity[best][i] > ACT_THRESH
        if engaged and both_vis and distinct(i):
            state = ("split", [top, bottom])
        elif speaking:
            state = ("single", [best])
        else:
            state = prev  # hysteresis band or silence: hold
        # never HOLD a split whose faces stopped being distinct/visible —
        # degrade to the dominant speaker instead of duplicating one person
        if state[0] == "split" and not (both_vis and distinct(i)):
            if best is not None:
                state = ("single", [best])
            elif absent_all[i]:
                state = ("single", [])
            else:
                held = prev[1][0] if prev[1] else a
                state = ("single", [held])
        # a single target must be visible; prefer the other conversation
        # member, then ANY visible track (this camera angle's tracks), then
        # hold through short dropouts — wide only after a sustained blackout
        if state[0] == "single" and state[1] and not visible[state[1][0]][i]:
            if vis_now:
                state = ("single", [max(vis_now, key=lambda t: activity[t][i])])
            elif absent_all[i]:
                state = ("single", [])
            else:
                state = prev
        raw.append(state)
        prev = state
    return _merge_runs(raw, fps)


def _long_invisible(vis: np.ndarray, fps: float) -> np.ndarray:
    """True inside invisible runs lasting >= ABSENT_GRACE_S (genuine absence,
    not a detection dropout)."""
    out = np.zeros(len(vis), dtype=bool)
    min_run = int(ABSENT_GRACE_S * fps)
    i, n = 0, len(vis)
    while i < n:
        if vis[i]:
            i += 1
            continue
        j = i
        while j < n and not vis[j]:
            j += 1
        if j - i >= min_run:
            out[i:j] = True
        i = j
    return out


def _merge_runs(raw: list[tuple], fps: float) -> list[dict]:
    """Merge per-frame states into runs + enforce minimum state duration."""
    min_frames = int(MIN_STATE_S * fps)
    min_split_frames = int(MIN_SPLIT_STATE_S * fps)
    runs = []
    for i, st in enumerate(raw):
        if runs and runs[-1]["state"] == st:
            runs[-1]["f1"] = i + 1
        else:
            runs.append({"state": st, "f0": i, "f1": i + 1})
    merged = []
    for r in runs:
        required = min_split_frames if r["state"][0] == "split" else min_frames
        if merged and (r["f1"] - r["f0"] < required):
            merged[-1]["f1"] = r["f1"]  # absorb short flicker into previous state
        elif merged and merged[-1]["state"] == r["state"]:
            merged[-1]["f1"] = r["f1"]
        else:
            merged.append(dict(r))
    return [
        {"f0": r["f0"], "f1": r["f1"], "mode": r["state"][0], "tids": list(r["state"][1])}
        for r in merged
    ]
