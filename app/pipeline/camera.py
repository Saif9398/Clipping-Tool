"""Camera planner: turns speaker analysis into one crop plan per output frame.

Movement model (design reference: smart-reframe; see docs/ATTRIBUTION.md):
- dead-zone: sub-threshold face wobble does not move the camera at all
- asymmetric easing: alpha ramps up with error size — quick to react to a real
  move, slow and gentle to settle (no floating-head, no jitter)
- velocity clamp: even a large error never yanks the frame
- hard cuts (filter reset + micro punch-in) on speaker change and on source
  scene boundaries — never a smeary pan across the set
- slow punch-in drift on long single-speaker runs (subtle, Phase F polish)

Frame layouts (output is ALWAYS 1080x1920):
  single: one 9:16 crop, face ~35% of crop height, eyes at ~42% from top
  split:  two stacked 1080x960 crops (aspect 1.125), one per speaker
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app import config
from app.pipeline.speaker import ClipAnalysis

# Composition
FACE_FRAC_SINGLE = 0.35
FACE_FRAC_SPLIT = 0.42
HEAD_Y_SINGLE = 0.42
HEAD_Y_SPLIT = 0.46
SPLIT_ASPECT = config.OUT_W / (config.OUT_H / 2)  # 1.125

# Smoothing
DEAD_ZONE_FRAC = 0.015     # of crop height
ALPHA_SLOW = 0.05
ALPHA_FAST = 0.32
RAMP_FRAC = 0.12           # error size (as frac of crop h) at which alpha saturates
ALPHA_SIZE = 0.05
VMAX_FRAC = 0.030          # max camera speed: frac of source width per frame
CUT_REACQUIRE_S = 0.75     # trust a lone coarse face just after a source shot cut

# Cut polish
PUNCH_IN = 1.035           # micro punch-in right after a cut, easing out over PUNCH_S
PUNCH_S = 0.5
DRIFT_START_S = 6.0        # slow zoom drift on long steady runs
DRIFT_FULL_S = 20.0
DRIFT_MAX = 0.05


@dataclass
class FramePlan:
    mode: str                 # "single" | "split"
    rects: list[tuple[float, float, float, float]]  # (x, y, w, h) source px, 1 or 2


class _Smoother:
    """Asymmetric dead-zone smoother for (cx, cy, h) of one crop rect."""

    def __init__(self, cx: float, cy: float, h: float, vmax: float):
        self.cx, self.cy, self.h = cx, cy, h
        self.vmax = vmax

    def step(self, tcx: float, tcy: float, th: float) -> tuple[float, float, float]:
        dz = DEAD_ZONE_FRAC * self.h
        ramp = RAMP_FRAC * self.h
        for attr, target in (("cx", tcx), ("cy", tcy)):
            cur = getattr(self, attr)
            err = target - cur
            a_err = abs(err)
            if a_err < dz:
                alpha = 0.02  # tiny drift correction only
            else:
                k = min(1.0, (a_err - dz) / max(ramp, 1e-6))
                alpha = ALPHA_SLOW + (ALPHA_FAST - ALPHA_SLOW) * k
            delta = np.clip(alpha * err, -self.vmax, self.vmax)
            setattr(self, attr, cur + delta)
        h_err = th - self.h
        if abs(h_err) > 0.02 * self.h:
            self.h += ALPHA_SIZE * h_err
        return self.cx, self.cy, self.h


def plan(analysis: ClipAnalysis, src_w: int, src_h: int,
         scene_cut_frames: list[int]) -> list[FramePlan]:
    n, fps = analysis.n_frames, analysis.fps
    cuts = set(scene_cut_frames)
    plans: list[FramePlan] = []
    smoothers: list[_Smoother] = []
    run_key = None
    run_start = 0
    coarse_reacquire_until = 0
    vmax = VMAX_FRAC * src_w

    state_of_frame = _expand_states(analysis.states, n)

    for i in range(n):
        mode, tids = state_of_frame[i]
        key = (mode, tuple(tids))
        is_cut = (key != run_key) or (i in cuts)
        if i in cuts:
            coarse_reacquire_until = i + max(1, int(round(CUT_REACQUIRE_S * fps)))
        if is_cut:
            if not tids and smoothers and i not in cuts:
                # Losing the face is not a camera cut: keep the smoother so the
                # frame EASES out to the wide fallback under the velocity clamp
                # instead of hard-snapping to a center crop.
                run_key = key
                smoothers = smoothers[:1]
            else:
                run_key = key
                run_start = i
                smoothers = []

        targets = _targets(
            analysis,
            i,
            mode,
            tids,
            src_w,
            src_h,
            prefer_coarse=i < coarse_reacquire_until,
        )
        if not smoothers:
            smoothers = [_Smoother(*t, vmax=vmax) for t in targets]
        else:
            for sm, t in zip(smoothers, targets):
                sm.step(*t)

        run_t = (i - run_start) / fps
        punch = 1.0 + (PUNCH_IN - 1.0) * max(0.0, 1.0 - run_t / PUNCH_S)
        drift = 1.0
        if mode == "single" and run_t > DRIFT_START_S:
            drift = 1.0 - DRIFT_MAX * min(1.0, (run_t - DRIFT_START_S) / (DRIFT_FULL_S - DRIFT_START_S))

        aspect = (config.OUT_W / config.OUT_H) if mode == "single" else SPLIT_ASPECT
        rects = [
            _clamp_rect(sm.cx, sm.cy, sm.h * drift / punch, aspect, src_w, src_h)
            for sm in smoothers
        ]
        plans.append(FramePlan(mode=mode, rects=rects))
    return plans


def _expand_states(states: list[dict], n: int) -> list[tuple[str, list[int]]]:
    out: list[tuple[str, list[int]]] = [("single", [])] * n
    for st in states:
        for i in range(st["f0"], min(st["f1"], n)):
            out[i] = (st["mode"], st["tids"])
    return out


def _target_from_box(b, mode: str, src_w: int, src_h: int) -> tuple[float, float, float]:
    """(cx, cy, crop_h) target in source px for one normalized face box."""
    # ZOOM_FACTOR <1 shrinks the face fraction -> larger crop -> looser framing
    # AND less upscaling in the renderer (crop_h is inverse to ZOOM_FACTOR)
    face_frac = (FACE_FRAC_SINGLE if mode == "single" else FACE_FRAC_SPLIT) * config.ZOOM_FACTOR
    head_y = HEAD_Y_SINGLE if mode == "single" else HEAD_Y_SPLIT
    fcx, fcy, fh = b[0] * src_w, b[1] * src_h, b[3] * src_h
    # A constant face FRACTION means a distant face gets an arbitrarily tiny
    # crop: measured on real footage, 25% of one clip's rects were upscaled >5x
    # (a ~250 px crop stretched to 1920) and the worst hit 9.6x. Past a point
    # there is simply no detail left to magnify and unsharp only sharpens the
    # mush, so the crop never shrinks below what MAX_UPSCALE allows — a small
    # face is framed smaller rather than rendered soft.
    out_h = config.OUT_H if mode == "single" else config.OUT_H / 2
    floor = min(out_h / max(1.0, config.MAX_UPSCALE), float(src_h))
    crop_h = float(np.clip(fh / face_frac, max(fh * 1.8, floor), float(src_h)))
    cy = fcy + (0.5 - head_y) * crop_h
    return (fcx, cy, crop_h)


def _targets(analysis: ClipAnalysis, i: int, mode: str, tids: list[int],
             src_w: int, src_h: int,
             prefer_coarse: bool = False) -> list[tuple[float, float, float]]:
    """(cx, cy, crop_h) targets in source px — one per rect slot."""
    coarse = analysis.coarse_faces[i] if i < len(analysis.coarse_faces) else []
    valid_coarse = [
        face for face in coarse
        if len(face) >= 4 and face[2] > 0 and face[3] > 0
    ]

    if not tids:
        # FaceMesh can be blacked out while the independent coarse detector
        # still has a clear face. Prefer that face over empty background.
        if mode == "single" and valid_coarse:
            face = max(valid_coarse, key=lambda b: b[2] * b[3])
            return [_target_from_box(face, mode, src_w, src_h)]
        return [(src_w / 2, src_h / 2, float(src_h))]

    # A high-rate speaker track can remain selected while its box is stale.
    # Match the independent coarse detector to its last position rather than
    # aiming a tight crop at an empty chair. Just after a source shot cut, one
    # coarse face is unambiguous and wins even if the stale track is still
    # inside its visibility grace window.
    if mode == "single" and len(tids) == 1 and valid_coarse:
        tid = tids[0]
        tracked = analysis.boxes[tid][i]
        if not analysis.visible[tid][i] or (prefer_coarse and len(valid_coarse) == 1):
            face = min(
                valid_coarse,
                key=lambda b: (b[0] - tracked[0]) ** 2 + (b[1] - tracked[1]) ** 2,
            )
            return [_target_from_box(face, mode, src_w, src_h)]

    return [_target_from_box(analysis.boxes[tid][i], mode, src_w, src_h) for tid in tids]


def split_rects_iou(box_a, box_b, src_w: int, src_h: int) -> float:
    """IoU of the two SOURCE crop rects a split would use for these two faces.
    Same crop math as plan() itself (single source of truth), same IoU measure
    as the QA gate — used by speaker's state machine to refuse a split whose
    halves would show (mostly) the same region. Faces can be 'distinct' by
    center distance while their crops (~3x face height wide) still overlap
    heavily; that gap produced rendered clips QA then flagged with
    "crop rects overlap N% — same source region"."""
    from app.pipeline.qa import _rect_iou

    ra = _clamp_rect(*_target_from_box(box_a, "split", src_w, src_h),
                     SPLIT_ASPECT, src_w, src_h)
    rb = _clamp_rect(*_target_from_box(box_b, "split", src_w, src_h),
                     SPLIT_ASPECT, src_w, src_h)
    return float(_rect_iou(ra, rb))


def _clamp_rect(cx: float, cy: float, crop_h: float, aspect: float,
                src_w: int, src_h: int) -> tuple[float, float, float, float]:
    crop_h = min(crop_h, src_h, src_w / aspect)
    crop_w = crop_h * aspect
    x = np.clip(cx - crop_w / 2, 0, src_w - crop_w)
    y = np.clip(cy - crop_h / 2, 0, src_h - crop_h)
    return (float(x), float(y), float(crop_w), float(crop_h))
