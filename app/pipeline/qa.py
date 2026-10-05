"""Post-render quality gate — runs after every clip encode, before it may be
marked done. Numeric checks (frame count/resolution) live in renderer._verify;
this module catches the VISUAL defect classes that slipped past them:

1. split-screen showing the same person twice   -> crop-rect geometry + pixel
   similarity of the two rendered halves
2. overlapping caption events (doubled text)    -> parse the generated .ass
3. broken frames (black/uniform)                -> sampled-frame statistics

Sample frames are saved to <output>/qa/clip_NN/ for human review. Any flag
marks the clip "review" instead of "done" — never silently ship a broken clip.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SPLIT_RECT_IOU_HARD = 0.50   # crop rects overlap this much -> same region, fail outright
SPLIT_RECT_IOU_SOFT = 0.20   # above this, corroborate with pixel similarity
SPLIT_NCC_FAIL = 0.80        # normalized cross-correlation of the two halves
UNIFORM_STD_MIN = 5.0        # grayscale std below this = black/uniform frame
N_SAMPLE_FRAMES = 5


def run_qa(out_path: Path, clip_dir: Path, plans: list, fps: float, qa_dir: Path) -> dict:
    flags: list[str] = []
    qa_dir.mkdir(parents=True, exist_ok=True)

    split_runs = _split_runs(plans)
    flags += _check_split_geometry(plans, split_runs)

    # sample only outside the fade-in/out zones (frame 0 is legitimately black)
    lo = int(0.4 * fps)
    hi = max(lo + 1, len(plans) - 1 - int(0.4 * fps))
    sample_idxs = sorted(
        {lo + int(k * (hi - lo) / (N_SAMPLE_FRAMES - 1)) for k in range(N_SAMPLE_FRAMES)}
        | {min(hi, max(lo, (f0 + f1) // 2)) for f0, f1 in split_runs}
    )
    frame_flags, frame_files, frames_by_idx = _sample_frames(out_path, sample_idxs, qa_dir)
    flags += frame_flags

    for f0, f1 in split_runs:
        mid = (f0 + f1) // 2
        img = frames_by_idx.get(mid)
        if img is None:
            continue
        ncc = _halves_similarity(img)
        top, bottom = plans[mid].rects[0], plans[mid].rects[1]
        if ncc > SPLIT_NCC_FAIL and _rect_iou(top, bottom) > SPLIT_RECT_IOU_SOFT:
            flags.append(
                f"split at {mid / fps:.1f}s: halves {ncc:.2f} similar — likely same person duplicated")

    flags += _check_ass_overlaps(clip_dir / "captions.ass")

    result = {"passed": not flags, "flags": flags, "frames": frame_files}
    if flags:
        log.warning("QA flagged %s: %s", out_path.name, "; ".join(flags))
    return result


def _split_runs(plans: list) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, p in enumerate(plans):
        if p.mode == "split" and len(p.rects) == 2:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(plans)))
    return runs


def _rect_iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _check_split_geometry(plans: list, split_runs: list) -> list[str]:
    flags = []
    for f0, f1 in split_runs:
        # test start/mid/end of each run — a run can drift into coincidence
        for i in {f0, (f0 + f1) // 2, f1 - 1}:
            top, bottom = plans[i].rects[0], plans[i].rects[1]
            iou = _rect_iou(top, bottom)
            if iou > SPLIT_RECT_IOU_HARD:
                flags.append(
                    f"split frames {f0}-{f1}: crop rects overlap {iou:.0%} — same source region")
                break
    return flags


def _sample_frames(out_path: Path, idxs: list[int], qa_dir: Path):
    import cv2

    flags, files, by_idx = [], [], {}
    cap = cv2.VideoCapture(str(out_path))
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            flags.append(f"could not decode frame {idx} from finished clip")
            continue
        by_idx[idx] = frame
        gray_std = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).std())
        if gray_std < UNIFORM_STD_MIN:
            flags.append(f"frame {idx} is near-uniform (std {gray_std:.1f}) — black/broken frame")
        fp = qa_dir / f"frame_{idx:05d}.jpg"
        cv2.imwrite(str(fp), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        files.append(fp.name)
    cap.release()
    return flags, files, by_idx


def _halves_similarity(frame: np.ndarray) -> float:
    """NCC between the rendered top and bottom halves (downscaled grayscale).
    The same face duplicated in both halves scores near 1.0; two different
    people (different faces/clothing/background) score well under 0.8."""
    import cv2

    h = frame.shape[0] // 2
    small = lambda img: cv2.resize(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 114)).astype(np.float32)
    a, b = small(frame[:h]), small(frame[h:2 * h])
    a -= a.mean()
    b -= b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


_DLG = re.compile(r"^Dialogue: \d+,(\d+):(\d\d):(\d\d)\.(\d\d),(\d+):(\d\d):(\d\d)\.(\d\d),")


def _check_ass_overlaps(ass_path: Path) -> list[str]:
    if not ass_path.exists():
        return ["captions.ass missing from clip work dir"]
    events = []
    for line in ass_path.read_text(encoding="utf-8-sig").splitlines():
        m = _DLG.match(line)
        if m:
            g = [int(x) for x in m.groups()]
            events.append((g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 100,
                           g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 100))
    events.sort()
    flags = []
    for (s1, e1), (s2, _) in zip(events, events[1:]):
        if e1 - s2 > 0.011:  # >1cs of true overlap -> two caption lines at once
            flags.append(f"caption events overlap at {s2:.2f}s ({e1 - s2:.2f}s of double text)")
            if len(flags) >= 3:
                flags.append("... further caption overlaps suppressed")
                break
    return flags
