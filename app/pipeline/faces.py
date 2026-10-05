"""Stage 2c: coarse face-track pre-computation across the whole video.

Sampled at ~3 fps on downscaled frames — cheap enough to run over a 75-min video
while transcription runs in a sibling process. The renderer later refines only the
approved clip ranges at higher temporal resolution (FaceMesh), so this pass just
needs to know where faces are and how many there are.

Output: work_dir/faces_coarse.json
  {"sample_fps": ~3, "entries": [{"t": sec, "f": [[cx, cy, w, h, conf], ...]}]}
  All coordinates normalized 0..1 relative to the full source frame.
"""
from __future__ import annotations

import json
from pathlib import Path

TARGET_SAMPLE_FPS = 3.0
DETECT_WIDTH = 480


def detect_faces_coarse(work_dir: Path, progress_cb=None) -> dict:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    from app.pipeline.mp_models import model_path

    cap = cv2.VideoCapture(str(work_dir / "source.mp4"))
    if not cap.isOpened():
        raise RuntimeError("cannot open source.mp4 for face detection")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, round(src_fps / TARGET_SAMPLE_FPS))

    entries = []
    opts = vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(model_path("blaze_face_short_range.tflite"))),
        running_mode=vision.RunningMode.VIDEO,
        min_detection_confidence=0.4,
    )
    with vision.FaceDetector.create_from_options(opts) as detector:
        idx = 0
        while True:
            if not cap.grab():
                break
            if idx % stride == 0:
                ok, frame = cap.retrieve()
                if ok:
                    h, w = frame.shape[:2]
                    scale = DETECT_WIDTH / w
                    sw, sh = DETECT_WIDTH, int(h * scale)
                    small = cv2.resize(frame, (sw, sh))
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    res = detector.detect_for_video(img, int(idx / src_fps * 1000))
                    faces = []
                    for det in (res.detections or []):
                        bb = det.bounding_box  # pixel coords of the downscaled frame
                        faces.append([
                            round((bb.origin_x + bb.width / 2) / sw, 4),
                            round((bb.origin_y + bb.height / 2) / sh, 4),
                            round(bb.width / sw, 4),
                            round(bb.height / sh, 4),
                            round(det.categories[0].score, 3) if det.categories else 0.5,
                        ])
                    entries.append({"t": round(idx / src_fps, 3), "f": faces})
                if progress_cb and total:
                    progress_cb(min(1.0, idx / total))
            idx += 1
    cap.release()

    result = {"sample_fps": src_fps / stride, "entries": entries}
    (work_dir / "faces_coarse.json").write_text(json.dumps(result), encoding="utf-8")
    return result


def faces_at(coarse: dict, t: float) -> list[list[float]]:
    """Nearest-sample lookup into the coarse face data."""
    entries = coarse["entries"]
    if not entries:
        return []
    import bisect

    times = [e["t"] for e in entries]
    i = bisect.bisect_left(times, t)
    if i <= 0:
        return entries[0]["f"]
    if i >= len(entries):
        return entries[-1]["f"]
    return entries[i]["f"] if abs(times[i] - t) < abs(times[i - 1] - t) else entries[i - 1]["f"]
