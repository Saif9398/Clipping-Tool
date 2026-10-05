"""Stage 2b: scene-boundary detection with PySceneDetect.

Output: work_dir/scenes.json  -> {"boundaries": [t0, t1, ...]} (seconds, sorted)
Boundaries are used to (a) seed candidate windows and (b) force camera cuts in the renderer
so a source-video cut never gets smoothed into a smeary pan.
"""
from __future__ import annotations

import json
from pathlib import Path


def detect_scenes(work_dir: Path, progress_cb=None) -> list[float]:
    from scenedetect import ContentDetector, open_video, SceneManager

    video = open_video(str(work_dir / "source.mp4"))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=27.0, min_scene_len=15))
    total = video.duration.get_frames() or 1
    # callback fires per detected cut; podcasts cut often enough for coarse progress.
    # position arrives as a FrameTimecode — convert before arithmetic.
    cb = (lambda _img, pos: progress_cb(min(1.0, pos.get_frames() / total))) if progress_cb else None
    # Downscale is automatic (scenedetect picks a factor from frame width)
    sm.detect_scenes(video, show_progress=False, callback=cb)
    scene_list = sm.get_scene_list()

    boundaries = sorted({round(s.get_seconds(), 3) for s, _ in scene_list} |
                        {round(e.get_seconds(), 3) for _, e in scene_list})
    (work_dir / "scenes.json").write_text(json.dumps({"boundaries": boundaries}), encoding="utf-8")
    return boundaries
