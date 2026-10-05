# Attribution and provenance

The project-wide license has not been selected. This document records design
references and dependency boundaries; it does not grant a license to this project
or to third-party material.

## Design references

- [AI-Youtube-Shorts-Generator](https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator):
  reference for transcript highlight ranking and short-form editorial scoring criteria.
- [openshorts](https://github.com/mutonby/openshorts): reference for the broad
  transcription, scene-analysis and vertical-video architecture.
- [smart-reframe](https://github.com/gauravzazz/smart-reframe): reference for
  dead-zone smoothing, camera velocity limits and mouth/audio speaker evidence.
  The inspected upstream repository declares no license. Do not vendor its code.
  This project's camera planner uses an error-adaptive smoother, frame plans and
  visibility handling; its speaker analysis uses normalized tracks, rolling mouth
  variance, transcript/audio gates and a single/split state machine. Comparison
  against the referenced upstream smoother and orchestration files found no
  multi-line exact source blocks beyond individual common statements.

These comparisons are a targeted provenance check, not a complete legal clearance.
Keep the distinction between algorithmic ideas and copied source explicit. Recheck
provenance and permissions before selecting a commercial distribution model.

## Installed dependencies and downloaded models

Python libraries are installed through the requirements files; their source and
license notices are not vendored here. Third-party packages keep their own terms.
The pipeline downloads MediaPipe face detector / landmark weights, the OpenCV
YuNet detector and the selected Whisper model at runtime. Their model terms are
separate from the application's license. Review upstream model documentation before
commercial use or redistributing weights.

FFmpeg and the optional NVIDIA CUDA runtime also have their own distribution terms.
Building or distributing a container requires checking those components and the
included codec build. The local Docker example is not a license grant.

No project-wide MIT, Apache, GPL or other license has been added. Preserve any
third-party notices required by components you later vendor or redistribute.
