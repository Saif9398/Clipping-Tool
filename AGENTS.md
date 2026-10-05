# Clipping Tool contributor notes

This is an independent local video-processing application. Keep application
changes confined to this repository and to its own runtime directories.

## Development

- Python 3.12; install `requirements.lock.txt` in a virtual environment.
- Run `python -m app.main` for the UI or `python -m app.cli --help` for the CLI.
- Cache public vision models before tests:
  `python -c "from app.pipeline.mp_models import ensure_models; ensure_models()"`.
- Install `requirements-dev.txt`, then run `python -m pytest tests/ -v`.

## Pipeline conventions

- Keep heavy imports such as OpenCV, MediaPipe and faster-whisper inside functions
  where possible; Windows worker processes re-import modules when spawned.
- Functions submitted to process pools must be top-level and picklable.
- Use relative FFmpeg filter-file names with the working directory set explicitly;
  absolute Windows paths inside filter expressions are fragile.
- Preserve frame-count-driven rendering and the 1080×1920 output contract.
- Keep face coordinates normalized until the camera planner converts to source pixels.
- Use synthetic fixtures in tests. Never include private videos, credentials or user data.
- Never add cloud lifecycle controls or automatic container publishing without a
  separate design and security review.

## Licensing

The project license is pending. Keep attribution and third-party notices intact.
See `docs/ATTRIBUTION.md`; do not vendor code from an unlicensed reference project.
