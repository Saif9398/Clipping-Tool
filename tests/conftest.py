"""Keep each test's mutable runtime data in its own credential-free directory."""
import pytest

from app import config


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    # Rendering, cache cleanup and retention tests all mutate runtime folders.
    # Sharing those folders made the suite sensitive to leftovers from prior runs.
    work, output = tmp_path / "work", tmp_path / "output"
    work.mkdir()
    output.mkdir()
    monkeypatch.setattr(config, "WORK_DIR", work)
    monkeypatch.setattr(config, "OUTPUT_DIR", output)
    for name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ADMIN_PASSWORD",
                 "ADMIN_PASSWORD_SHA256", "SESSION_SECRET"):
        monkeypatch.setattr(config, name, "")
    monkeypatch.setattr(config, "SCORING_PROVIDER", "local")
    from tests import test_synthetic
    synthetic_work, synthetic_output = work / "synthetic", output / "synthetic"
    synthetic_work.mkdir()
    synthetic_output.mkdir()
    monkeypatch.setattr(test_synthetic, "WORK", synthetic_work)
    monkeypatch.setattr(test_synthetic, "OUT", synthetic_output)
