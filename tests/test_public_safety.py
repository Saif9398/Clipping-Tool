"""Safety regressions: local shutdown cannot invoke a provider lifecycle API.

Run with public vision model files cached, using a credential-free test instance.
"""
from __future__ import annotations

import os
import unittest
import urllib.request
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import config
from app import main


class ProductionIsolationTests(unittest.TestCase):
    def test_stop_stays_local_even_with_provider_environment(self):
        # Deliberately artificial values. Inherited provider variables must not
        # activate infrastructure controls in this independent application.
        provider_env = {"RUNPOD_POD_ID": "synthetic-unused-pod",
                        "RUNPOD_API_KEY": "synthetic-unused-value"}
        old_flags = main._shutdown_requested, main._restart_requested
        try:
            main._shutdown_requested = main._restart_requested = False
            with (patch.dict(os.environ, provider_env),
                  patch.object(config, "ADMIN_PASSWORD", ""),
                  patch.object(config, "ADMIN_PASSWORD_SHA256", ""),
                  patch.object(main.jobs, "cancel_active_jobs_for_shutdown", return_value=[]),
                  patch.object(main, "_schedule_process_shutdown") as stop,
                  patch.object(urllib.request, "urlopen", side_effect=AssertionError("outbound lifecycle request"))):
                response = TestClient(main.app).post("/api/server/stop", json={})
            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.json(), {"ok": True, "already_stopping": False,
                                              "cancelled_jobs": []})
            stop.assert_called_once_with()
        finally:
            main._shutdown_requested, main._restart_requested = old_flags

    def test_private_settings_and_session_are_not_returned(self):
        # An accidentally added settings field would expose user-supplied keys.
        with (patch.object(config, "ADMIN_PASSWORD", ""),
              patch.object(config, "ADMIN_PASSWORD_SHA256", ""),
              patch.object(config, "OPENAI_API_KEY", "synthetic-private-value"),
              patch.object(config, "SESSION_SECRET", "synthetic-session-value")):
            response = TestClient(main.app).get("/api/settings")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()), {"scoring_provider", "has_openai_key", "has_gemini_key"})
        self.assertNotIn("synthetic-private-value", response.text)
        self.assertNotIn("synthetic-session-value", response.text)


if __name__ == "__main__":
    unittest.main()
