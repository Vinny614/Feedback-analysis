import io
import os
import threading
import time
import unittest
from unittest.mock import patch

import pandas as pd

import app as feedback_app

FEEDBACK_TEXT_INDEX = 0
AZURE_SENTIMENT_INDEX = 1
LANGUAGE_MODEL_SENTIMENT_INDEX = 7
ASYNC_TEST_TIMEOUT_SECONDS = 2


class AppTests(unittest.TestCase):
    def setUp(self):
        feedback_app.app.config["TESTING"] = True
        self.client = feedback_app.app.test_client()
        self._env = os.environ.copy()
        os.environ["DEMO_USE_MOCK_ANALYZERS"] = "true"
        os.environ["MAX_UPLOAD_ROWS"] = "200"
        with feedback_app._jobs_lock:
            feedback_app._analysis_jobs.clear()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)
        with feedback_app._jobs_lock:
            feedback_app._analysis_jobs.clear()

    @staticmethod
    def _excel_bytes(df: pd.DataFrame) -> io.BytesIO:
        payload = io.BytesIO()
        with pd.ExcelWriter(payload, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        payload.seek(0)
        return payload

    def test_upload_enqueues_job_and_status_returns_progress_payload(self):
        df = pd.DataFrame({"Feedback": ["Great support", "Needs improvement"]})
        file_obj = self._excel_bytes(df)

        with patch.object(
            feedback_app._job_executor,
            "submit",
            side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs),
        ):
            response = self.client.post(
                "/",
                data={"feedback_file": (file_obj, "feedback.xlsx")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        with feedback_app._jobs_lock:
            self.assertEqual(len(feedback_app._analysis_jobs), 1)
            job_id = next(iter(feedback_app._analysis_jobs))

        status_response = self.client.get(f"/jobs/{job_id}/status")
        self.assertEqual(status_response.status_code, 200)
        payload = status_response.get_json()
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["processed_rows"], 2)
        self.assertEqual(payload["total_rows"], 2)
        self.assertTrue(payload["download_url"])
        self.assertIn("azure_sentiment", payload["table_columns"])
        self.assertIn("overall_summary", payload)
        self.assertIn("trend", payload["overall_summary"])
        self.assertIn("recommended_actions", payload["overall_summary"])

    def test_upload_rejects_files_above_max_rows(self):
        os.environ["MAX_UPLOAD_ROWS"] = "1"
        df = pd.DataFrame({"Feedback": ["Great support", "Needs improvement"]})
        file_obj = self._excel_bytes(df)

        response = self.client.post(
            "/",
            data={"feedback_file": (file_obj, "feedback.xlsx")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("at most 1 row", response.get_data(as_text=True))

    def test_upload_accepts_file_at_max_rows_boundary(self):
        os.environ["MAX_UPLOAD_ROWS"] = "2"
        df = pd.DataFrame({"Feedback": ["Great support", "Needs improvement"]})
        file_obj = self._excel_bytes(df)

        with patch.object(
            feedback_app._job_executor,
            "submit",
            side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs),
        ):
            response = self.client.post(
                "/",
                data={"feedback_file": (file_obj, "feedback.xlsx")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        with feedback_app._jobs_lock:
            self.assertEqual(len(feedback_app._analysis_jobs), 1)

    def test_model_label_prefers_model_name_and_version_over_deployment_name(self):
        os.environ["PHI_DEPLOYMENT_NAME"] = "chat-model"
        os.environ["PHI_MODEL_NAME"] = "gpt-4.1-mini"
        os.environ["PHI_MODEL_VERSION"] = "2025-04-14"

        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("gpt-4.1-mini (2025-04-14)", page)
        self.assertNotIn(">chat-model<", page)

    def test_status_returns_partial_table_updates_while_job_is_running(self):
        df = pd.DataFrame({"Feedback": ["Great support", "Needs improvement"]})
        file_obj = self._excel_bytes(df)
        phi_started = threading.Event()
        allow_phi_completion = threading.Event()

        class AzureStub:
            def analyze(self, texts):
                return [
                    {
                        "sentiment": "positive",
                        "opinion_mining": [{"target": "support", "sentiment": "positive"}],
                        "key_phrases": ["great support"],
                        "confidence_positive": 0.95,
                        "confidence_neutral": 0.05,
                        "confidence_negative": 0.0,
                    },
                    {
                        "sentiment": "negative",
                        "opinion_mining": [{"target": "improvement", "sentiment": "negative"}],
                        "key_phrases": ["needs improvement"],
                        "confidence_positive": 0.05,
                        "confidence_neutral": 0.1,
                        "confidence_negative": 0.85,
                    },
                ]

        class PhiStub:
            def analyze(self, texts):
                phi_started.set()
                if not allow_phi_completion.wait(timeout=ASYNC_TEST_TIMEOUT_SECONDS):
                    raise TimeoutError("Timed out waiting to finish language model analysis.")
                return [
                    {
                        "sentiment": "positive",
                        "opinion_mining": ["helpful support"],
                        "key_phrases": ["support"],
                    },
                    {
                        "sentiment": "negative",
                        "opinion_mining": ["needs improvement"],
                        "key_phrases": ["improvement"],
                    },
                ]

        with patch.object(
            feedback_app, "_create_analyzers", return_value=(AzureStub(), PhiStub(), "Demo heuristic analyzer")
        ):
            response = self.client.post(
                "/",
                data={"feedback_file": (file_obj, "feedback.xlsx")},
                content_type="multipart/form-data",
            )

            self.assertEqual(response.status_code, 200)
            self.assertTrue(phi_started.wait(timeout=ASYNC_TEST_TIMEOUT_SECONDS))

            with feedback_app._jobs_lock:
                job_id = next(iter(feedback_app._analysis_jobs))

            interim_response = self.client.get(f"/jobs/{job_id}/status")
            self.assertEqual(interim_response.status_code, 200)
            interim_payload = interim_response.get_json()
            self.assertEqual(interim_payload["status"], "running")
            self.assertEqual(interim_payload["processed_rows"], 0)
            self.assertEqual(
                interim_payload["table_rows"][0][FEEDBACK_TEXT_INDEX], "Great support"
            )
            self.assertEqual(
                interim_payload["table_rows"][0][AZURE_SENTIMENT_INDEX], "positive"
            )
            self.assertEqual(
                interim_payload["table_rows"][0][LANGUAGE_MODEL_SENTIMENT_INDEX], ""
            )
            self.assertIsNone(interim_payload["download_url"])

            allow_phi_completion.set()
            deadline = time.monotonic() + ASYNC_TEST_TIMEOUT_SECONDS
            final_payload = None
            while time.monotonic() < deadline:
                final_response = self.client.get(f"/jobs/{job_id}/status")
                final_payload = final_response.get_json()
                if final_payload["status"] == "completed":
                    break
                time.sleep(0.05)

            self.assertIsNotNone(final_payload)
            self.assertEqual(final_payload["status"], "completed")
            self.assertEqual(final_payload["processed_rows"], 2)
            self.assertEqual(
                final_payload["table_rows"][0][LANGUAGE_MODEL_SENTIMENT_INDEX], "positive"
            )
            self.assertTrue(final_payload["download_url"])


if __name__ == "__main__":
    unittest.main()
