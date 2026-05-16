import io
import os
import unittest
from unittest.mock import patch

import pandas as pd

import app as feedback_app


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


if __name__ == "__main__":
    unittest.main()
