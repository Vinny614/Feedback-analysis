import os
import unittest
from unittest.mock import MagicMock, patch

import app as feedback_app
import news_summariser


class TestClassifySourceLean(unittest.TestCase):
    def test_known_sources_returned_from_dict(self):
        self.assertEqual(news_summariser.classify_source_lean("BBC"), "centre")
        self.assertEqual(news_summariser.classify_source_lean("bbc news"), "centre")
        self.assertEqual(news_summariser.classify_source_lean("Reuters"), "centre")
        self.assertEqual(news_summariser.classify_source_lean("The Guardian"), "centre-left")
        self.assertEqual(news_summariser.classify_source_lean("guardian"), "centre-left")
        self.assertEqual(news_summariser.classify_source_lean("The Telegraph"), "centre-right")
        self.assertEqual(news_summariser.classify_source_lean("Daily Mail"), "right")
        self.assertEqual(news_summariser.classify_source_lean("Fox News"), "right")
        self.assertEqual(news_summariser.classify_source_lean("Financial Times"), "centre-right")
        self.assertEqual(news_summariser.classify_source_lean("The Washington Post"), "centre-left")

    def test_known_source_case_insensitive(self):
        self.assertEqual(news_summariser.classify_source_lean("BBC NEWS"), "centre")
        self.assertEqual(news_summariser.classify_source_lean("  BBC  "), "centre")

    def test_unknown_source_without_openai_returns_unknown(self):
        lean = news_summariser.classify_source_lean(
            "Some Obscure Local Paper",
            openai_endpoint="",
            openai_deployment="",
        )
        self.assertEqual(lean, "unknown")

    def test_unknown_source_with_openai_returns_valid_lean(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "centre-left"}}]
        }

        with patch.object(news_summariser, "_post_json_with_retry", return_value=mock_response):
            lean = news_summariser.classify_source_lean(
                "Some Unknown Outlet",
                openai_endpoint="https://example.openai.azure.com",
                openai_deployment="chat-model",
                openai_api_version="2025-01-01-preview",
                openai_api_key="fake-key",
            )
        self.assertEqual(lean, "centre-left")

    def test_unknown_source_openai_invalid_response_returns_unknown(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "i don't know, maybe moderate?"}}]
        }

        with patch.object(news_summariser, "_post_json_with_retry", return_value=mock_response):
            lean = news_summariser.classify_source_lean(
                "Mystery Paper",
                openai_endpoint="https://example.openai.azure.com",
                openai_deployment="chat-model",
                openai_api_version="2025-01-01-preview",
                openai_api_key="fake-key",
            )
        self.assertEqual(lean, "unknown")

    def test_unknown_source_openai_exception_returns_unknown(self):
        with patch.object(news_summariser, "_post_json_with_retry", side_effect=Exception("timeout")):
            lean = news_summariser.classify_source_lean(
                "Mystery Paper",
                openai_endpoint="https://example.openai.azure.com",
                openai_deployment="chat-model",
                openai_api_version="2025-01-01-preview",
                openai_api_key="fake-key",
            )
        self.assertEqual(lean, "unknown")


class TestFetchArticleText(unittest.TestCase):
    def test_returns_article_body_text(self):
        html = """
        <html><body>
        <nav>Navigation</nav>
        <article>
          <p>This is the main story content about an important event.</p>
          <p>More details follow here.</p>
        </article>
        </body></html>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            text = news_summariser.fetch_article_text("http://example.com/article")

        self.assertIn("main story content", text)
        self.assertNotIn("Navigation", text)

    def test_falls_back_to_paragraphs_when_no_article_tag(self):
        html = """
        <html><body>
        <div class="content">
          <p>Paragraph one.</p>
          <p>Paragraph two.</p>
        </div>
        </body></html>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            text = news_summariser.fetch_article_text("http://example.com/article")

        self.assertIn("Paragraph one", text)
        self.assertIn("Paragraph two", text)

    def test_returns_empty_string_on_request_failure(self):
        import requests as req
        with patch("requests.get", side_effect=req.Timeout()):
            text = news_summariser.fetch_article_text("http://example.com/article")

        self.assertEqual(text, "")

    def test_truncates_at_word_limit(self):
        many_words = " ".join([f"word{i}" for i in range(2000)])
        html = f"<html><body><article><p>{many_words}</p></article></body></html>"
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            text = news_summariser.fetch_article_text("http://example.com/article")

        self.assertLessEqual(len(text.split()), news_summariser._ARTICLE_WORD_LIMIT)

    def test_malformed_html_does_not_raise(self):
        mock_response = MagicMock()
        mock_response.text = "<html><bod<p>broken<p>html</html"
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            try:
                news_summariser.fetch_article_text("http://example.com/broken")
            except Exception as exc:  # pragma: no cover
                self.fail(f"fetch_article_text raised an exception on malformed HTML: {exc}")


class TestGenerateBalancedSummary(unittest.TestCase):
    def _make_grouped_articles(self):
        return {
            "left": [
                {
                    "title": "Left piece",
                    "url": "http://left.example.com",
                    "source_name": "The Guardian",
                    "lean": "left",
                    "body": "Left argument about policy.",
                    "description": "",
                }
            ],
            "centre-left": [],
            "centre": [
                {
                    "title": "Wire piece",
                    "url": "http://reuters.example.com",
                    "source_name": "Reuters",
                    "lean": "centre",
                    "body": "Neutral report on policy.",
                    "description": "",
                }
            ],
            "centre-right": [],
            "right": [
                {
                    "title": "Right piece",
                    "url": "http://right.example.com",
                    "source_name": "Daily Mail",
                    "lean": "right",
                    "body": "Right argument about policy.",
                    "description": "",
                }
            ],
            "unknown": [],
        }

    def test_returns_correct_structure(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"overview":"Overview text.","left_arguments":["Arg A"],"right_arguments":["Arg B"],"consensus":["Point X"]}'
                    }
                }
            ]
        }

        grouped = self._make_grouped_articles()

        with patch.object(news_summariser, "_post_json_with_retry", return_value=mock_response):
            result = news_summariser.generate_balanced_summary(
                topic="test topic",
                grouped_articles=grouped,
                openai_endpoint="https://example.openai.azure.com",
                openai_deployment="chat-model",
                openai_api_version="2025-01-01-preview",
                openai_api_key="fake-key",
            )

        self.assertIn("overview", result)
        self.assertIn("left_arguments", result)
        self.assertIn("right_arguments", result)
        self.assertIn("consensus", result)
        self.assertIn("sources", result)
        self.assertIn("caveat", result)
        self.assertEqual(result["overview"], "Overview text.")
        self.assertEqual(result["left_arguments"], ["Arg A"])
        self.assertEqual(result["right_arguments"], ["Arg B"])
        self.assertEqual(result["consensus"], ["Point X"])
        self.assertIsInstance(result["sources"], list)
        self.assertEqual(len(result["sources"]), 3)

    def test_caveat_set_when_only_one_perspective(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"overview":"Overview.","left_arguments":[],"right_arguments":[],"consensus":[]}'
                    }
                }
            ]
        }

        grouped = {k: [] for k in ("left", "centre-left", "centre", "centre-right", "right", "unknown")}
        grouped["centre"] = [
            {
                "title": "Centre piece",
                "url": "http://centre.example.com",
                "source_name": "Reuters",
                "lean": "centre",
                "body": "Report.",
                "description": "",
            }
        ]

        with patch.object(news_summariser, "_post_json_with_retry", return_value=mock_response):
            result = news_summariser.generate_balanced_summary(
                topic="test",
                grouped_articles=grouped,
                openai_endpoint="https://example.openai.azure.com",
                openai_deployment="chat-model",
                openai_api_version="2025-01-01-preview",
                openai_api_key="fake-key",
            )

        self.assertIn("caveat", result)
        self.assertTrue(result["caveat"])

    def test_raises_value_error_when_endpoint_missing(self):
        grouped = {k: [] for k in ("left", "centre-left", "centre", "centre-right", "right", "unknown")}
        with self.assertRaises(ValueError):
            news_summariser.generate_balanced_summary(
                topic="test",
                grouped_articles=grouped,
                openai_endpoint="",
                openai_deployment="",
            )


class TestNewsSummariserFlaskRoutes(unittest.TestCase):
    def setUp(self):
        feedback_app.app.config["TESTING"] = True
        self.client = feedback_app.app.test_client()
        self._env = os.environ.copy()
        os.environ["BING_SEARCH_KEY"] = "fake-bing-key"
        with feedback_app._jobs_lock:
            feedback_app._analysis_jobs.clear()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)
        with feedback_app._jobs_lock:
            feedback_app._analysis_jobs.clear()

    def test_get_renders_page(self):
        response = self.client.get("/news-summariser")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"News Topic Summariser", response.data)

    def test_post_empty_topic_returns_error(self):
        response = self.client.post("/news-summariser", data={"topic": "", "freshness": "Month"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Please enter a news topic", response.data)

    def test_post_topic_too_long_returns_error(self):
        long_topic = "x" * 201
        response = self.client.post("/news-summariser", data={"topic": long_topic, "freshness": "Month"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"200 characters", response.data)

    def test_post_missing_bing_key_returns_error(self):
        del os.environ["BING_SEARCH_KEY"]
        response = self.client.post("/news-summariser", data={"topic": "climate change", "freshness": "Month"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"BING_SEARCH_KEY", response.data)

    def test_post_valid_topic_enqueues_job_and_returns_status_url(self):
        with patch.object(
            feedback_app._job_executor,
            "submit",
            return_value=MagicMock(),
        ):
            response = self.client.post("/news-summariser", data={"topic": "climate change", "freshness": "Month"})

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"news-summariser/status", response.data)
        with feedback_app._jobs_lock:
            self.assertEqual(len(feedback_app._analysis_jobs), 1)
            job_id = next(iter(feedback_app._analysis_jobs))
            job = feedback_app._analysis_jobs[job_id]
        self.assertEqual(job["type"], "news")
        self.assertEqual(job["status"], "queued")

    def test_status_endpoint_returns_404_for_unknown_job(self):
        response = self.client.get("/news-summariser/status/nonexistent")
        self.assertEqual(response.status_code, 404)

    def test_status_endpoint_returns_job_state(self):
        job_id = "testjob123"
        with feedback_app._jobs_lock:
            feedback_app._analysis_jobs[job_id] = {
                "type": "news",
                "status": "completed",
                "error": None,
                "result": {
                    "overview": "Test overview.",
                    "left_arguments": ["Arg A"],
                    "right_arguments": ["Arg B"],
                    "consensus": ["Point X"],
                    "sources": [],
                    "caveat": "",
                },
            }

        response = self.client.get(f"/news-summariser/status/{job_id}")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "completed")
        self.assertIn("result", payload)
        self.assertEqual(payload["result"]["overview"], "Test overview.")

    def test_status_endpoint_rejects_non_news_job_id(self):
        job_id = "feedbackjob456"
        with feedback_app._jobs_lock:
            feedback_app._analysis_jobs[job_id] = {
                "status": "completed",
                "processed_rows": 1,
                "total_rows": 1,
                "feedback_column": "Feedback",
                "language_model_used": "demo",
                "error": None,
                "download_url": None,
                "output_df": None,
            }

        response = self.client.get(f"/news-summariser/status/{job_id}")
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
