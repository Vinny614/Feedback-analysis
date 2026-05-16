import json
import unittest
from unittest.mock import patch

import pandas as pd
import requests

from feedback_analysis import (
    MAX_RETRY_ATTEMPTS,
    _post_with_retry,
    _build_service_headers,
    PhiAnalyzer,
    detect_feedback_column,
    enrich_feedback_dataframe,
)


class StubAnalyzer:
    def __init__(self, rows):
        self.rows = rows

    def analyze(self, texts):
        return self.rows


class FeedbackAnalysisTests(unittest.TestCase):
    @staticmethod
    def _build_response(status_code, payload, headers=None):
        response = requests.Response()
        response.status_code = status_code
        response._content = json.dumps(payload).encode("utf-8")
        response.headers = headers or {}
        response.url = "https://example.test"
        return response

    def test_detect_feedback_column_prefers_named_column(self):
        df = pd.DataFrame({"ID": [1], "Feedback": ["Great service"]})
        self.assertEqual(detect_feedback_column(df), "Feedback")

    def test_enrich_feedback_dataframe_adds_expected_columns(self):
        df = pd.DataFrame({"Feedback": ["Great support", "Needs improvement"]})

        azure_stub = StubAnalyzer(
            [
                {
                    "sentiment": "positive",
                    "opinion_mining": [{"target": "support", "sentiment": "positive"}],
                    "key_phrases": ["great support"],
                    "confidence_positive": 0.99,
                    "confidence_neutral": 0.01,
                    "confidence_negative": 0.0,
                },
                {
                    "sentiment": "negative",
                    "opinion_mining": [{"target": "improvement", "sentiment": "negative"}],
                    "key_phrases": ["needs improvement"],
                    "confidence_positive": 0.01,
                    "confidence_neutral": 0.1,
                    "confidence_negative": 0.89,
                },
            ]
        )
        phi_stub = StubAnalyzer(
            [
                {
                    "sentiment": "positive",
                    "opinion_mining": ["support team helpful"],
                    "key_phrases": ["support", "helpful"],
                },
                {
                    "sentiment": "negative",
                    "opinion_mining": ["slow response"],
                    "key_phrases": ["slow", "response"],
                },
            ]
        )

        enriched = enrich_feedback_dataframe(df, "Feedback", azure_stub, phi_stub)

        self.assertIn("azure_sentiment", enriched.columns)
        self.assertIn("azure_opinion_mining", enriched.columns)
        self.assertIn("azure_key_phrases", enriched.columns)
        self.assertIn("language_model_sentiment", enriched.columns)
        self.assertIn("language_model_opinion_mining", enriched.columns)
        self.assertIn("language_model_key_phrases", enriched.columns)

        self.assertEqual(enriched.loc[0, "azure_sentiment"], "positive")
        self.assertEqual(enriched.loc[1, "language_model_sentiment"], "negative")
        self.assertEqual(enriched.loc[0, "azure_confidence_positive"], 0.99)
        self.assertEqual(enriched.loc[1, "azure_confidence_negative"], 0.89)
        self.assertEqual(enriched.loc[0, "azure_opinion_mining"], "support:positive")

    def test_enrich_feedback_dataframe_raises_on_size_mismatch(self):
        df = pd.DataFrame({"Feedback": ["Great support", "Needs improvement"]})
        azure_stub = StubAnalyzer([{"sentiment": "positive"}])
        phi_stub = StubAnalyzer([{"sentiment": "positive"}, {"sentiment": "negative"}])

        with self.assertRaises(ValueError):
            enrich_feedback_dataframe(df, "Feedback", azure_stub, phi_stub)

    def test_build_service_headers_prefers_api_key_when_present(self):
        with patch("feedback_analysis._credential.get_token") as get_token:
            headers = _build_service_headers("secret-key", "api-key")

        self.assertEqual(
            headers, {"api-key": "secret-key", "Content-Type": "application/json"}
        )
        get_token.assert_not_called()

    def test_build_service_headers_uses_bearer_token_when_api_key_missing(self):
        with patch("feedback_analysis._credential.get_token") as get_token:
            get_token.return_value.token = "token-value"
            headers = _build_service_headers("", "api-key")

        self.assertEqual(
            headers,
            {
                "Authorization": "Bearer token-value",
                "Content-Type": "application/json",
            },
        )
        get_token.assert_called_once()

    def test_build_service_headers_raises_clear_error_when_credentials_missing(self):
        with patch("feedback_analysis._credential.get_token", side_effect=RuntimeError("no cred")):
            with self.assertRaisesRegex(ValueError, "Azure credentials are unavailable"):
                _build_service_headers("", "api-key")

    def test_post_with_retry_retries_429_then_succeeds(self):
        first = self._build_response(429, {"error": {"message": "rate limit"}}, {"Retry-After": "0"})
        second = self._build_response(200, {"ok": True})

        with (
            patch("feedback_analysis.requests.post", side_effect=[first, second]) as post_mock,
            patch("feedback_analysis.time.sleep") as sleep_mock,
        ):
            response = _post_with_retry(
                "https://example.test",
                headers={"api-key": "test"},
                json={"sample": "payload"},
                timeout=30,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(post_mock.call_count, 2)
        sleep_mock.assert_called_once_with(0)

    def test_post_with_retry_raises_after_max_retries(self):
        throttled = self._build_response(429, {"error": {"message": "rate limit"}}, {"Retry-After": "0"})

        with (
            patch("feedback_analysis.requests.post", return_value=throttled) as post_mock,
            patch("feedback_analysis.time.sleep") as sleep_mock,
        ):
            with self.assertRaises(requests.HTTPError):
                _post_with_retry(
                    "https://example.test",
                    headers={"api-key": "test"},
                    json={"sample": "payload"},
                    timeout=30,
                )

        self.assertEqual(post_mock.call_count, MAX_RETRY_ATTEMPTS + 1)
        self.assertEqual(sleep_mock.call_count, MAX_RETRY_ATTEMPTS)

    def test_phi_analyzer_returns_empty_for_no_texts(self):
        analyzer = PhiAnalyzer(endpoint="https://example.test", deployment="phi", api_key="test-key")

        with patch("feedback_analysis._post_with_retry") as post_mock:
            self.assertEqual(analyzer.analyze([]), [])

        post_mock.assert_not_called()

    def test_phi_analyzer_analyzes_multiple_rows(self):
        analyzer = PhiAnalyzer(endpoint="https://example.test", deployment="phi", api_key="test-key")
        texts = ["Great support", "Needs improvement", "Fast response"]
        expected = {
            "Great support": {"sentiment": "positive", "opinion_mining": ["support"], "key_phrases": ["support"]},
            "Needs improvement": {"sentiment": "negative", "opinion_mining": ["improvement"], "key_phrases": ["improvement"]},
            "Fast response": {"sentiment": "positive", "opinion_mining": ["response"], "key_phrases": ["response"]},
        }

        with patch.object(
            PhiAnalyzer, "_analyze_one", side_effect=lambda text, headers=None: expected[text]
        ) as analyze_one_mock:
            result = analyzer.analyze(texts)

        self.assertEqual(result, [expected[text] for text in texts])
        self.assertEqual(result, [expected["Great support"], expected["Needs improvement"], expected["Fast response"]])
        self.assertEqual(analyze_one_mock.call_count, len(texts))


if __name__ == "__main__":
    unittest.main()
