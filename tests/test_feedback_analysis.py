import json
import unittest
from unittest.mock import patch

import pandas as pd
import requests

from feedback_analysis import (
    MAX_RETRY_ATTEMPTS,
    _post_with_retry,
    _build_service_headers,
    AzureLanguageAnalyzer,
    PhiAnalyzer,
    build_language_model_enrichment_values,
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

    def test_build_language_model_enrichment_values_formats_item_scores(self):
        values = build_language_model_enrichment_values(
            [
                {
                    "sentiment": "mixed",
                    "opinion_mining": [
                        {"item": "support", "positivity_score": 82},
                        {"target": "response time", "score": 0.2},
                        "pricing",
                    ],
                    "key_phrases": ["support", "response time"],
                }
            ],
            1,
        )

        self.assertEqual(
            values["language_model_opinion_mining"][0],
            "support (82/100); response time (60/100); pricing",
        )

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
        self.assertEqual(analyze_one_mock.call_count, len(texts))

    def test_azure_analyzer_batches_requests_for_large_input(self):
        analyzer = AzureLanguageAnalyzer(
            endpoint="https://example.test", api_version="2023-04-01", api_key="test-key"
        )
        texts = [f"Feedback {i}" for i in range(30)]

        def _sentiment_response(count):
            return self._build_response(200, {
                "results": {
                    "documents": [
                        {
                            "id": str(i + 1),
                            "sentiment": "positive",
                            "confidenceScores": {"positive": 1.0, "neutral": 0.0, "negative": 0.0},
                            "sentences": [],
                        }
                        for i in range(count)
                    ]
                }
            })

        def _keyphrase_response(count):
            return self._build_response(200, {
                "results": {
                    "documents": [
                        {"id": str(i + 1), "keyPhrases": []}
                        for i in range(count)
                    ]
                }
            })

        # 30 texts → 2 batches (25 + 5) → 4 POST calls total
        side_effects = [
            _sentiment_response(25),
            _keyphrase_response(25),
            _sentiment_response(5),
            _keyphrase_response(5),
        ]

        with patch("feedback_analysis._post_with_retry", side_effect=side_effects) as post_mock:
            results = analyzer.analyze(texts)

        self.assertEqual(len(results), 30)
        self.assertEqual(post_mock.call_count, 4)
        self.assertEqual(results[0]["sentiment"], "positive")
        self.assertEqual(results[29]["sentiment"], "positive")

    def test_azure_analyzer_returns_placeholders_for_failed_batch(self):
        analyzer = AzureLanguageAnalyzer(
            endpoint="https://example.test", api_version="2023-04-01", api_key="test-key"
        )
        texts = ["Feedback A", "Feedback B", "Feedback C"]

        with patch(
            "feedback_analysis._post_with_retry",
            side_effect=requests.HTTPError("service unavailable"),
        ):
            results = analyzer.analyze(texts)

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(result["sentiment"], "")
            self.assertEqual(result["opinion_mining"], [])
            self.assertIsNone(result["confidence_positive"])

    def test_phi_analyzer_returns_placeholder_for_failed_row(self):
        analyzer = PhiAnalyzer(
            endpoint="https://example.test", deployment="phi", api_key="test-key"
        )
        texts = ["Good service", "Bad row", "OK service"]

        good_response = self._build_response(200, {
            "choices": [{"message": {"content": json.dumps(
                {"sentiment": "positive", "opinion_mining": [], "key_phrases": []}
            )}}]
        })

        def side_effect(url, headers, json, timeout):
            if json["messages"][1]["content"] == "Bad row":
                raise requests.HTTPError("service error")
            return good_response

        with patch("feedback_analysis._post_with_retry", side_effect=side_effect):
            results = analyzer.analyze(texts)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["sentiment"], "positive")
        self.assertEqual(results[1]["sentiment"], "")
        self.assertEqual(results[1]["opinion_mining"], [])
        self.assertEqual(results[2]["sentiment"], "positive")


if __name__ == "__main__":
    unittest.main()
