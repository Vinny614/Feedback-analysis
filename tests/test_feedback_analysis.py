import unittest

import pandas as pd

from feedback_analysis import detect_feedback_column, enrich_feedback_dataframe


class _StubAnalyzer:
    def __init__(self, rows):
        self.rows = rows

    def analyze(self, texts):
        return self.rows


class FeedbackAnalysisTests(unittest.TestCase):
    def test_detect_feedback_column_prefers_named_column(self):
        df = pd.DataFrame({"ID": [1], "Feedback": ["Great service"]})
        self.assertEqual(detect_feedback_column(df), "Feedback")

    def test_enrich_feedback_dataframe_adds_expected_columns(self):
        df = pd.DataFrame({"Feedback": ["Great support", "Needs improvement"]})

        azure_stub = _StubAnalyzer(
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
        phi_stub = _StubAnalyzer(
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
        self.assertIn("phi_sentiment", enriched.columns)
        self.assertIn("phi_opinion_mining", enriched.columns)
        self.assertIn("phi_key_phrases", enriched.columns)

        self.assertEqual(enriched.loc[0, "azure_sentiment"], "positive")
        self.assertEqual(enriched.loc[1, "phi_sentiment"], "negative")
        self.assertEqual(enriched.loc[0, "azure_confidence_positive"], 0.99)
        self.assertEqual(enriched.loc[1, "azure_confidence_negative"], 0.89)
        self.assertEqual(enriched.loc[0, "azure_opinion_mining"], "support:positive")


if __name__ == "__main__":
    unittest.main()
