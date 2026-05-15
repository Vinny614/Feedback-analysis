import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
import requests


PREFERRED_FEEDBACK_COLUMNS = {
    "feedback",
    "comment",
    "comments",
    "review",
    "text",
    "message",
    "feedback_text",
}

POSITIVE_HINTS = {"good", "great", "helpful", "excellent", "love", "fast", "happy"}
NEGATIVE_HINTS = {"bad", "poor", "slow", "issue", "problem", "hate", "delay"}


@dataclass
class DemoHeuristicAnalyzer:
    mode: str

    def analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        results = []
        for text in texts:
            words = [token.strip(".,!?;:").lower() for token in text.split() if token.strip()]
            positives = [word for word in words if word in POSITIVE_HINTS]
            negatives = [word for word in words if word in NEGATIVE_HINTS]

            if len(positives) > len(negatives):
                sentiment = "positive"
            elif len(negatives) > len(positives):
                sentiment = "negative"
            elif positives or negatives:
                sentiment = "mixed"
            else:
                sentiment = "neutral"

            key_phrases = list(dict.fromkeys(words[:5]))
            if self.mode == "azure":
                total = max(len(positives) + len(negatives), 1)
                results.append(
                    {
                        "sentiment": sentiment,
                        "opinion_mining": [
                            {"target": token, "sentiment": "positive"} for token in positives[:3]
                        ]
                        + [{"target": token, "sentiment": "negative"} for token in negatives[:3]],
                        "key_phrases": key_phrases,
                        "confidence_positive": round(len(positives) / total, 2),
                        "confidence_neutral": 0.0 if positives or negatives else 1.0,
                        "confidence_negative": round(len(negatives) / total, 2),
                    }
                )
            else:
                results.append(
                    {
                        "sentiment": sentiment,
                        "opinion_mining": positives[:3] + negatives[:3],
                        "key_phrases": key_phrases,
                    }
                )
        return results


@dataclass
class AzureLanguageAnalyzer:
    endpoint: str
    api_key: str
    api_version: str = "2023-04-01"

    def analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not self.endpoint or not self.api_key:
            raise ValueError(
                "Azure Language credentials are missing. Set AZURE_LANGUAGE_ENDPOINT and AZURE_LANGUAGE_KEY."
            )

        documents = [{"id": str(i + 1), "language": "en", "text": text} for i, text in enumerate(texts)]

        sentiment_response = requests.post(
            f"{self.endpoint.rstrip('/')}/language/:analyze-text?api-version={self.api_version}",
            headers={"Ocp-Apim-Subscription-Key": self.api_key, "Content-Type": "application/json"},
            json={
                "kind": "SentimentAnalysis",
                "analysisInput": {"documents": documents},
                "parameters": {"opinionMining": True},
            },
            timeout=30,
        )
        sentiment_response.raise_for_status()

        keyphrase_response = requests.post(
            f"{self.endpoint.rstrip('/')}/language/:analyze-text?api-version={self.api_version}",
            headers={"Ocp-Apim-Subscription-Key": self.api_key, "Content-Type": "application/json"},
            json={
                "kind": "KeyPhraseExtraction",
                "analysisInput": {"documents": documents},
                "parameters": {},
            },
            timeout=30,
        )
        keyphrase_response.raise_for_status()

        sentiments = {
            int(item["id"]) - 1: item
            for item in sentiment_response.json().get("results", {}).get("documents", [])
        }
        key_phrases = {
            int(item["id"]) - 1: item.get("keyPhrases", [])
            for item in keyphrase_response.json().get("results", {}).get("documents", [])
        }

        results: List[Dict[str, Any]] = []
        for idx in range(len(texts)):
            sentiment_doc = sentiments.get(idx, {})
            confidence = sentiment_doc.get("confidenceScores", {})
            opinions = []
            for sentence in sentiment_doc.get("sentences", []):
                for mined_target in sentence.get("targets", []):
                    opinions.append(
                        {
                            "target": mined_target.get("text", ""),
                            "sentiment": mined_target.get("sentiment", ""),
                        }
                    )

            results.append(
                {
                    "sentiment": sentiment_doc.get("sentiment", ""),
                    "opinion_mining": opinions,
                    "key_phrases": key_phrases.get(idx, []),
                    "confidence_positive": confidence.get("positive"),
                    "confidence_neutral": confidence.get("neutral"),
                    "confidence_negative": confidence.get("negative"),
                }
            )
        return results


@dataclass
class PhiAnalyzer:
    endpoint: str
    api_key: str
    deployment: str
    api_version: str = "2024-06-01"

    def analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not self.endpoint or not self.api_key or not self.deployment:
            raise ValueError(
                "Phi model credentials are missing. Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, and PHI_DEPLOYMENT_NAME."
            )

        results = []
        for text in texts:
            response = requests.post(
                f"{self.endpoint.rstrip('/')}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}",
                headers={"api-key": self.api_key, "Content-Type": "application/json"},
                json={
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a feedback analyst. Return JSON with keys sentiment (positive|neutral|negative|mixed), "
                                "opinion_mining (array of short opinions), and key_phrases (array)."
                            ),
                        },
                        {"role": "user", "content": text},
                    ],
                    "response_format": {"type": "json_object"},
                },
                timeout=30,
            )
            response.raise_for_status()
            content = (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "{}")
            )
            parsed = _safe_parse_json(content)
            results.append(
                {
                    "sentiment": parsed.get("sentiment", ""),
                    "opinion_mining": parsed.get("opinion_mining", []),
                    "key_phrases": parsed.get("key_phrases", []),
                }
            )
        return results


def _safe_parse_json(value: str) -> Dict[str, Any]:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def detect_feedback_column(df: pd.DataFrame) -> str:
    lower_map = {str(column).strip().lower(): column for column in df.columns}
    for preferred in PREFERRED_FEEDBACK_COLUMNS:
        if preferred in lower_map:
            return lower_map[preferred]

    text_columns = [
        column
        for column in df.columns
        if pd.api.types.is_string_dtype(df[column]) or df[column].dtype == object
    ]
    if text_columns:
        return text_columns[0]

    return df.columns[0]


def _format_list(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value if str(item).strip())
    if value is None:
        return ""
    return str(value)


def enrich_feedback_dataframe(
    df: pd.DataFrame, feedback_column: str, azure_analyzer: Any, phi_analyzer: Any
) -> pd.DataFrame:
    enriched = df.copy()
    texts = enriched[feedback_column].fillna("").astype(str).tolist()

    azure_results = azure_analyzer.analyze(texts)
    phi_results = phi_analyzer.analyze(texts)

    if len(azure_results) != len(enriched) or len(phi_results) != len(enriched):
        raise ValueError("Analyzer output size does not match the number of feedback rows.")

    enriched["azure_sentiment"] = [result.get("sentiment", "") for result in azure_results]
    enriched["azure_opinion_mining"] = [
        _format_list(
            [
                f"{item.get('target', '').strip()}:{item.get('sentiment', '').strip()}" if isinstance(item, dict) else item
                for item in result.get("opinion_mining", [])
            ]
        )
        for result in azure_results
    ]
    enriched["azure_key_phrases"] = [_format_list(result.get("key_phrases", [])) for result in azure_results]
    enriched["azure_confidence_positive"] = [result.get("confidence_positive", "") for result in azure_results]
    enriched["azure_confidence_neutral"] = [result.get("confidence_neutral", "") for result in azure_results]
    enriched["azure_confidence_negative"] = [result.get("confidence_negative", "") for result in azure_results]

    enriched["phi_sentiment"] = [result.get("sentiment", "") for result in phi_results]
    enriched["phi_opinion_mining"] = [_format_list(result.get("opinion_mining", [])) for result in phi_results]
    enriched["phi_key_phrases"] = [_format_list(result.get("key_phrases", [])) for result in phi_results]

    return enriched
