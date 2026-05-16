import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List

import pandas as pd
import requests
from azure.identity import DefaultAzureCredential

_credential = DefaultAzureCredential()
_COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"
_logger = logging.getLogger(__name__)


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
MAX_KEY_PHRASES = 5
MAX_OPINION_TOKENS = 3
MIN_SENTIMENT_TOTAL = 1
MAX_RETRY_ATTEMPTS = 4
BASE_RETRY_SECONDS = 1
MAX_RETRY_SECONDS = 30
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
def _read_positive_int_env(name: str, default: int) -> int:
    value = os.getenv(name, str(default)).strip()
    try:
        parsed = int(value)
    except ValueError:
        _logger.warning("Invalid %s value '%s'; falling back to %d.", name, value, default)
        return default
    if parsed < 1:
        _logger.warning("Invalid %s value '%s'; falling back to %d.", name, value, default)
        return default
    return parsed


MAX_CONCURRENT_ANALYSIS_REQUESTS = _read_positive_int_env(
    "MAX_CONCURRENT_ANALYSIS_REQUESTS", 4
)
# Default is conservative to reduce throttling/timeout pressure on shared quotas;
# increase via MAX_CONCURRENT_ANALYSIS_REQUESTS when capacity allows.
_AZURE_MAX_BATCH_SIZE = 25
_DEFAULT_ANALYSIS_CHUNK_SIZE = 25

ENRICHMENT_COLUMNS = [
    "azure_sentiment",
    "azure_opinion_mining",
    "azure_key_phrases",
    "azure_confidence_positive",
    "azure_confidence_neutral",
    "azure_confidence_negative",
    "language_model_sentiment",
    "language_model_opinion_mining",
    "language_model_key_phrases",
]

_EMPTY_AZURE_ROW: Dict[str, Any] = {
    "sentiment": "",
    "opinion_mining": [],
    "key_phrases": [],
    "confidence_positive": None,
    "confidence_neutral": None,
    "confidence_negative": None,
}


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

            key_phrases = list(dict.fromkeys(words[:MAX_KEY_PHRASES]))
            if self.mode == "azure":
                total = max(len(positives) + len(negatives), MIN_SENTIMENT_TOTAL)
                results.append(
                    {
                        "sentiment": sentiment,
                        "opinion_mining": [
                            {"target": token, "sentiment": "positive"}
                            for token in positives[:MAX_OPINION_TOKENS]
                        ]
                        + [
                            {"target": token, "sentiment": "negative"}
                            for token in negatives[:MAX_OPINION_TOKENS]
                        ],
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
                        "opinion_mining": positives[:MAX_OPINION_TOKENS]
                        + negatives[:MAX_OPINION_TOKENS],
                        "key_phrases": key_phrases,
                    }
                )
        return results


@dataclass
class AzureLanguageAnalyzer:
    endpoint: str
    api_version: str = "2023-04-01"
    api_key: str = ""

    def analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not self.endpoint:
            raise ValueError(
                "Azure Language endpoint is missing. Set AZURE_LANGUAGE_ENDPOINT."
            )

        headers = _build_service_headers(
            api_key=self.api_key, api_key_header="Ocp-Apim-Subscription-Key"
        )

        all_results: List[Dict[str, Any]] = []
        for chunk_start in range(0, len(texts), _AZURE_MAX_BATCH_SIZE):
            chunk = texts[chunk_start : chunk_start + _AZURE_MAX_BATCH_SIZE]
            try:
                all_results.extend(self._analyze_chunk(chunk, headers))
            except Exception:
                _logger.warning(
                    "Azure Language batch failed for rows %d-%d; using empty placeholders.",
                    chunk_start,
                    chunk_start + len(chunk) - 1,
                    exc_info=True,
                )
                all_results.extend([dict(_EMPTY_AZURE_ROW) for _ in chunk])

        return all_results

    def _analyze_chunk(
        self, texts: List[str], headers: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        documents = [
            {"id": str(i + 1), "language": "en", "text": text}
            for i, text in enumerate(texts)
        ]

        sentiment_response = _post_with_retry(
            f"{self.endpoint.rstrip('/')}/language/:analyze-text?api-version={self.api_version}",
            headers=headers,
            json={
                "kind": "SentimentAnalysis",
                "analysisInput": {"documents": documents},
                "parameters": {"opinionMining": True},
            },
            timeout=30,
        )

        keyphrase_response = _post_with_retry(
            f"{self.endpoint.rstrip('/')}/language/:analyze-text?api-version={self.api_version}",
            headers=headers,
            json={
                "kind": "KeyPhraseExtraction",
                "analysisInput": {"documents": documents},
                "parameters": {},
            },
            timeout=30,
        )

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
    deployment: str
    api_version: str = "2025-01-01-preview"
    api_key: str = ""

    def analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not self.endpoint or not self.deployment:
            raise ValueError(
                "Phi model configuration is missing. Set AZURE_OPENAI_ENDPOINT and PHI_DEPLOYMENT_NAME."
            )

        if not texts:
            return []

        headers = _build_service_headers(api_key=self.api_key, api_key_header="api-key")
        max_workers = min(MAX_CONCURRENT_ANALYSIS_REQUESTS, len(texts))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(partial(self._analyze_one, headers=headers), texts))

    def _analyze_one(self, text: str, headers: Dict[str, str]) -> Dict[str, Any]:
        try:
            response = _post_with_retry(
                f"{self.endpoint.rstrip('/')}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}",
                headers=headers,
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
            response_payload = response.json()
            choice = response_payload.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "{}")
            parsed = _safe_parse_json(content)
            return {
                "sentiment": parsed.get("sentiment", ""),
                "opinion_mining": parsed.get("opinion_mining", []),
                "key_phrases": parsed.get("key_phrases", []),
            }
        except Exception:
            _logger.warning("Language model analysis failed for a row; using empty placeholder.", exc_info=True)
            return {"sentiment": "", "opinion_mining": [], "key_phrases": []}


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


def _build_service_headers(api_key: str, api_key_header: str) -> Dict[str, str]:
    if api_key:
        return {api_key_header: api_key, "Content-Type": "application/json"}

    try:
        token = _credential.get_token(_COGNITIVE_SERVICES_SCOPE).token
    except Exception as exc:
        raise ValueError(
            "Azure credentials are unavailable. Set the Azure service key environment variables or sign in with az login."
        ) from exc

    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _parse_retry_after_seconds(response: requests.Response) -> int | None:
    value = (response.headers or {}).get("Retry-After", "").strip()
    if not value:
        return None
    try:
        seconds = int(value)
    except ValueError:
        return None
    return max(seconds, 0)


def _post_with_retry(url: str, headers: Dict[str, str], json: Dict[str, Any], timeout: int) -> requests.Response:
    last_response: requests.Response | None = None
    for retry_index in range(MAX_RETRY_ATTEMPTS + 1):
        response = requests.post(url, headers=headers, json=json, timeout=timeout)
        last_response = response

        if response.status_code not in RETRYABLE_STATUS_CODES:
            response.raise_for_status()
            return response

        if retry_index < MAX_RETRY_ATTEMPTS:
            retry_after = _parse_retry_after_seconds(response)
            if retry_after is None:
                retry_after = min(BASE_RETRY_SECONDS * (2**retry_index), MAX_RETRY_SECONDS)
            time.sleep(retry_after)

    if last_response is None:
        raise RuntimeError("Retry loop exited unexpectedly.")
    last_response.raise_for_status()
    return last_response


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


def _format_azure_opinions(opinion_mining: Any) -> str:
    normalized = []
    for item in opinion_mining or []:
        if isinstance(item, dict):
            target = item.get("target", "").strip()
            sentiment = item.get("sentiment", "").strip()
            normalized.append(f"{target}:{sentiment}" if target or sentiment else "")
        else:
            normalized.append(str(item))
    return _format_list(normalized)


def enrich_feedback_dataframe(
    df: pd.DataFrame, feedback_column: str, azure_analyzer: Any, phi_analyzer: Any
) -> pd.DataFrame:
    enriched = df.copy()
    texts = enriched[feedback_column].fillna("").astype(str).tolist()

    enrichment_values = _build_enrichment_values(texts, len(enriched), azure_analyzer, phi_analyzer)

    for column_name, values in enrichment_values.items():
        enriched[column_name] = values

    return enriched


def enrich_feedback_dataframe_in_chunks(
    df: pd.DataFrame,
    feedback_column: str,
    azure_analyzer: Any,
    phi_analyzer: Any,
    chunk_size: int = _DEFAULT_ANALYSIS_CHUNK_SIZE,
) -> pd.DataFrame:
    enriched = df.copy()
    for column_name in ENRICHMENT_COLUMNS:
        enriched[column_name] = ""

    safe_chunk_size = max(1, int(chunk_size))
    total_rows = len(enriched)
    for chunk_start in range(0, total_rows, safe_chunk_size):
        chunk_end = min(chunk_start + safe_chunk_size, total_rows)
        chunk = enriched.iloc[chunk_start:chunk_end].copy()
        chunk_texts = chunk[feedback_column].fillna("").astype(str).tolist()
        enrichment_values = _build_enrichment_values(
            chunk_texts, len(chunk), azure_analyzer, phi_analyzer
        )
        for column_name, values in enrichment_values.items():
            enriched.loc[chunk.index, column_name] = values

    return enriched


def _build_enrichment_values(
    texts: List[str], expected_rows: int, azure_analyzer: Any, phi_analyzer: Any
) -> Dict[str, List[Any]]:
    azure_results = azure_analyzer.analyze(texts)
    language_model_results = phi_analyzer.analyze(texts)

    if len(azure_results) != expected_rows or len(language_model_results) != expected_rows:
        raise ValueError("Analyzer output size does not match the number of feedback rows.")

    return {
        "azure_sentiment": [result.get("sentiment", "") for result in azure_results],
        "azure_opinion_mining": [
            _format_azure_opinions(result.get("opinion_mining", [])) for result in azure_results
        ],
        "azure_key_phrases": [_format_list(result.get("key_phrases", [])) for result in azure_results],
        "azure_confidence_positive": [result.get("confidence_positive", "") for result in azure_results],
        "azure_confidence_neutral": [result.get("confidence_neutral", "") for result in azure_results],
        "azure_confidence_negative": [result.get("confidence_negative", "") for result in azure_results],
        "language_model_sentiment": [result.get("sentiment", "") for result in language_model_results],
        "language_model_opinion_mining": [
            _format_list(result.get("opinion_mining", [])) for result in language_model_results
        ],
        "language_model_key_phrases": [
            _format_list(result.get("key_phrases", [])) for result in language_model_results
        ],
    }
