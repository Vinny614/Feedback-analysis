import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

_logger = logging.getLogger(__name__)

_BING_SEARCH_RESULTS = 30
_KNOWN_SOURCE_LEANS: Dict[str, str] = {
    # Centre / international wire services
    "reuters": "centre",
    "ap": "centre",
    "associated press": "centre",
    "bbc": "centre",
    "bbc news": "centre",
    "euronews": "centre",
    "france 24": "centre",
    "al jazeera": "centre",
    "bloomberg": "centre",
    # Centre-left
    "the guardian": "centre-left",
    "guardian": "centre-left",
    "the independent": "centre-left",
    "independent": "centre-left",
    "the new york times": "centre-left",
    "new york times": "centre-left",
    "nyt": "centre-left",
    "the washington post": "centre-left",
    "washington post": "centre-left",
    "cnn": "centre-left",
    "nbc news": "centre-left",
    "msnbc": "centre-left",
    "the mirror": "centre-left",
    "mirror": "centre-left",
    "huffpost": "centre-left",
    "the huffington post": "centre-left",
    # Centre-right
    "financial times": "centre-right",
    "ft": "centre-right",
    "the economist": "centre-right",
    "economist": "centre-right",
    "the times": "centre-right",
    "times": "centre-right",
    "the telegraph": "centre-right",
    "telegraph": "centre-right",
    "wall street journal": "centre-right",
    "wsj": "centre-right",
    "fox business": "centre-right",
    "cnbc": "centre-right",
    "the spectator": "centre-right",
    "spectator": "centre-right",
    # Right
    "daily mail": "right",
    "mail online": "right",
    "the sun": "right",
    "new york post": "right",
    "fox news": "right",
    "breitbart": "right",
    "the express": "right",
    "daily express": "right",
    "gb news": "right",
    "talk tv": "right",
    "newsmax": "right",
    # Left
    "the new statesman": "left",
    "new statesman": "left",
    "jacobin": "left",
    "the nation": "left",
    "mother jones": "left",
    "vice": "left",
    "novara media": "left",
}

_VALID_LEANS = {"left", "centre-left", "centre", "centre-right", "right"}

_MAX_RETRY_ATTEMPTS = 3
_BASE_RETRY_SECONDS = 1
_MAX_RETRY_SECONDS = 16
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _post_json_with_retry(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: int = 30,
) -> requests.Response:
    last_response: Optional[requests.Response] = None
    for attempt in range(_MAX_RETRY_ATTEMPTS + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError):
            if attempt < _MAX_RETRY_ATTEMPTS:
                time.sleep(min(_BASE_RETRY_SECONDS * (2**attempt), _MAX_RETRY_SECONDS))
                continue
            raise
        last_response = response
        if response.status_code not in _RETRYABLE_STATUS_CODES:
            response.raise_for_status()
            return response
        if attempt < _MAX_RETRY_ATTEMPTS:
            retry_after_raw = (response.headers or {}).get("Retry-After", "").strip()
            try:
                retry_after = max(int(retry_after_raw), 0)
            except (ValueError, TypeError):
                retry_after = min(_BASE_RETRY_SECONDS * (2**attempt), _MAX_RETRY_SECONDS)
            time.sleep(retry_after)
    if last_response is None:
        raise RuntimeError("Retry loop exited unexpectedly.")
    last_response.raise_for_status()
    return last_response


def _safe_parse_json(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        for pattern in (r"\{.*\}", r"\[.*\]"):
            match = re.search(pattern, value, re.DOTALL)
            if not match:
                continue
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
        return {}


def _extract_source_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host.split(".")[0].replace("-", " ").strip()


def _normalise_grounded_articles(raw_articles: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_articles, dict):
        candidates = (
            raw_articles.get("articles")
            or raw_articles.get("results")
            or raw_articles.get("items")
            or []
        )
    elif isinstance(raw_articles, list):
        candidates = raw_articles
    else:
        candidates = []

    articles: List[Dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        source_name = item.get("source_name", "")
        if not source_name:
            source = item.get("source")
            if isinstance(source, dict):
                source_name = source.get("name", "")
            elif isinstance(source, str):
                source_name = source
        article = {
            "title": str(item.get("title", item.get("name", ""))).strip(),
            "url": str(item.get("url", item.get("link", ""))).strip(),
            "description": str(item.get("description", item.get("snippet", ""))).strip(),
            "source_name": str(source_name).strip(),
            "published_date": str(
                item.get("published_date", item.get("datePublished", ""))
            ).strip(),
            "body": str(item.get("body", item.get("content", ""))).strip(),
        }
        if not article["body"]:
            article["body"] = article["description"]
        if not article["source_name"] and article["url"]:
            article["source_name"] = _extract_source_name_from_url(article["url"])
        if article["title"] or article["url"] or article["description"]:
            articles.append(article)
    return articles[:_BING_SEARCH_RESULTS]


def search_news_articles(
    topic: str,
    freshness: str = "Month",
    openai_endpoint: str = "",
    openai_deployment: str = "",
    openai_api_version: str = "",
    openai_api_key: str = "",
    bing_connection_id: str = "",
) -> List[Dict[str, Any]]:
    """Search news with Azure OpenAI + Bing grounding and return structured metadata.

    Args:
        topic: User-supplied news topic string.
        freshness: Search freshness hint – "Day", "Week", or "Month".
        openai_endpoint: Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT).
        openai_deployment: Azure OpenAI deployment name.
        openai_api_version: Azure OpenAI API version.
        openai_api_key: Azure OpenAI key (optional when managed identity is used).
        bing_connection_id: Azure Bing grounding connection resource ID.

    Returns:
        List of dicts with keys: title, url, description, source_name, published_date.

    Raises:
        ValueError: If required OpenAI/Bing grounding configuration is missing.
        requests.RequestException: On HTTP errors.
    """
    endpoint = (openai_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")).rstrip("/")
    deployment = openai_deployment or os.getenv("LANGUAGE_MODEL_DEPLOYMENT_NAME", "") or os.getenv(
        "PHI_DEPLOYMENT_NAME", ""
    )
    api_version = openai_api_version or os.getenv(
        "AZURE_OPENAI_API_VERSION", "2025-01-01-preview"
    )
    api_key = openai_api_key or os.getenv("AZURE_OPENAI_KEY", "")
    connection_id = bing_connection_id or os.getenv("BING_CONNECTION_ID", "")

    if not endpoint or not deployment:
        raise ValueError(
            "Azure OpenAI endpoint or deployment is missing. Set AZURE_OPENAI_ENDPOINT and PHI_DEPLOYMENT_NAME (or LANGUAGE_MODEL_DEPLOYMENT_NAME)."
        )
    if not connection_id:
        raise ValueError(
            "Bing grounding connection is not configured. Set BING_CONNECTION_ID environment variable."
        )

    valid_freshness = {"Day", "Week", "Month"}
    if freshness not in valid_freshness:
        freshness = "Month"

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["api-key"] = api_key
    else:
        try:
            from azure.identity import DefaultAzureCredential

            credential = DefaultAzureCredential()
            token = credential.get_token(
                "https://cognitiveservices.azure.com/.default"
            ).token
            headers["Authorization"] = f"Bearer {token}"
        except Exception as exc:
            raise ValueError(
                "Azure credentials are unavailable. Set AZURE_OPENAI_KEY or configure managed identity."
            ) from exc

    user_prompt = f"""Find recent news coverage for topic: {topic}
Freshness: {freshness}

Return only valid JSON with this shape:
{{
  "articles": [
    {{
      "title": "...",
      "url": "...",
      "description": "...",
      "source_name": "...",
      "published_date": "...",
      "body": "..."
    }}
  ]
}}

Use at most {_BING_SEARCH_RESULTS} articles.
"""

    response = _post_json_with_retry(
        f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}",
        headers=headers,
        payload={
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a news research assistant. Return JSON only.",
                },
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "data_sources": [
                {
                    "type": "bing_grounding",
                    "parameters": {"connection_id": connection_id},
                }
            ],
        },
        timeout=45,
    )
    response_data = response.json()
    message = response_data.get("choices", [{}])[0].get("message", {})
    parsed = _safe_parse_json(message.get("content", "{}"))
    articles = _normalise_grounded_articles(parsed)

    if articles:
        return articles

    citations = message.get("context", {}).get("citations", [])
    citation_articles = [
        {
            "title": str(item.get("title", "")).strip(),
            "url": str(item.get("url", "")).strip(),
            "description": str(item.get("content", "")).strip(),
            "source_name": str(item.get("provider", "")).strip(),
            "published_date": "",
            "body": str(item.get("content", "")).strip(),
        }
        for item in citations
        if isinstance(item, dict)
    ]
    return _normalise_grounded_articles(citation_articles)


def classify_source_lean(
    source_name: str,
    openai_endpoint: str = "",
    openai_deployment: str = "",
    openai_api_version: str = "",
    openai_api_key: str = "",
) -> str:
    """Classify the political/editorial lean of a news source.

    First checks the curated dictionary; falls back to Azure OpenAI for unknown sources.
    Returns one of: "left", "centre-left", "centre", "centre-right", "right", "unknown".
    """
    normalised = source_name.strip().lower()
    if normalised in _KNOWN_SOURCE_LEANS:
        return _KNOWN_SOURCE_LEANS[normalised]

    endpoint = openai_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")
    deployment = openai_deployment or os.getenv("LANGUAGE_MODEL_DEPLOYMENT_NAME", "") or os.getenv("PHI_DEPLOYMENT_NAME", "")
    api_version = openai_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    api_key = openai_api_key or os.getenv("AZURE_OPENAI_KEY", "")

    if not endpoint or not deployment:
        return "unknown"

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["api-key"] = api_key
    else:
        try:
            from azure.identity import DefaultAzureCredential
            credential = DefaultAzureCredential()
            token = credential.get_token("https://cognitiveservices.azure.com/.default").token
            headers["Authorization"] = f"Bearer {token}"
        except Exception:
            return "unknown"

    prompt = (
        f"Classify the political/editorial lean of the news source '{source_name}'. "
        "Reply with exactly one of: left, centre-left, centre, centre-right, right, unknown. "
        "No explanation."
    )

    try:
        response = _post_json_with_retry(
            f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions?api-version={api_version}",
            headers=headers,
            payload={
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": "You are a media bias classifier. Reply with a single label only."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 10,
            },
            timeout=15,
        )
        content = (
            response.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "unknown")
            .strip()
            .lower()
        )
        if content in _VALID_LEANS:
            return content
    except Exception:
        _logger.warning("Failed to classify source lean via OpenAI for '%s'.", source_name, exc_info=True)

    return "unknown"


def _group_articles_by_lean(
    articles: List[Dict[str, Any]],
    openai_endpoint: str = "",
    openai_deployment: str = "",
    openai_api_version: str = "",
    openai_api_key: str = "",
) -> Dict[str, List[Dict[str, Any]]]:
    """Return articles grouped by their editorial lean label."""
    groups: Dict[str, List[Dict[str, Any]]] = {
        "left": [],
        "centre-left": [],
        "centre": [],
        "centre-right": [],
        "right": [],
        "unknown": [],
    }
    for article in articles:
        lean = classify_source_lean(
            article.get("source_name", ""),
            openai_endpoint=openai_endpoint,
            openai_deployment=openai_deployment,
            openai_api_version=openai_api_version,
            openai_api_key=openai_api_key,
        )
        article["lean"] = lean
        groups[lean].append(article)
    return groups


def generate_balanced_summary(
    topic: str,
    grouped_articles: Dict[str, List[Dict[str, Any]]],
    openai_endpoint: str = "",
    openai_deployment: str = "",
    openai_api_version: str = "",
    openai_api_key: str = "",
) -> Dict[str, Any]:
    """Generate a balanced, multi-perspective news summary using Azure OpenAI.

    Returns a dict with keys:
        overview, left_arguments, right_arguments, consensus, sources, caveat.
    """
    endpoint = openai_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")
    deployment = openai_deployment or os.getenv("LANGUAGE_MODEL_DEPLOYMENT_NAME", "") or os.getenv("PHI_DEPLOYMENT_NAME", "")
    api_version = openai_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    api_key = openai_api_key or os.getenv("AZURE_OPENAI_KEY", "")

    if not endpoint or not deployment:
        raise ValueError(
            "Azure OpenAI endpoint or deployment is not configured. "
            "Set AZURE_OPENAI_ENDPOINT and PHI_DEPLOYMENT_NAME (or LANGUAGE_MODEL_DEPLOYMENT_NAME)."
        )

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["api-key"] = api_key
    else:
        try:
            from azure.identity import DefaultAzureCredential
            credential = DefaultAzureCredential()
            token = credential.get_token("https://cognitiveservices.azure.com/.default").token
            headers["Authorization"] = f"Bearer {token}"
        except Exception as exc:
            raise ValueError(
                "Azure credentials are unavailable. Set AZURE_OPENAI_KEY or configure managed identity."
            ) from exc

    left_buckets = ["left", "centre-left"]
    right_buckets = ["centre-right", "right"]
    centre_buckets = ["centre"]

    def _format_articles(buckets: List[str], max_articles: int = 5) -> str:
        parts: List[str] = []
        count = 0
        for bucket in buckets:
            for article in grouped_articles.get(bucket, []):
                if count >= max_articles:
                    break
                body = (article.get("body") or article.get("description") or "").strip()
                if body:
                    parts.append(
                        f"[{article.get('source_name', 'Unknown')}] {article.get('title', '')}: {body[:800]}"
                    )
                    count += 1
        return "\n\n".join(parts) if parts else "No articles available."

    left_text = _format_articles(left_buckets)
    right_text = _format_articles(right_buckets)
    centre_text = _format_articles(centre_buckets)

    total_left = sum(len(grouped_articles.get(b, [])) for b in left_buckets)
    total_right = sum(len(grouped_articles.get(b, [])) for b in right_buckets)
    perspective_count = (1 if total_left > 0 else 0) + (1 if total_right > 0 else 0)
    caveat = ""
    if perspective_count < 2:
        caveat = (
            "Note: Fewer than two distinct editorial perspectives were found in the retrieved articles. "
            "The summary may not fully represent all viewpoints."
        )

    system_prompt = (
        "You are a balanced news analyst. You synthesise news articles from multiple editorial "
        "perspectives into a structured, impartial summary. You do not editorialize or take sides."
    )

    user_prompt = f"""Topic: {topic}

CENTRE / WIRE SERVICE ARTICLES:
{centre_text}

LEFT-LEANING ARTICLES:
{left_text}

RIGHT-LEANING ARTICLES:
{right_text}

Produce a balanced summary as a JSON object with exactly these keys:
- "overview": A 3-5 sentence neutral factual overview of the topic.
- "left_arguments": An array of 3-5 strings, each a key argument or perspective found in left-leaning sources. If none, return an empty array.
- "right_arguments": An array of 3-5 strings, each a key argument or perspective found in right-leaning sources. If none, return an empty array.
- "consensus": An array of 2-4 strings representing points of agreement across perspectives. If none, return an empty array.

Return only valid JSON. No additional text outside the JSON object."""

    response = _post_json_with_retry(
        f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions?api-version={api_version}",
        headers=headers,
        payload={
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        },
        timeout=60,
    )

    content = (
        response.json()
        .get("choices", [{}])[0]
        .get("message", {})
        .get("content", "{}")
    )
    parsed = _safe_parse_json(content)

    sources: List[Dict[str, str]] = []
    seen_urls: set = set()
    for lean_key in ("left", "centre-left", "centre", "centre-right", "right", "unknown"):
        for article in grouped_articles.get(lean_key, []):
            url = article.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append(
                    {
                        "title": article.get("title", ""),
                        "url": url,
                        "source_name": article.get("source_name", ""),
                        "lean": article.get("lean", lean_key),
                    }
                )

    return {
        "overview": parsed.get("overview", ""),
        "left_arguments": parsed.get("left_arguments", []),
        "right_arguments": parsed.get("right_arguments", []),
        "consensus": parsed.get("consensus", []),
        "sources": sources,
        "caveat": caveat,
    }


def run_news_summary(
    topic: str,
    freshness: str = "Month",
) -> Dict[str, Any]:
    """Orchestrate the full news-summary pipeline and return structured results.

    This is the main entry point called from app.py background jobs.
    """
    articles = search_news_articles(topic, freshness=freshness)
    if not articles:
        return {
            "overview": f"No news articles were found for the topic: {topic}.",
            "left_arguments": [],
            "right_arguments": [],
            "consensus": [],
            "sources": [],
            "caveat": "No articles were returned by Grounding with Bing Search.",
        }

    grouped = _group_articles_by_lean(articles)
    summary = generate_balanced_summary(topic, grouped)
    return summary
