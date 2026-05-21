import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests

_logger = logging.getLogger(__name__)

_BING_SEARCH_RESULTS = 30
_ARTICLE_WORD_LIMIT = 1200
_ARTICLE_FETCH_TIMEOUT = 5
_FALLBACK_MIN_WORDS = 100
_MAX_FETCH_WORKERS = 10

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
        match = re.search(r"\{.*\}", value, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def search_news_articles(
    topic: str,
    freshness: str = "Month",
    bing_endpoint: str = "",
    bing_key: str = "",
) -> List[Dict[str, Any]]:
    """Search Bing News for articles about *topic* and return structured metadata.

    Args:
        topic: User-supplied news topic string.
        freshness: Bing freshness filter – "Day", "Week", or "Month".
        bing_endpoint: Bing Search base URL (defaults to env var BING_SEARCH_ENDPOINT).
        bing_key: Bing subscription key (defaults to env var BING_SEARCH_KEY).

    Returns:
        List of dicts with keys: title, url, description, source_name, published_date.

    Raises:
        ValueError: If no Bing key is configured.
        requests.RequestException: On HTTP errors.
    """
    endpoint = (bing_endpoint or os.getenv("BING_SEARCH_ENDPOINT", "https://api.bing.microsoft.com")).rstrip("/")
    key = bing_key or os.getenv("BING_SEARCH_KEY", "")

    if not key:
        raise ValueError(
            "Bing Search API key is not configured. Set BING_SEARCH_KEY environment variable."
        )

    valid_freshness = {"Day", "Week", "Month"}
    if freshness not in valid_freshness:
        freshness = "Month"

    headers = {
        "Ocp-Apim-Subscription-Key": key,
    }
    params = {
        "q": topic,
        "mkt": "en-US",
        "count": _BING_SEARCH_RESULTS,
        "freshness": freshness,
        "sortBy": "Relevance",
        "textDecorations": False,
        "textFormat": "Raw",
    }

    response = requests.get(
        f"{endpoint}/v7.0/news/search",
        headers=headers,
        params=params,
        timeout=15,
    )
    response.raise_for_status()

    data = response.json()
    articles: List[Dict[str, Any]] = []
    for item in data.get("value", []):
        provider_list = item.get("provider", [])
        source_name = provider_list[0].get("name", "") if provider_list else ""
        articles.append(
            {
                "title": item.get("name", ""),
                "url": item.get("url", ""),
                "description": item.get("description", ""),
                "source_name": source_name,
                "published_date": item.get("datePublished", ""),
            }
        )
    return articles


def fetch_article_text(url: str) -> str:
    """Fetch and extract the main body text from a news article URL.

    Falls back to an empty string on any error (timeouts, paywalls, JS-heavy pages).
    """
    try:
        from bs4 import BeautifulSoup  # imported here to keep the module importable without bs4 in tests
    except ImportError:
        return ""

    try:
        resp = requests.get(
            url,
            timeout=_ARTICLE_FETCH_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; NewsSummariserBot/1.0)"},
        )
        resp.raise_for_status()
        html = resp.text
    except Exception:
        return ""

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        try:
            from bs4 import BeautifulSoup as BS  # noqa: F811
            soup = BS(html, "html.parser")
        except Exception:
            return ""

    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "iframe"]):
        tag.decompose()

    text = ""
    body = soup.find("article") or soup.find("main")
    if body:
        text = body.get_text(separator=" ", strip=True)
    if not text:
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)

    words = text.split()
    if len(words) > _ARTICLE_WORD_LIMIT:
        words = words[:_ARTICLE_WORD_LIMIT]
    return " ".join(words)


def _fetch_articles_parallel(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enrich each article dict with a 'body' key by fetching full text in parallel."""
    enriched = [dict(a) for a in articles]
    url_to_index: Dict[str, int] = {}
    for i, article in enumerate(enriched):
        url = article.get("url", "")
        if url:
            url_to_index[url] = i

    with ThreadPoolExecutor(max_workers=_MAX_FETCH_WORKERS) as executor:
        future_to_url = {
            executor.submit(fetch_article_text, url): url for url in url_to_index
        }
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            idx = url_to_index[url]
            try:
                body = future.result()
            except Exception:
                body = ""
            enriched[idx]["body"] = body

    for article in enriched:
        article.setdefault("body", "")
        words = article["body"].split()
        if len(words) < _FALLBACK_MIN_WORDS:
            article["body"] = article.get("description", "")

    return enriched


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
            "caveat": "No articles were returned by the Bing News Search API.",
        }

    enriched = _fetch_articles_parallel(articles)
    grouped = _group_articles_by_lean(enriched)
    summary = generate_balanced_summary(topic, grouped)
    return summary
