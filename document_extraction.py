import io
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from feedback_analysis import _build_service_headers, _post_with_retry, _safe_parse_json

_logger = logging.getLogger(__name__)

MAX_DOCUMENT_CHARS = 50_000

_SYSTEM_PROMPT = (
    "You are a document analyst. Extract structured information from the provided document text. "
    "Return JSON with exactly these keys:\n"
    "- title: the document title (read from the document or generated from the content, as a short string)\n"
    "- summary: a concise summary of the document in exactly 1-2 sentences\n"
    "- key_events: an array of objects in chronological order, each with keys: "
    "event (brief description of the event), date (date string or null if not provided), "
    "time (time string or null if not provided). "
    "Include only major events. If no events are present, return an empty array."
)


def extract_text_from_docx(file_bytes: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def extract_text_from_pdf(file_bytes: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n".join(pages)


def extract_text(file_bytes: bytes, filename: str) -> str:
    filename_lower = filename.lower()
    if filename_lower.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    if filename_lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    raise ValueError("Unsupported file format. Please upload a .docx or .pdf file.")


def _normalise_key_events(raw: Any) -> List[Dict[str, Optional[str]]]:
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        event_text = str(item.get("event", "")).strip()
        if not event_text:
            continue
        result.append(
            {
                "event": event_text,
                "date": str(item["date"]).strip() if item.get("date") else None,
                "time": str(item["time"]).strip() if item.get("time") else None,
            }
        )
    return result


@dataclass
class DocumentFormatter:
    endpoint: str
    deployment: str
    api_version: str = "2025-01-01-preview"
    api_key: str = ""

    def format(self, text: str) -> Dict[str, Any]:
        if not self.endpoint or not self.deployment:
            raise ValueError(
                "Azure OpenAI configuration is missing. "
                "Set AZURE_OPENAI_ENDPOINT and a deployment name."
            )

        truncated = text[:MAX_DOCUMENT_CHARS]
        headers = _build_service_headers(api_key=self.api_key, api_key_header="api-key")
        response = _post_with_retry(
            f"{self.endpoint.rstrip('/')}/openai/deployments/{self.deployment}"
            f"/chat/completions?api-version={self.api_version}",
            headers=headers,
            json={
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": truncated},
                ],
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        choice = response.json().get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "{}")
        parsed = _safe_parse_json(content)
        return {
            "title": str(parsed.get("title", "")).strip() or "Untitled Document",
            "summary": str(parsed.get("summary", "")).strip() or "No summary available.",
            "key_events": _normalise_key_events(parsed.get("key_events", [])),
        }


@dataclass
class MockDocumentFormatter:
    def format(self, text: str) -> Dict[str, Any]:
        words = text.split()
        title = " ".join(words[:6]).rstrip(".,;:") if words else "Sample Document"
        return {
            "title": title,
            "summary": (
                "This is a demo summary of the uploaded document. "
                "The document has been processed by the mock analyzer."
            ),
            "key_events": [
                {"event": "Document created", "date": "2024-01-01", "time": None},
                {"event": "Key milestone reached", "date": "2024-06-15", "time": "10:00"},
            ],
        }
