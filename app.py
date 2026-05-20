import base64
import io
import logging
import os
import re
import threading
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request, url_for

import document_extraction as doc_extraction
from feedback_analysis import (
    AZURE_ENRICHMENT_COLUMNS,
    ENRICHMENT_COLUMNS,
    LANGUAGE_MODEL_ENRICHMENT_COLUMNS,
    AzureLanguageAnalyzer,
    DemoHeuristicAnalyzer,
    PhiAnalyzer,
    build_azure_enrichment_values,
    build_language_model_enrichment_values,
    detect_feedback_column,
)

app = Flask(__name__)
logger = logging.getLogger(__name__)
_jobs_lock = threading.Lock()
_analysis_jobs: Dict[str, Dict[str, Any]] = {}
POSITIVE_MENTION_THRESHOLD = 60
NEGATIVE_MENTION_THRESHOLD = 40
MAX_ACTION_THEMES = 3
TREND_POSITIVE_RATIO_THRESHOLD = 0.5
TREND_NEGATIVE_RATIO_LOW_THRESHOLD = 0.25
TREND_NEGATIVE_RATIO_HIGH_THRESHOLD = 0.4
MENTION_SCORE_PATTERN = re.compile(r"^(?P<item>.+)\s\((?P<score>\d{1,3})/100\)$")


def _read_positive_int_env(name: str, default: int) -> int:
    value = os.getenv(name, str(default)).strip()
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Invalid %s value '%s'; falling back to %d.", name, value, default)
        return default
    if parsed < 1:
        logger.warning("Invalid %s value '%s'; falling back to %d.", name, value, default)
        return default
    return parsed


_job_executor = ThreadPoolExecutor(
    max_workers=_read_positive_int_env("MAX_BACKGROUND_JOBS", 2)
)


def _resolve_language_model_deployment_name() -> str:
    return os.getenv("LANGUAGE_MODEL_DEPLOYMENT_NAME", "").strip() or os.getenv(
        "PHI_DEPLOYMENT_NAME", ""
    ).strip()


def _resolve_language_model_name() -> str:
    model_name = os.getenv("LANGUAGE_MODEL_NAME", "").strip() or os.getenv(
        "PHI_MODEL_NAME", ""
    ).strip()
    model_version = os.getenv("LANGUAGE_MODEL_VERSION", "").strip() or os.getenv(
        "PHI_MODEL_VERSION", ""
    ).strip()
    if model_name and model_version:
        return f"{model_name} ({model_version})"
    if model_name:
        return model_name
    return _resolve_language_model_deployment_name() or "Not configured"


def _split_semicolon_values(value: Any) -> list[str]:
    return [item.strip() for item in str(value).split(";") if item.strip()]


def _extract_mentions_with_scores(value: Any) -> list[tuple[str, int | None]]:
    mentions: list[tuple[str, int | None]] = []
    for chunk in _split_semicolon_values(value):
        match = MENTION_SCORE_PATTERN.match(chunk)
        if match:
            try:
                score = int(match.group("score"))
            except ValueError:
                score = None
            mentions.append((match.group("item").strip(), score))
        else:
            mentions.append((chunk, None))
    return mentions


def _build_overall_summary(output_df: pd.DataFrame) -> Dict[str, Any]:
    if output_df.empty:
        return {
            "trend": "No feedback has been analyzed yet.",
            "positive_trends": [],
            "negative_trends": [],
            "recommended_actions": [],
        }

    sentiment_counter: Counter[str] = Counter()
    trend_counter: Counter[str] = Counter()
    positive_items: Counter[str] = Counter()
    negative_items: Counter[str] = Counter()

    for _, row in output_df.iterrows():
        sentiment = str(row.get("language_model_sentiment", "")).strip().lower()
        if not sentiment:
            sentiment = str(row.get("azure_sentiment", "")).strip().lower()
        if sentiment in {"positive", "neutral", "negative", "mixed"}:
            sentiment_counter[sentiment] += 1

        for key_column in ("language_model_key_phrases", "azure_key_phrases"):
            for phrase in _split_semicolon_values(row.get(key_column, "")):
                trend_counter[phrase.lower()] += 1

        for mention, score in _extract_mentions_with_scores(
            row.get("language_model_opinion_mining", "")
        ):
            if not mention:
                continue
            mention_key = mention.lower()
            if score is None:
                continue
            if score >= POSITIVE_MENTION_THRESHOLD:
                positive_items[mention_key] += 1
            elif score <= NEGATIVE_MENTION_THRESHOLD:
                negative_items[mention_key] += 1

    total_sentiments = sum(sentiment_counter.values())
    if total_sentiments:
        positive_ratio = sentiment_counter.get("positive", 0) / total_sentiments
        negative_ratio = sentiment_counter.get("negative", 0) / total_sentiments
        if (
            positive_ratio >= TREND_POSITIVE_RATIO_THRESHOLD
            and negative_ratio < TREND_NEGATIVE_RATIO_LOW_THRESHOLD
        ):
            trend_text = "Overall sentiment trend is positive."
        elif negative_ratio >= TREND_NEGATIVE_RATIO_HIGH_THRESHOLD:
            trend_text = "Overall sentiment trend is negative."
        else:
            trend_text = "Overall sentiment trend is mixed."
    else:
        trend_text = "Not enough sentiment data to determine a clear trend yet."

    top_phrases = [name for name, _ in trend_counter.most_common(5)]
    if top_phrases:
        trend_text += f" Frequent themes: {', '.join(top_phrases)}."

    positive_trends = [name for name, _ in positive_items.most_common(5)]
    negative_trends = [name for name, _ in negative_items.most_common(5)]

    recommended_actions: list[str] = []
    if negative_trends:
        recommended_actions.append(
            f"Prioritize improvements on: {', '.join(negative_trends[:MAX_ACTION_THEMES])}."
        )
    if positive_trends:
        recommended_actions.append(
            f"Preserve and scale strengths in: {', '.join(positive_trends[:MAX_ACTION_THEMES])}."
        )
    if not recommended_actions:
        recommended_actions.append(
            "Collect more feedback detail to produce clearer action priorities."
        )

    return {
        "trend": trend_text,
        "positive_trends": positive_trends,
        "negative_trends": negative_trends,
        "recommended_actions": recommended_actions,
    }


def _format_request_exception(exc: requests.RequestException) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return ""

    message = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            error_obj = payload.get("error", {})
            if isinstance(error_obj, dict):
                message = str(error_obj.get("message", "")).strip()
            if not message:
                message = str(payload.get("message", "")).strip()
    except ValueError:
        message = ""

    if not message:
        message = (response.text or "").strip()

    if message:
        message = " ".join(message.split())
        message = message[:300]

    status = getattr(response, "status_code", "")
    return f"(HTTP {status}: {message})" if message else f"(HTTP {status})"


def _build_download_link(df: pd.DataFrame) -> str:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    payload = base64.b64encode(output.read()).decode("utf-8")
    return (
        f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{payload}"
    )


def _create_analyzers(use_mock: bool):
    if use_mock:
        return (
            DemoHeuristicAnalyzer(mode="azure"),
            DemoHeuristicAnalyzer(mode="phi"),
            "Demo heuristic analyzer",
        )

    return (
        AzureLanguageAnalyzer(
            endpoint=os.getenv("AZURE_LANGUAGE_ENDPOINT", ""),
            api_version=os.getenv("AZURE_LANGUAGE_API_VERSION", "2023-04-01"),
            api_key=os.getenv("AZURE_LANGUAGE_KEY", ""),
        ),
        PhiAnalyzer(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            deployment=_resolve_language_model_deployment_name(),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            api_key=os.getenv("AZURE_OPENAI_KEY", ""),
        ),
        _resolve_language_model_name(),
    )


def _initialize_output_dataframe(input_df: pd.DataFrame) -> pd.DataFrame:
    output_df = input_df.copy()
    for column in ENRICHMENT_COLUMNS:
        output_df[column] = ""
    return output_df


def _enqueue_analysis_job(input_df: pd.DataFrame, feedback_column: str, use_mock: bool) -> str:
    job_id = uuid.uuid4().hex
    output_df = _initialize_output_dataframe(input_df)
    with _jobs_lock:
        _analysis_jobs[job_id] = {
            "status": "queued",
            "processed_rows": 0,
            "total_rows": len(input_df),
            "feedback_column": feedback_column,
            "language_model_used": "Pending",
            "error": None,
            "output_df": output_df,
            "download_url": None,
        }

    _job_executor.submit(_process_analysis_job, job_id, feedback_column, use_mock)
    return job_id


def _process_analysis_job(job_id: str, feedback_column: str, use_mock: bool) -> None:
    try:
        azure_analyzer, phi_analyzer, language_model_used = _create_analyzers(use_mock)
        chunk_size = _read_positive_int_env("ANALYSIS_CHUNK_SIZE", 10)

        with _jobs_lock:
            job = _analysis_jobs.get(job_id)
            if job is None:
                return
            job["status"] = "running"
            job["language_model_used"] = language_model_used
            total_rows = int(job["total_rows"])

        for chunk_start in range(0, total_rows, chunk_size):
            with _jobs_lock:
                job = _analysis_jobs.get(job_id)
                if job is None:
                    return
                output_df: pd.DataFrame = job["output_df"]
                chunk_end = min(chunk_start + chunk_size, total_rows)
                input_chunk = output_df.iloc[chunk_start:chunk_end, :].copy()
            chunk_texts = input_chunk[feedback_column].fillna("").astype(str).tolist()

            azure_enrichment_values = build_azure_enrichment_values(
                azure_analyzer.analyze(chunk_texts), len(input_chunk)
            )

            with _jobs_lock:
                job = _analysis_jobs.get(job_id)
                if job is None:
                    return
                output_df = job["output_df"]
                row_slice = slice(chunk_start, chunk_end)
                for column_name in AZURE_ENRICHMENT_COLUMNS:
                    output_df.iloc[row_slice, output_df.columns.get_loc(column_name)] = (
                        azure_enrichment_values[column_name]
                    )

            language_model_enrichment_values = build_language_model_enrichment_values(
                phi_analyzer.analyze(chunk_texts), len(input_chunk)
            )

            with _jobs_lock:
                job = _analysis_jobs.get(job_id)
                if job is None:
                    return
                output_df = job["output_df"]
                row_slice = slice(chunk_start, chunk_end)
                for column_name in LANGUAGE_MODEL_ENRICHMENT_COLUMNS:
                    output_df.iloc[row_slice, output_df.columns.get_loc(column_name)] = (
                        language_model_enrichment_values[column_name]
                    )
                job["processed_rows"] = chunk_end

        with _jobs_lock:
            job = _analysis_jobs.get(job_id)
            if job is None:
                return
            job["status"] = "completed"
            job["download_url"] = _build_download_link(job["output_df"])
    except ValueError as exc:
        _set_job_failure(job_id, str(exc))
    except requests.RequestException as exc:
        logger.exception("External analysis service request failed in background job.")
        details = _format_request_exception(exc)
        _set_job_failure(
            job_id,
            (
                "Analysis request to Azure services failed. Check endpoint configuration and identity permissions. "
                f"{details}"
            ).strip(),
        )
    except Exception:
        logger.exception("Unexpected error while processing feedback upload job.")
        _set_job_failure(job_id, "Unable to process the uploaded file.")


def _set_job_failure(job_id: str, error_message: str) -> None:
    with _jobs_lock:
        job = _analysis_jobs.get(job_id)
        if job is None:
            return
        job["status"] = "failed"
        job["error"] = error_message


@app.route("/", methods=["GET", "POST"])
def index():
    language_model_name = _resolve_language_model_name()
    context = {
        "table_html": None,
        "error": None,
        "download_url": None,
        "feedback_column": None,
        "language_model_used": language_model_name,
        "overall_summary": None,
        "job_id": None,
        "status_url": None,
        "active_page": "feedback",
    }
    if request.method == "POST":
        uploaded_file = request.files.get("feedback_file")
        if not uploaded_file or uploaded_file.filename == "":
            context["error"] = "Please upload an Excel file (.xlsx)."
            return render_template("index.html", **context)

        try:
            input_df = pd.read_excel(uploaded_file)
            if input_df.empty:
                raise ValueError("Uploaded file is empty.")

            max_rows = _read_positive_int_env("MAX_UPLOAD_ROWS", 200)
            if len(input_df) > max_rows:
                raise ValueError(
                    f"File contains {len(input_df):,} rows. "
                    f"Please upload a file with at most {max_rows:,} {'row' if max_rows == 1 else 'rows'} at a time."
                )

            feedback_column = detect_feedback_column(input_df)
            use_mock = os.getenv("DEMO_USE_MOCK_ANALYZERS", "false").lower() == "true"
            output_df = _initialize_output_dataframe(input_df)
            job_id = _enqueue_analysis_job(input_df, feedback_column, use_mock)
            context["table_html"] = output_df.to_html(
                index=False, classes="govuk-table result-table"
            )
            context["job_id"] = job_id
            context["status_url"] = url_for("analysis_job_status", job_id=job_id)
            context["feedback_column"] = feedback_column
            context["language_model_used"] = "Pending"
        except ValueError as exc:
            context["error"] = str(exc)
        except requests.RequestException as exc:
            logger.exception("External analysis service request failed.")
            details = _format_request_exception(exc)
            context["error"] = (
                "Analysis request to Azure services failed. Check endpoint configuration and identity permissions. "
                f"{details}"
            ).strip()
        except Exception:
            logger.exception("Unexpected error while processing feedback upload.")
            context["error"] = "Unable to process the uploaded file."

    return render_template("index.html", **context)


@app.route("/document-extraction", methods=["GET", "POST"])
def document_extraction():
    context: Dict[str, Any] = {"error": None, "result": None, "active_page": "document"}
    if request.method == "POST":
        uploaded_file = request.files.get("document_file")
        if not uploaded_file or uploaded_file.filename == "":
            context["error"] = "Please upload a Word (.docx) or PDF (.pdf) file."
            return render_template("document_extraction.html", **context)

        filename = uploaded_file.filename
        if not (filename.lower().endswith(".docx") or filename.lower().endswith(".pdf")):
            context["error"] = "Unsupported file format. Please upload a .docx or .pdf file."
            return render_template("document_extraction.html", **context)

        try:
            file_bytes = uploaded_file.read()
            text = doc_extraction.extract_text(file_bytes, filename)
            if not text.strip():
                raise ValueError("No text could be extracted from the uploaded document.")

            use_mock = os.getenv("DEMO_USE_MOCK_ANALYZERS", "false").lower() == "true"
            if use_mock:
                formatter: Any = doc_extraction.MockDocumentFormatter()
            else:
                formatter = doc_extraction.DocumentFormatter(
                    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                    deployment=_resolve_language_model_deployment_name(),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
                    api_key=os.getenv("AZURE_OPENAI_KEY", ""),
                )
            context["result"] = formatter.format(text)
        except ValueError as exc:
            context["error"] = str(exc)
        except requests.RequestException as exc:
            logger.exception("External service request failed during document extraction.")
            details = _format_request_exception(exc)
            context["error"] = (
                "Document extraction request to Azure services failed. "
                "Check endpoint configuration and identity permissions. "
                f"{details}"
            ).strip()
        except Exception:
            logger.exception("Unexpected error during document extraction.")
            context["error"] = "Unable to process the uploaded document."

    return render_template("document_extraction.html", **context)


@app.route("/jobs/<job_id>/status", methods=["GET"])
def analysis_job_status(job_id: str):
    with _jobs_lock:
        job = _analysis_jobs.get(job_id)
        if job is None:
            return jsonify({"error": "Job not found."}), 404
        snapshot = {
            "status": job["status"],
            "processed_rows": int(job["processed_rows"]),
            "total_rows": int(job["total_rows"]),
            "feedback_column": job["feedback_column"],
            "language_model_used": job["language_model_used"],
            "error": job["error"],
            "download_url": job["download_url"],
            "table_columns": list(job["output_df"].columns),
            "table_rows": job["output_df"]
            .where(job["output_df"].notna(), "")
            .astype(str)
            .values.tolist(),
            "overall_summary": _build_overall_summary(job["output_df"]),
        }
    response = jsonify(snapshot)
    response.headers["Cache-Control"] = "no-store"
    return response


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", os.getenv("FLASK_PORT", "8000"))),
    )
