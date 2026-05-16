import base64
import io
import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request, url_for

from feedback_analysis import (
    ENRICHMENT_COLUMNS,
    AzureLanguageAnalyzer,
    DemoHeuristicAnalyzer,
    PhiAnalyzer,
    detect_feedback_column,
    enrich_feedback_dataframe,
)

app = Flask(__name__)
logger = logging.getLogger(__name__)
_jobs_lock = threading.Lock()
_analysis_jobs: Dict[str, Dict[str, Any]] = {}


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
    return _resolve_language_model_deployment_name() or "Not configured"


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

            enriched_chunk = enrich_feedback_dataframe(
                input_chunk, feedback_column, azure_analyzer, phi_analyzer
            )

            with _jobs_lock:
                job = _analysis_jobs.get(job_id)
                if job is None:
                    return
                output_df = job["output_df"]
                output_df.loc[input_chunk.index, ENRICHMENT_COLUMNS] = enriched_chunk[
                    ENRICHMENT_COLUMNS
                ]
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
        "job_id": None,
        "status_url": None,
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
            context["table_html"] = output_df.to_html(index=False, classes="result-table")
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
        }
    response = jsonify(snapshot)
    response.headers["Cache-Control"] = "no-store"
    return response


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", os.getenv("FLASK_PORT", "8000"))),
    )
