import base64
import io
import logging
import os

import pandas as pd
import requests
from flask import Flask, render_template, request

from feedback_analysis import (
    AzureLanguageAnalyzer,
    DemoHeuristicAnalyzer,
    PhiAnalyzer,
    detect_feedback_column,
    enrich_feedback_dataframe,
)

app = Flask(__name__)
logger = logging.getLogger(__name__)


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


@app.route("/", methods=["GET", "POST"])
def index():
    default_model_name = os.getenv("PHI_DEPLOYMENT_NAME", "").strip() or "Not configured"
    context = {
        "table_html": None,
        "error": None,
        "download_url": None,
        "feedback_column": None,
        "language_model_used": default_model_name,
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

            feedback_column = detect_feedback_column(input_df)
            use_mock = os.getenv("DEMO_USE_MOCK_ANALYZERS", "false").lower() == "true"
            if use_mock:
                azure_analyzer = DemoHeuristicAnalyzer(mode="azure")
                phi_analyzer = DemoHeuristicAnalyzer(mode="phi")
                context["language_model_used"] = "Demo heuristic analyzer"
            else:
                azure_analyzer = AzureLanguageAnalyzer(
                    endpoint=os.getenv("AZURE_LANGUAGE_ENDPOINT", ""),
                    api_version=os.getenv("AZURE_LANGUAGE_API_VERSION", "2023-04-01"),
                    api_key=os.getenv("AZURE_LANGUAGE_KEY", ""),
                )
                phi_analyzer = PhiAnalyzer(
                    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                    deployment=os.getenv("PHI_DEPLOYMENT_NAME", ""),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
                    api_key=os.getenv("AZURE_OPENAI_KEY", ""),
                )
                context["language_model_used"] = (
                    os.getenv("PHI_DEPLOYMENT_NAME", "").strip() or "Not configured"
                )

            output_df = enrich_feedback_dataframe(
                input_df, feedback_column, azure_analyzer, phi_analyzer
            )
            context["table_html"] = output_df.to_html(index=False, classes="result-table")
            context["download_url"] = _build_download_link(output_df)
            context["feedback_column"] = feedback_column
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


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", os.getenv("FLASK_PORT", "8000"))),
    )
