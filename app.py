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
    context = {"table_html": None, "error": None, "download_url": None, "feedback_column": None}
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
            else:
                azure_analyzer = AzureLanguageAnalyzer(
                    endpoint=os.getenv("AZURE_LANGUAGE_ENDPOINT", ""),
                    api_key=os.getenv("AZURE_LANGUAGE_KEY", ""),
                    api_version=os.getenv("AZURE_LANGUAGE_API_VERSION", "2023-04-01"),
                )
                phi_analyzer = PhiAnalyzer(
                    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                    api_key=os.getenv("AZURE_OPENAI_KEY", ""),
                    deployment=os.getenv("PHI_DEPLOYMENT_NAME", ""),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
                )

            output_df = enrich_feedback_dataframe(
                input_df, feedback_column, azure_analyzer, phi_analyzer
            )
            context["table_html"] = output_df.to_html(index=False, classes="result-table")
            context["download_url"] = _build_download_link(output_df)
            context["feedback_column"] = feedback_column
        except ValueError as exc:
            context["error"] = str(exc)
        except requests.RequestException:
            logger.exception("External analysis service request failed.")
            context["error"] = "Analysis request to Azure services failed. Check endpoint and key configuration."
        except Exception:
            logger.exception("Unexpected error while processing feedback upload.")
            context["error"] = "Unable to process the uploaded file."

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", os.getenv("FLASK_PORT", "8000"))),
    )
