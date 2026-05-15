# Feedback-analysis

A minimal Flask demo for analyzing feedback from an uploaded Excel file.

## What it does

- Accepts an Excel file upload (`.xlsx` / `.xls`) containing feedback rows
- Detects the feedback text column automatically
- Adds Azure AI Language outputs as new columns:
  - sentiment
  - opinion mining
  - key phrases
  - confidence scores
- Adds Phi model outputs as new columns:
  - sentiment
  - opinion mining
  - key phrases
- Renders the enriched table in the web UI
- Lets you download the enriched file as Excel

## Environment variables

Set these before running the app:

- `AZURE_LANGUAGE_ENDPOINT`
- `AZURE_LANGUAGE_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_KEY`
- `PHI_DEPLOYMENT_NAME`
- `AZURE_LANGUAGE_API_VERSION` (optional, defaults to `2023-04-01`)
- `AZURE_OPENAI_API_VERSION` (optional, defaults to `2024-06-01`)

Defaults were validated with this demo implementation and are recommended unless your Azure resources require newer API versions.

Optional for local/demo verification without Azure credentials:

- `DEMO_USE_MOCK_ANALYZERS=true`

## Run locally

```bash
python -m pip install -r requirements.txt
python app.py
```

Open `http://localhost:8000`.

Optional host/port overrides:

- `FLASK_HOST` (defaults to `127.0.0.1`)
- `FLASK_PORT` (defaults to `8000`)

## Run tests

```bash
python -m unittest discover -s tests
```
