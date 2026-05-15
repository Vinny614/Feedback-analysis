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

## Provision Azure infrastructure with Terraform

Terraform files are available in `infra/terraform` and create:

- Resource Group
- Azure AI Language (`TextAnalytics`) account
- Azure OpenAI account
- Azure OpenAI deployment for Phi model

1. Authenticate to Azure (for example with `az login`).
2. Initialize and apply Terraform:

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform apply
```

3. Use Terraform outputs to set app environment variables:

- `azure_language_endpoint` -> `AZURE_LANGUAGE_ENDPOINT`
- `azure_language_key` -> `AZURE_LANGUAGE_KEY`
- `azure_openai_endpoint` -> `AZURE_OPENAI_ENDPOINT`
- `azure_openai_key` -> `AZURE_OPENAI_KEY`
- `phi_deployment_name` -> `PHI_DEPLOYMENT_NAME`

## Run locally

```bash
python -m pip install -r requirements.txt
python app.py
```

Open `http://localhost:8000`.

Optional host/port overrides:

- `FLASK_HOST` (defaults to `127.0.0.1`)
- `FLASK_PORT` (defaults to `8000`)

## Run on Azure Web App (Linux)

1. Create a Python App Service and configure these app settings:
   - `AZURE_LANGUAGE_ENDPOINT`
   - `AZURE_LANGUAGE_KEY`
   - `AZURE_OPENAI_ENDPOINT`
   - `AZURE_OPENAI_KEY`
   - `PHI_DEPLOYMENT_NAME`
   - `DEMO_USE_MOCK_ANALYZERS` (optional)

2. Deploy this repository to the Web App.

3. Set the Startup Command to:

```bash
gunicorn --bind=0.0.0.0:$PORT wsgi:application
```

The app now also supports `PORT` automatically when run directly.

## Run tests

```bash
python -m unittest discover -s tests
```
