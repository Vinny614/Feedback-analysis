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

## Authentication — RBAC, no keys

All Azure service calls use **Managed Identity / Entra ID (AAD) bearer tokens** via
[`DefaultAzureCredential`](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential).
No API keys are stored or required.

- **On Azure App Service**: the system-assigned managed identity authenticates automatically.
- **Locally**: `az login` is used; `DefaultAzureCredential` picks up your CLI session.

## Environment variables

Set these before running the app (no keys needed):

- `AZURE_LANGUAGE_ENDPOINT`
- `AZURE_OPENAI_ENDPOINT`
- `PHI_DEPLOYMENT_NAME`
- `AZURE_LANGUAGE_API_VERSION` (optional, defaults to `2023-04-01`)
- `AZURE_OPENAI_API_VERSION` (optional, defaults to `2024-06-01`)

Optional for local/demo verification without Azure credentials:

- `DEMO_USE_MOCK_ANALYZERS=true`

## Demo: one-command build and teardown with Terraform

Terraform provisions **everything** — AI services, RBAC assignments, and the App Service.
`terraform destroy` tears it all down cleanly.

### What Terraform creates

- Resource Group
- Azure AI Language (`TextAnalytics`) account — key auth disabled
- Azure OpenAI account + Phi model deployment — key auth disabled
- **RBAC role assignments** for the identity running Terraform (for local development)
- App Service Plan + Linux Web App with system-assigned managed identity
- **RBAC role assignments** for the App Service managed identity

### Deploy

1. Authenticate to Azure:

```bash
az login
```

2. Initialize and apply Terraform:

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform apply
```

3. Deploy the application code to the App Service:

```bash
cd ../..
az webapp up \
  --name <app_name_from_terraform_output> \
  --resource-group feedback-analysis-rg \
  --runtime "PYTHON:3.11"
```

4. The app URL is shown in the `app_url` Terraform output.

### Tear down

```bash
cd infra/terraform
terraform destroy
```

This removes all Azure resources created for the demo.

## Run locally

```bash
az login
python -m pip install -r requirements.txt
python app.py
```

Open `http://localhost:8000`.

`DefaultAzureCredential` uses your `az login` session to authenticate to Azure AI services.
Your account must have the **Cognitive Services User** role on the Language resource and the
**Cognitive Services OpenAI User** role on the OpenAI resource — Terraform assigns these
automatically for the identity that runs `terraform apply`.

Optional host/port overrides:

- `FLASK_HOST` (defaults to `127.0.0.1`)
- `FLASK_PORT` (defaults to `8000`)

## Run on Azure Web App (Linux)

The App Service is provisioned by Terraform with the correct managed identity and RBAC
assignments. Deploy the code with:

```bash
az webapp up \
  --name <app_name> \
  --resource-group feedback-analysis-rg \
  --runtime "PYTHON:3.11"
```

The startup command is already configured in Terraform:

```
gunicorn --bind=0.0.0.0:$PORT wsgi:application
```

## Run tests

```bash
python -m unittest discover -s tests
```
