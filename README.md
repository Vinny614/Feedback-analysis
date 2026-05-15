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
- Adds Azure OpenAI chat-model outputs as new columns:
  - sentiment
  - opinion mining
  - key phrases
- Renders the enriched table in the web UI
- Lets you download the enriched file as Excel

## Authentication — Entra ID locally, app settings in Azure

All Azure service calls use **Managed Identity / Entra ID (AAD) bearer tokens** via
[`DefaultAzureCredential`](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential)
when no service keys are configured.

- **On Azure App Service**: Terraform injects the AI service endpoints, deployment name,
  and access keys into application settings.
- **Locally**: `az login` is used; `DefaultAzureCredential` picks up your CLI session.

## Environment variables

For **local runs** (`python app.py`), set these before starting the app (no keys needed):

- `AZURE_LANGUAGE_ENDPOINT`
- `AZURE_OPENAI_ENDPOINT`
- `PHI_DEPLOYMENT_NAME`
- `AZURE_LANGUAGE_API_VERSION` (optional, defaults to `2023-04-01`)
- `AZURE_OPENAI_API_VERSION` (optional, defaults to `2024-06-01`)

Optional if you want to use service keys instead of Entra ID:

- `AZURE_LANGUAGE_KEY`
- `AZURE_OPENAI_KEY`

Optional for local/demo verification without Azure credentials:

- `DEMO_USE_MOCK_ANALYZERS=true`

For **Azure Web App deployments provisioned by Terraform**, the required endpoints,
deployment name, and service keys are set automatically in App Service application settings.

## Demo: one-command build and teardown with Terraform

Terraform provisions **everything** — AI services and the App Service.
`terraform destroy` tears it all down cleanly.

### What Terraform creates

- Resource Group
- Azure AI Language (`TextAnalytics`) account
- Azure OpenAI account + chat model deployment
- App Service Plan + Linux Web App

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

## GitHub Actions deployment (with teardown option)

This repository includes a manual workflow at:

- `.github/workflows/deploy.yml`

The workflow opts JavaScript-based GitHub Actions into the Node.js 24 runtime to stay ahead of the Node.js 20 deprecation on GitHub-hosted runners.

The workflow supports two operations:

- `deploy`: runs `terraform apply` and then deploys app code with `az webapp up`
- `teardown`: runs `terraform destroy` (requires `confirm_teardown=DESTROY`)

### Required GitHub repository secrets

- `AZURE_CLIENT_ID`
- `AZURE_TENANT_ID`
- `AZURE_SUBSCRIPTION_ID`
- `TERRAFORM_TFVARS` (optional, multi-line content for `infra/terraform/terraform.tfvars`)

### How to run

1. Open **Actions** in GitHub.
2. Select **Deploy or Teardown Azure Environment**.
3. Click **Run workflow**.
4. Choose:
   - `deploy` to provision + deploy app code, or
   - `teardown` and set `confirm_teardown` to `DESTROY` to remove resources.

## Run locally

```bash
az login
python -m pip install -r requirements.txt
python app.py
```

Open `http://localhost:8000`.

`DefaultAzureCredential` uses your `az login` session to authenticate to Azure AI services.
If you are using Entra ID instead of service keys, your account must have the
**Cognitive Services User** role on the Language resource and the
**Cognitive Services OpenAI User** role on the OpenAI resource.

Optional host/port overrides:

- `FLASK_HOST` (defaults to `127.0.0.1`)
- `FLASK_PORT` (defaults to `8000`)

## Run on Azure Web App (Linux)

The App Service is provisioned by Terraform with the required AI service endpoints,
deployment name, and keys. Deploy the code with:

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
