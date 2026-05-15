locals {
  unique_suffix_length      = 6
  max_language_prefix_chars = 14
  max_openai_prefix_chars   = 12
  max_app_prefix_chars      = 16

  normalized_prefix = regexreplace(lower(var.name_prefix), "[^a-z0-9]", "")
  effective_prefix  = local.normalized_prefix != "" ? local.normalized_prefix : "feedback"

  language_prefix = substr(
    local.effective_prefix,
    0,
    min(local.max_language_prefix_chars, length(local.effective_prefix))
  )
  openai_prefix = substr(
    local.effective_prefix,
    0,
    min(local.max_openai_prefix_chars, length(local.effective_prefix))
  )
  app_prefix = substr(
    local.effective_prefix,
    0,
    min(local.max_app_prefix_chars, length(local.effective_prefix))
  )

  language_name = "${local.language_prefix}lang${random_string.suffix.result}"
  openai_name   = "${local.openai_prefix}openai${random_string.suffix.result}"
  app_name      = "${local.app_prefix}app${random_string.suffix.result}"
}

data "azurerm_client_config" "current" {}

resource "random_string" "suffix" {
  length  = local.unique_suffix_length
  upper   = false
  special = false
  numeric = true
}

resource "azurerm_resource_group" "this" {
  name     = var.resource_group_name
  location = var.location
}

resource "azurerm_cognitive_account" "language" {
  name                  = local.language_name
  location              = azurerm_resource_group.this.location
  resource_group_name   = azurerm_resource_group.this.name
  kind                  = "TextAnalytics"
  sku_name              = var.azure_language_sku
  local_auth_enabled    = false
}

resource "azurerm_cognitive_account" "openai" {
  name                  = local.openai_name
  location              = azurerm_resource_group.this.location
  resource_group_name   = azurerm_resource_group.this.name
  kind                  = "OpenAI"
  sku_name              = var.azure_openai_sku
  custom_subdomain_name = local.openai_name
  local_auth_enabled    = false
}

resource "azurerm_cognitive_deployment" "phi" {
  name                 = var.phi_deployment_name
  cognitive_account_id = azurerm_cognitive_account.openai.id

  model {
    format  = "OpenAI"
    name    = var.phi_model_name
    version = var.phi_model_version
  }

  sku {
    name     = "Standard"
    capacity = var.phi_deployment_capacity
  }
}

# ── RBAC for the user running terraform (local development) ──────────────────

resource "azurerm_role_assignment" "deployer_language" {
  scope                = azurerm_cognitive_account.language.id
  role_definition_name = "Cognitive Services User"
  principal_id         = data.azurerm_client_config.current.object_id
}

resource "azurerm_role_assignment" "deployer_openai" {
  scope                = azurerm_cognitive_account.openai.id
  role_definition_name = "Cognitive Services OpenAI User"
  principal_id         = data.azurerm_client_config.current.object_id
}

# ── App Service ───────────────────────────────────────────────────────────────

resource "azurerm_service_plan" "this" {
  name                = "${local.app_prefix}-plan-${random_string.suffix.result}"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  os_type             = "Linux"
  sku_name            = var.app_service_sku
}

resource "azurerm_linux_web_app" "this" {
  name                = local.app_name
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  service_plan_id     = azurerm_service_plan.this.id

  identity {
    type = "SystemAssigned"
  }

  site_config {
    application_stack {
      python_version = "3.11"
    }
    startup_command = "gunicorn --bind=0.0.0.0:$PORT wsgi:application"
  }

  app_settings = {
    AZURE_LANGUAGE_ENDPOINT        = azurerm_cognitive_account.language.endpoint
    AZURE_OPENAI_ENDPOINT          = azurerm_cognitive_account.openai.endpoint
    PHI_DEPLOYMENT_NAME            = azurerm_cognitive_deployment.phi.name
    SCM_DO_BUILD_DURING_DEPLOYMENT = "true"
  }
}

# RBAC for the App Service managed identity

resource "azurerm_role_assignment" "app_language" {
  scope                = azurerm_cognitive_account.language.id
  role_definition_name = "Cognitive Services User"
  principal_id         = azurerm_linux_web_app.this.identity[0].principal_id
}

resource "azurerm_role_assignment" "app_openai" {
  scope                = azurerm_cognitive_account.openai.id
  role_definition_name = "Cognitive Services OpenAI User"
  principal_id         = azurerm_linux_web_app.this.identity[0].principal_id
}
