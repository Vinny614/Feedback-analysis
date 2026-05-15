locals {
  unique_suffix_length      = 6
  max_language_prefix_chars = 14
  max_openai_prefix_chars   = 12

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

  language_name = "${local.language_prefix}lang${random_string.suffix.result}"
  openai_name   = "${local.openai_prefix}openai${random_string.suffix.result}"
}

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
  name                = local.language_name
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  kind                = "TextAnalytics"
  sku_name            = var.azure_language_sku
}

resource "azurerm_cognitive_account" "openai" {
  name                  = local.openai_name
  location              = azurerm_resource_group.this.location
  resource_group_name   = azurerm_resource_group.this.name
  kind                  = "OpenAI"
  sku_name              = var.azure_openai_sku
  custom_subdomain_name = local.openai_name
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
