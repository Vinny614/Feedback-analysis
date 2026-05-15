resource "random_string" "suffix" {
  length  = 6
  upper   = false
  special = false
  numeric = true
}

locals {
  normalized_prefix = regexreplace(lower(var.name_prefix), "[^a-z0-9]", "")
  language_name     = "${substr(local.normalized_prefix, 0, 14)}lang${random_string.suffix.result}"
  openai_name       = "${substr(local.normalized_prefix, 0, 12)}openai${random_string.suffix.result}"
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
