output "azure_language_endpoint" {
  description = "Value for AZURE_LANGUAGE_ENDPOINT."
  value       = azurerm_cognitive_account.language.endpoint
}

output "azure_openai_endpoint" {
  description = "Value for AZURE_OPENAI_ENDPOINT."
  value       = azurerm_cognitive_account.openai.endpoint
}

output "phi_deployment_name" {
  description = "Value for PHI_DEPLOYMENT_NAME."
  value       = azurerm_cognitive_deployment.phi.name
}

output "app_url" {
  description = "Public URL of the deployed demo web application."
  value       = "https://${azurerm_linux_web_app.this.default_hostname}"
}

output "app_name" {
  description = "Azure Web App name."
  value       = azurerm_linux_web_app.this.name
}

output "resource_group_name" {
  description = "Azure resource group name used for the deployment."
  value       = azurerm_resource_group.this.name
}

output "bing_connection_id" {
  description = "Resource ID used for Bing grounding connection."
  value       = azurerm_bing_grounding_service.this.id
}
