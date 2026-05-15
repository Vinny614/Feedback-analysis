output "azure_language_endpoint" {
  description = "Value for AZURE_LANGUAGE_ENDPOINT."
  value       = azurerm_cognitive_account.language.endpoint
}

output "azure_language_key" {
  description = "Value for AZURE_LANGUAGE_KEY."
  value       = azurerm_cognitive_account.language.primary_access_key
  sensitive   = true
}

output "azure_openai_endpoint" {
  description = "Value for AZURE_OPENAI_ENDPOINT."
  value       = azurerm_cognitive_account.openai.endpoint
}

output "azure_openai_key" {
  description = "Value for AZURE_OPENAI_KEY."
  value       = azurerm_cognitive_account.openai.primary_access_key
  sensitive   = true
}

output "phi_deployment_name" {
  description = "Value for PHI_DEPLOYMENT_NAME."
  value       = azurerm_cognitive_deployment.phi.name
}
