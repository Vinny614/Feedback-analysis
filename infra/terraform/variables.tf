variable "resource_group_name" {
  description = "Azure resource group name."
  type        = string
  default     = "feedback-analysis-rg"
}

variable "location" {
  description = "Azure region for all resources."
  type        = string
  default     = "eastus"
}

variable "app_service_location" {
  description = "Azure region for the Linux Web App resources. Defaults to westus2 to avoid eastus App Service quota issues."
  type        = string
  default     = "westus2"
}

variable "name_prefix" {
  description = "Prefix used for globally unique Azure service names."
  type        = string
  default     = "feedbackanalysis"
}

variable "azure_language_sku" {
  description = "SKU for Azure AI Language."
  type        = string
  default     = "S"
}

variable "azure_openai_sku" {
  description = "SKU for Azure OpenAI."
  type        = string
  default     = "S0"
}

variable "phi_deployment_name" {
  description = "Deployment name used by the application for the Azure OpenAI chat model."
  type        = string
  default     = "chat-model"
}

variable "phi_model_name" {
  description = "Model name for the Azure OpenAI chat deployment."
  type        = string
  default     = "gpt-4o-mini"
}

variable "phi_model_version" {
  description = "Model version for the Azure OpenAI chat deployment."
  type        = string
  default     = "2024-07-18"
}

variable "phi_deployment_capacity" {
  description = "Capacity units for the Azure OpenAI deployment."
  type        = number
  default     = 1
}

variable "app_service_sku" {
  description = "SKU for the App Service Plan (e.g. B1, B2, S1)."
  type        = string
  default     = "B1"
}
