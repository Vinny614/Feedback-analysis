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
  description = "Deployment name used by the application for the Phi model."
  type        = string
  default     = "phi-mini"
}

variable "phi_model_name" {
  description = "Model name for Azure OpenAI deployment."
  type        = string
  default     = "Phi-3-mini-4k-instruct"
}

variable "phi_model_version" {
  description = "Model version for Azure OpenAI deployment."
  type        = string
  default     = "1"
}

variable "phi_deployment_capacity" {
  description = "Capacity units for the Azure OpenAI deployment."
  type        = number
  default     = 1
}
