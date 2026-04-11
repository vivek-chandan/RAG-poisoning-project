"""Custom exceptions for secure RAG services."""


class SecureRAGError(Exception):
    """Base exception for secure RAG failures."""


class ConfigurationError(SecureRAGError):
    """Raised for invalid runtime configuration."""


class ProviderError(SecureRAGError):
    """Raised for external provider failures."""
