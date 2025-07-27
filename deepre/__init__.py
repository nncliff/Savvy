"""Google Vertex AI integration for GraphRAG."""

from .register_google_provider import register_google_providers

# Register Google providers when package is imported
register_google_providers()