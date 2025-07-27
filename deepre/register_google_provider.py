"""Register Google Vertex AI provider with GraphRAG ModelFactory."""

from graphrag.language_model.factory import ModelFactory
from google_embedding_provider import GoogleVertexAIEmbeddingModel

def register_google_providers():
    """Register Google Vertex AI providers with GraphRAG."""
    
    # Register the Google Vertex AI embedding model
    ModelFactory.register_embedding(
        "google_vertex_ai_embedding",
        lambda **config: GoogleVertexAIEmbeddingModel(
            model=config.get("model", "text-embedding-004"),
            api_key=config.get("api_key"),
            project_id=config.get("project_id"),
            region=config.get("region", "us-central1"),
            **config.get("kwargs", {}),
        ),
    )

if __name__ == "__main__":
    # Register the providers when this module is imported
    register_google_providers()
    print("Google Vertex AI providers registered successfully")