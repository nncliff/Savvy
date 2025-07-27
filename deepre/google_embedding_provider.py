"""Google Vertex AI text-embedding-004 provider for GraphRAG."""

import asyncio
from typing import Any, Sequence
import httpx
import os
from graphrag.language_model.protocol.base import EmbeddingModel


class GoogleVertexAIEmbeddingModel(EmbeddingModel):
    """Google Vertex AI text-embedding-004 embedding model provider."""

    def __init__(
        self,
        model: str = "text-embedding-004",
        api_key: str | None = None,
        project_id: str | None = None,  # Optional for Gemini API
        region: str = "us-central1",
        use_vertex_ai: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the Google embedding model."""
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.use_vertex_ai = use_vertex_ai
        
        if self.use_vertex_ai:
            # Vertex AI endpoint (requires project ID)
            self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
            self.region = region
            self.base_url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{region}"
            if not self.project_id:
                raise ValueError("Google Cloud project ID is required for Vertex AI")
        else:
            # Gemini API endpoint (no project ID required)
            self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
        if not self.api_key:
            raise ValueError("Google API key is required")

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        """Embed a single text asynchronously."""
        result = await self.aembed_batch([text], **kwargs)
        return result[0]

    async def aembed_batch(
        self, texts: Sequence[str], **kwargs: Any
    ) -> list[list[float]]:
        """Embed a batch of texts asynchronously."""
        if self.use_vertex_ai:
            # Vertex AI format
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            instances = [{"content": text} for text in texts]
            payload = {
                "instances": instances,
                "parameters": {"autoTruncate": True}
            }
            url = f"{self.base_url}/publishers/google/models/{self.model}:predict"
        else:
            # Gemini API format - handle single text differently
            if len(texts) == 1:
                headers = {
                    "Content-Type": "application/json",
                }
                payload = {
                    "content": {"parts": [{"text": texts[0]}]},
                    "task_type": "RETRIEVAL_DOCUMENT"
                }
                url = f"{self.base_url}/models/{self.model}:embedContent?key={self.api_key}"
            else:
                # Batch processing for Gemini
                headers = {
                    "Content-Type": "application/json",
                }
                payload = {
                    "requests": [
                        {
                            "model": f"models/{self.model}",
                            "content": {"parts": [{"text": text}]},
                            "task_type": "RETRIEVAL_DOCUMENT"
                        }
                        for text in texts
                    ]
                }
                url = f"{self.base_url}/models/{self.model}:batchEmbedContents?key={self.api_key}"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=headers,
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()
            
            data = response.json()
            
            if self.use_vertex_ai:
                # Vertex AI response format
                embeddings = [pred["embeddings"]["values"] for pred in data["predictions"]]
            else:
                # Gemini API response format
                if len(texts) == 1:
                    embeddings = [data["embedding"]["values"]]
                else:
                    # Handle batch response - check actual structure
                    if "embeddings" in data:
                        embeddings = []
                        for item in data["embeddings"]:
                            if "embedding" in item:
                                embeddings.append(item["embedding"]["values"])
                            elif "values" in item:
                                embeddings.append(item["values"])
                            else:
                                # Log the actual structure for debugging
                                print(f"Unexpected embedding structure: {item}")
                                embeddings.append([])
                    else:
                        # Fallback if structure is completely different
                        print(f"Unexpected response structure: {data}")
                        embeddings = [[]] * len(texts)
            
            return embeddings

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        """Embed a single text synchronously."""
        try:
            # Try to get existing event loop, create new one if not available
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = None
            
            if loop and loop.is_running():
                # Running in async context, use sync HTTP request
                return self._sync_embed(text, **kwargs)
            else:
                # Not in async context, use asyncio.run
                return asyncio.run(self.aembed(text, **kwargs))
        except Exception:
            # Fallback to sync method
            return self._sync_embed(text, **kwargs)

    def embed_batch(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        """Embed a batch of texts synchronously."""
        try:
            # Try to get existing event loop, create new one if not available
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = None
            
            if loop and loop.is_running():
                # Running in async context, use sync HTTP request
                return self._sync_embed_batch(texts, **kwargs)
            else:
                # Not in async context, use asyncio.run
                return asyncio.run(self.aembed_batch(texts, **kwargs))
        except Exception:
            # Fallback to sync method
            return self._sync_embed_batch(texts, **kwargs)

    def _sync_embed(self, text: str, **kwargs: Any) -> list[float]:
        """Fallback synchronous embedding using httpx."""
        import httpx
        
        if self.use_vertex_ai:
            # Vertex AI format
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            instances = [{"content": text}]
            payload = {
                "instances": instances,
                "parameters": {"autoTruncate": True}
            }
            url = f"{self.base_url}/publishers/google/models/{self.model}:predict"
        else:
            # Gemini API format
            headers = {
                "Content-Type": "application/json",
            }
            payload = {
                "content": {"parts": [{"text": text}]},
                "task_type": "RETRIEVAL_DOCUMENT"
            }
            url = f"{self.base_url}/models/{self.model}:embedContent?key={self.api_key}"

        response = httpx.post(url, headers=headers, json=payload, timeout=30.0)
        response.raise_for_status()
        
        data = response.json()
        
        if self.use_vertex_ai:
            return data["predictions"][0]["embeddings"]["values"]
        else:
            return data["embedding"]["values"]

    def _sync_embed_batch(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        """Fallback synchronous batch embedding."""
        # For batch, use single requests to avoid complexity
        return [self._sync_embed(text, **kwargs) for text in texts]