"""
Embeddings module for Universal Semantic Structure.

Provides embedding functionality via OpenRouter API.
"""

import asyncio
import os
from pathlib import Path
from typing import List, Optional

import httpx

# Load .env from project root
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, rely on system env vars


class EmbeddingClient:
    """Client for generating embeddings via OpenRouter API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model or os.environ.get(
            "OPENROUTER_EMBED_MODEL", "openai/text-embedding-3-small"
        )
        self.base_url = base_url

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var or pass api_key."
            )

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": texts,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Sort by index to ensure correct order
            embeddings = sorted(data["data"], key=lambda x: x["index"])
            return [e["embedding"] for e in embeddings]

    def embed_sync(self, text: str) -> List[float]:
        """Synchronous wrapper for embed()."""
        return asyncio.run(self.embed(text))

    def embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous wrapper for embed_batch()."""
        return asyncio.run(self.embed_batch(texts))


# Module-level singleton
_embedding_client: Optional[EmbeddingClient] = None


def get_embedding_client() -> EmbeddingClient:
    """Get or create the embedding client singleton."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client


async def embed_text(text: str) -> List[float]:
    """Convenience function to embed text."""
    client = get_embedding_client()
    return await client.embed(text)


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Convenience function to embed multiple texts."""
    client = get_embedding_client()
    return await client.embed_batch(texts)
