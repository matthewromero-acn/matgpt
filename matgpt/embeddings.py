"""
RAG embedding support for MatGPT.

Uses LM Studio's OpenAI-compatible /v1/embeddings endpoint.
Embedding models are auto-detected by name hints.
"""
from __future__ import annotations

import math
import struct

from openai import OpenAI

from matgpt.config import Config

# Name fragments that identify embedding models
_EMBEDDING_HINTS = ("embed", "nomic", "bge", "e5", "gte", "minilm")


def _is_embedding_model(name: str) -> bool:
    low = name.lower()
    return any(h in low for h in _EMBEDDING_HINTS)


class EmbeddingClient:
    """Thin wrapper around LM Studio's embeddings endpoint."""

    def __init__(self, config: Config) -> None:
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
        self._model: str | None = None

    def _get_model(self) -> str:
        if self._model:
            return self._model
        models = [m.id for m in self._client.models.list().data]
        for m in models:
            if _is_embedding_model(m):
                self._model = m
                return m
        raise RuntimeError(
            "No embedding model found in LM Studio. "
            "Load a model whose name contains 'embed', 'nomic', 'bge', 'e5', or 'gte'."
        )

    def embed(self, text: str) -> list[float]:
        """Return a float embedding vector for the given text."""
        model = self._get_model()
        response = self._client.embeddings.create(model=model, input=text[:8000])
        return response.data[0].embedding

    @property
    def model(self) -> str | None:
        return self._model


# ── Serialization ─────────────────────────────────────────────────────────────

def serialize(embedding: list[float]) -> bytes:
    """Pack a float list to compact binary (float32, 4 bytes/dim)."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def deserialize(data: bytes) -> list[float]:
    """Unpack binary blob back to a float list."""
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


# ── Similarity ────────────────────────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 on zero vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)
