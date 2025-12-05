"""Embedding adapter for OpenRouter or other OpenAI-compatible providers."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import List, Sequence

import openai

try:  # pragma: no cover - llama-index version shim
    from llama_index.core.base.embeddings.base import BaseEmbedding  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from llama_index.core.embeddings.base import BaseEmbedding  # type: ignore
    except ImportError:  # pragma: no cover
        try:
            from llama_index.embeddings.base import BaseEmbedding  # type: ignore
        except ImportError:  # pragma: no cover
            class BaseEmbedding(object):  # type: ignore
                """Fallback so the adapter can still operate without llama-index."""

                def __init__(self, *args, **kwargs):
                    super().__init__()

logger = logging.getLogger(__name__)


class OpenRouterEmbedding(BaseEmbedding):
    """Lightweight embedding wrapper that accepts arbitrary model names."""

    def __init__(self, api_key: str, base_url: str, model: str):
        if not api_key:
            raise ValueError("OpenRouterEmbedding requires a valid API key")
        if not base_url:
            raise ValueError("OpenRouterEmbedding requires a base URL")
        super().__init__(model_name=model)
        self._model = model
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url.rstrip("/"))
        try:
            self._max_retries = max(1, int(os.getenv("EMBEDDING_MAX_RETRIES", "3")))
        except ValueError:
            self._max_retries = 3
        try:
            self._retry_backoff = float(os.getenv("EMBEDDING_RETRY_BACKOFF", "1.5"))
        except ValueError:
            self._retry_backoff = 1.5

    def _batch_embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=list(texts),
                )
                data = getattr(response, "data", None)
                if not data:
                    raise ValueError("No embedding data received")
                return [item.embedding for item in data]
            except Exception as exc:  # pragma: no cover - network heavy
                last_error = exc
                logger.warning(
                    "Embedding request failed (attempt %d/%d): %s",
                    attempt,
                    self._max_retries,
                    exc,
                )
                if attempt == self._max_retries:
                    break
                time.sleep(self._retry_backoff * attempt)
        raise RuntimeError("Embedding API failed") from last_error

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._batch_embed([text])[0]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._batch_embed([query])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)
