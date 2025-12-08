"""语义切分工具，复刻 legacy agent 的分段策略。"""
from __future__ import annotations

import math
import re
from typing import List, Sequence

try:
    from llama_index.core.base.embeddings.base import BaseEmbedding  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from llama_index.core.embeddings.base import BaseEmbedding  # type: ignore
    except ImportError:  # pragma: no cover
        try:
            from llama_index.embeddings.base import BaseEmbedding  # type: ignore
        except ImportError:  # pragma: no cover
            class BaseEmbedding(object):  # type: ignore
                def get_text_embedding(self, text: str):  # noqa: D401
                    raise NotImplementedError

_SENTENCE_PATTERN = re.compile(r"(?<=[。！？!?])\s*")


def _split_sentences(text: str) -> List[str]:
    sentences = [segment.strip() for segment in _SENTENCE_PATTERN.split(text) if segment and segment.strip()]
    if sentences:
        return sentences
    return [text.strip()] if text else []


def semantic_split(
    text: str,
    embed_model: BaseEmbedding,
    *,
    threshold: float = 0.5,
    min_sentence_len: int = 20,
) -> List[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    try:
        embeddings: List[Sequence[float]] = [embed_model.get_text_embedding(sentence) for sentence in sentences]
    except Exception:
        return [text]

    chunks: List[List[str]] = [[sentences[0]]]

    def _cosine(v1: Sequence[float], v2: Sequence[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if not norm1 or not norm2:
            return 0.0
        return dot / (norm1 * norm2)

    for idx in range(1, len(sentences)):
        prev_sentence = sentences[idx - 1]
        similarity = _cosine(embeddings[idx - 1], embeddings[idx])
        is_prev_short = len(prev_sentence) < min_sentence_len

        if similarity < threshold and not is_prev_short:
            chunks.append([sentences[idx]])
        else:
            chunks[-1].append(sentences[idx])

    return ["".join(chunk) for chunk in chunks if chunk]


__all__ = ["semantic_split"]
