"""配置加载模块：统一从 .env / 环境变量读取运行所需参数。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent


def _env(key: str, *, default: Optional[str] = None, required: bool = False) -> str:
    value = os.getenv(key)
    if value not in (None, ""):
        return value
    if default is not None:
        return default
    if required:
        raise ValueError(f"缺少必要环境变量: {key}")
    return ""


def _env_optional(key: str) -> Optional[str]:
    value = os.getenv(key)
    if value in (None, ""):
        return None
    return value


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class EmbeddingConfig:
    api_key: str
    base_url: str
    model: str


@dataclass(frozen=True)
class LLMConfig:
    api_key: Optional[str]
    base_url: str
    model: str


@dataclass(frozen=True)
class StorageConfig:
    sqlite_path: Path
    chroma_path: Path


@dataclass(frozen=True)
class RuntimeToggles:
    similarity_top_k: int
    answer_context_k: int
    enable_focus_hints: bool
    enable_time_filters: bool
    llm_temperature: float


@dataclass(frozen=True)
class Settings:
    description: str
    embedding: EmbeddingConfig
    llm: LLMConfig
    chunking_method: str
    storage: StorageConfig
    runtime: RuntimeToggles


def load_settings(description: Optional[str] = None) -> Settings:
    """从环境变量构造 Settings。"""
    load_dotenv()

    embedding = EmbeddingConfig(
        api_key=_env("EMBEDDING_API_KEY", required=True),
        base_url=_env("EMBEDDING_BASE_URL", default="https://api.openai.com/v1"),
        model=_env("EMBEDDING_MODEL", required=True),
    )

    llm = LLMConfig(
        api_key=_env_optional("LLM_API_KEY"),
        base_url=_env("LLM_BASE_URL", default="https://api.openai.com/v1"),
        model=_env("LLM_MODEL", required=True),

    )

    chunking_method = _env("CHUNK_METHOD", default="semantic_split")
    sqlite_default = PROJECT_ROOT / "sqlite_db" / "sqlite.db"

    chroma_default = PROJECT_ROOT / "chroma_db"

    storage = StorageConfig(
        sqlite_path=Path(_env("SQLITE_PATH", default=str(sqlite_default))).resolve(),
        chroma_path=Path(_env("CHROMA_PATH", default=str(chroma_default))).resolve(),

    )

    runtime = RuntimeToggles(
        similarity_top_k=int(os.getenv("SIMILARITY_TOP_K", "5")),
        answer_context_k=int(os.getenv("ANSWER_CONTEXT_K", "5")),
        enable_focus_hints=_env_bool("ENABLE_FOCUS_HINTS", True),
        enable_time_filters=_env_bool("ENABLE_TIME_FILTERS", True),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.7),
    )

    desc = description or os.getenv("PROJECT_DESCRIPTION", "NJU RAG Hub Next")

    return Settings(
        description=desc,
        embedding=embedding,
        llm=llm,
        chunking_method=chunking_method,
        storage=storage,
        runtime=runtime,
    )


__all__ = [
    "Settings",
    "EmbeddingConfig",
    "LLMConfig",
    "StorageConfig",
    "RuntimeToggles",
    "load_settings",
]
