"""Configuration management for the Advanced RAG system."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Graceful import of python-dotenv
try:
    from dotenv import load_dotenv

    _HAS_DOTENV = True
except ImportError:
    load_dotenv = None
    _HAS_DOTENV = False


@dataclass
class RAGConfig:
    """Configuration settings for the Advanced RAG system.

    Attributes:
        embedding_provider: Embedding model provider ('mock', 'local', 'openai', 'gemini', 'qwen').
        reranker_provider: Reranking provider ('none', 'qwen', 'cohere').
        generator_provider: Generation provider ('mock', 'openai', 'gemini').
        chunk_size: Target token/character chunk size.
        chunk_overlap: Overlap size between adjacent chunks.
        bm25_top_k: Number of candidates retrieved by BM25 sparse search.
        dense_top_k: Number of candidates retrieved by dense vector search.
        final_top_k: Final number of ranked chunks delivered to the generator.
        rrf_k: Reciprocal Rank Fusion smoothing constant.
        max_retry_depth: Maximum recursive query expansion/retry attempts.
        confidence_threshold: Minimum confidence score to accept an answer without fallback.
        corpus_dir: Directory containing source Markdown documents.
        benchmark_path: Path to the benchmark evaluation queries JSON file.
        embedding_model: Optional specific model name for embeddings.
        reranker_model: Optional specific model name for reranking.
        generator_model: Optional specific model name for generation.
        openai_api_key: Optional OpenAI API key.
        gemini_api_key: Optional Gemini API key.
        cohere_api_key: Optional Cohere API key.
        qwen_api_key: Optional Qwen / DashScope API key.
    """

    embedding_provider: str = "mock"
    reranker_provider: str = "none"
    generator_provider: str = "mock"
    chunk_size: int = 800
    chunk_overlap: int = 100
    bm25_top_k: int = 20
    dense_top_k: int = 20
    final_top_k: int = 5
    rrf_k: int = 60
    max_retry_depth: int = 2
    confidence_threshold: float = 0.3
    corpus_dir: str = "data/university"
    benchmark_path: str = "data/benchmark_queries.json"

    # Optional model specifications and credentials
    embedding_model: str | None = None
    reranker_model: str | None = None
    generator_model: str | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    cohere_api_key: str | None = None
    qwen_api_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary."""
        return asdict(self)


def _load_env_fallback(filepath: Path) -> None:
    """Parse a simple .env file when python-dotenv is not installed."""
    if not filepath.is_file():
        return
    try:
        content = filepath.read_text(encoding="utf-8")
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass


def _get_env_int(key: str, default: int) -> int:
    """Safely fetch an integer from environment variables."""
    val = os.getenv(key)
    if val is not None and val.strip():
        try:
            return int(val.strip())
        except ValueError:
            pass
    return default


def _get_env_float(key: str, default: float) -> float:
    """Safely fetch a float from environment variables."""
    val = os.getenv(key)
    if val is not None and val.strip():
        try:
            return float(val.strip())
        except ValueError:
            pass
    return default


def load_config(env_path: str | Path | None = None, **overrides: Any) -> RAGConfig:
    """Load RAG configuration from environment variables, .env file, and explicit overrides.

    Args:
        env_path: Optional path to a specific .env file.
        **overrides: Key-value pairs to override any configuration setting.

    Returns:
        Configured RAGConfig instance.
    """
    # Attempt to load .env file
    if env_path is not None:
        target_path = Path(env_path)
        if _HAS_DOTENV and load_dotenv is not None:
            load_dotenv(dotenv_path=target_path)
        else:
            _load_env_fallback(target_path)
    else:
        if _HAS_DOTENV and load_dotenv is not None:
            load_dotenv()
        else:
            _load_env_fallback(Path(".env"))

    # Extract environment values with fallback to defaults
    cfg_data: dict[str, Any] = {
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "mock").lower(),
        "reranker_provider": os.getenv("RERANKER_PROVIDER", "none").lower(),
        "generator_provider": os.getenv("GENERATOR_PROVIDER", "mock").lower(),
        "chunk_size": _get_env_int("CHUNK_SIZE", 800),
        "chunk_overlap": _get_env_int("CHUNK_OVERLAP", 100),
        "bm25_top_k": _get_env_int("BM25_TOP_K", 20),
        "dense_top_k": _get_env_int("DENSE_TOP_K", 20),
        "final_top_k": _get_env_int("FINAL_TOP_K", 5),
        "rrf_k": _get_env_int("RRF_K", 60),
        "max_retry_depth": _get_env_int("MAX_RETRY_DEPTH", 2),
        "confidence_threshold": _get_env_float("CONFIDENCE_THRESHOLD", 0.3),
        "corpus_dir": os.getenv("CORPUS_DIR", "data/university"),
        "benchmark_path": os.getenv("BENCHMARK_PATH", "data/benchmark_queries.json"),
        "embedding_model": os.getenv("EMBEDDING_MODEL") or os.getenv("OPENAI_EMBEDDING_MODEL"),
        "reranker_model": os.getenv("RERANKER_MODEL"),
        "generator_model": os.getenv("GENERATOR_MODEL"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
        "cohere_api_key": os.getenv("COHERE_API_KEY"),
        "qwen_api_key": os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
    }

    # Apply explicit overrides
    for key, value in overrides.items():
        if value is not None:
            cfg_data[key] = value

    # Filter out keys not present in RAGConfig fields
    field_names = {f.name for f in RAGConfig.__dataclass_fields__.values()}
    filtered_data = {k: v for k, v in cfg_data.items() if k in field_names}

    return RAGConfig(**filtered_data)
