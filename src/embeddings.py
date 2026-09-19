from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

# Multilingual model suitable for the Vietnamese corpora used in this Lab.
# The local backend remains optional; required checkpoints use MockEmbedder.
LOCAL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"


class MockEmbedder:
    """Deterministic embedding backend used by tests and default classroom runs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._backend_name = "mock embeddings fallback"

    def __call__(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).hexdigest()
        seed = int(digest, 16)
        vector = []
        for _ in range(self.dim):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vector.append((seed / 0xFFFFFFFF) * 2 - 1)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class LocalEmbedder:
    """Sentence Transformers-backed local embedder."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._backend_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return [float(value) for value in embedding]


class OpenAIEmbedder:
    """OpenAI embeddings API-backed embedder."""

    def __init__(self, model_name: str = OPENAI_EMBEDDING_MODEL) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self._backend_name = model_name
        self.client = OpenAI()

    def __call__(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model_name, input=text)
        return [float(value) for value in response.data[0].embedding]


try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


class GeminiEmbedder:
    """Google Gemini embeddings API-backed embedder (google-genai SDK).

    Free-tier alternative to OpenAI for students without an OpenAI key —
    a Gemini API key (aistudio.google.com) has a free quota, no billing card needed.
    """

    def __init__(self, model_name: str = GEMINI_EMBEDDING_MODEL) -> None:
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required for GeminiEmbedder")
        self.model_name = model_name
        self._backend_name = model_name
        self.client = genai.Client(api_key=api_key)
        self._cache_path = Path("data/.embeddings_cache_gemini.json")
        self._cache: dict[str, list[float]] = {}
        self._load_disk_cache()

    def _load_disk_cache(self) -> None:
        try:
            if self._cache_path.exists():
                with open(self._cache_path, encoding="utf-8") as f:
                    self._cache = json.load(f)
        except Exception:
            self._cache = {}

    def _save_disk_cache(self) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f)
        except Exception:
            pass

    def __call__(self, text: str) -> list[float]:
        import time
        if text in self._cache:
            return self._cache[text]
        for attempt in range(3):
            try:
                response = self.client.models.embed_content(model=self.model_name, contents=text)
                vec = [float(value) for value in response.embeddings[0].values]
                self._cache[text] = vec
                self._save_disk_cache()
                return vec
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    time.sleep(12)
                else:
                    time.sleep(1)
        return MockEmbedder()(text)

    def embed_batch(self, texts: list[str], batch_size: int = 25) -> list[list[float]]:
        """Embed a list of texts in batches to respect API rate limits and optimize latency."""
        import time

        results: list[list[float]] = [None] * len(texts)  # type: ignore
        to_fetch_indices = []
        to_fetch_texts = []

        for idx, t in enumerate(texts):
            if t in self._cache:
                results[idx] = self._cache[t]
            else:
                to_fetch_indices.append(idx)
                to_fetch_texts.append(t)

        if not to_fetch_texts:
            return results

        for i in range(0, len(to_fetch_texts), batch_size):
            batch = to_fetch_texts[i : i + batch_size]
            batch_idxs = to_fetch_indices[i : i + batch_size]
            max_retries = 3
            success = False
            for attempt in range(max_retries):
                try:
                    response = self.client.models.embed_content(
                        model=self.model_name,
                        contents=batch,
                    )
                    for b_idx, emb in zip(batch_idxs, response.embeddings):
                        vec = [float(v) for v in emb.values]
                        results[b_idx] = vec
                        self._cache[texts[b_idx]] = vec
                    success = True
                    break
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        time.sleep(15 * (attempt + 1))
                    else:
                        time.sleep(2)

            if not success:
                # Fallback to deterministic vector for failed items
                mock = MockEmbedder()
                for b_idx, t in zip(batch_idxs, batch):
                    results[b_idx] = mock(t)

        self._save_disk_cache()
        return results


def get_embedder(provider: str | None = None) -> Callable[[str], list[float]]:
    """Factory function to get embedder based on provider or environment."""
    chosen = provider or os.getenv(EMBEDDING_PROVIDER_ENV, "mock").lower()

    if chosen == "gemini":
        try:
            return GeminiEmbedder()
        except Exception as e:
            print(f"[WARN] Failed to initialize GeminiEmbedder ({e}), falling back to MockEmbedder")
            return MockEmbedder()
    elif chosen == "local":
        try:
            return LocalEmbedder()
        except Exception as e:
            print(f"[WARN] Failed to initialize LocalEmbedder ({e}), falling back to MockEmbedder")
            return MockEmbedder()
    elif chosen == "openai":
        try:
            return OpenAIEmbedder()
        except Exception as e:
            print(f"[WARN] Failed to initialize OpenAIEmbedder ({e}), falling back to MockEmbedder")
            return MockEmbedder()
    return MockEmbedder()


_mock_embed = MockEmbedder()

