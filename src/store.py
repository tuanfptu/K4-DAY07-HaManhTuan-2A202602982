from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

    # ------------------------------------------------------------------
    # Helper: normalize a Document into a stored record
    # ------------------------------------------------------------------

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # Copy metadata so we never mutate the caller's object
        metadata = dict(doc.metadata) if doc.metadata else {}

        # Guarantee doc_id exists — delete_document depends on it.
        # When chunks are created from a file, the caller should set
        # metadata["doc_id"] to the *file* stem; if they don't, fall
        # back to doc.id so nothing breaks.
        if "doc_id" not in metadata:
            metadata["doc_id"] = doc.id

        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": self._embedding_fn(doc.content),
        }

    # ------------------------------------------------------------------
    # Helper: similarity search over an arbitrary set of records
    # ------------------------------------------------------------------

    def _search_records(
        self,
        query: str,
        records: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not records:
            return []

        query_embedding = self._embedding_fn(query)

        scored: list[tuple[float, dict[str, Any]]] = []
        for record in records:
            score = _dot(query_embedding, record["embedding"])
            scored.append((score, record))

        # Sort by score descending
        scored.sort(key=lambda pair: pair[0], reverse=True)

        # Return top_k results WITHOUT the embedding (keeps output clean)
        results: list[dict[str, Any]] = []
        for score, record in scored[:top_k]:
            results.append({
                "id": record["id"],
                "content": record["content"],
                "metadata": record["metadata"],
                "score": score,
            })

        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        for doc in docs:
            self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: dict = None,
    ) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # No filter → same as plain search (both go through _search_records)
        if not metadata_filter:
            return self._search_records(query, self._store, top_k)

        # Filter BEFORE searching — not after!
        # If we searched first and then discarded non-matching results,
        # we could end up with 0 results even though matching docs exist
        # (because non-matching docs consumed all top_k slots).
        candidates = [
            record for record in self._store
            if all(
                record["metadata"].get(key) == value
                for key, value in metadata_filter.items()
            )
        ]

        return self._search_records(query, candidates, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        before = len(self._store)
        self._store = [
            record for record in self._store
            if record["metadata"].get("doc_id") != doc_id
        ]
        return len(self._store) < before
