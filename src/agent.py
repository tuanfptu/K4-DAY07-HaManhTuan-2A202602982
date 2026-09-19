from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(
        self,
        store: EmbeddingStore,
        llm_fn: Callable[[str], str],
    ) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # Handle empty store — return message, don't call LLM
        if self.store.get_collection_size() == 0:
            return "Không tìm thấy tài liệu nào trong cơ sở tri thức."

        # 1. Retrieve top-k relevant chunks
        results = self.store.search(question, top_k=top_k)

        if not results:
            return "Không tìm thấy thông tin liên quan đến câu hỏi."

        # 2. Build context with numbered chunks + source info
        context_parts: list[str] = []
        for i, result in enumerate(results, start=1):
            source = result.get("metadata", {}).get("doc_id", "unknown")
            context_parts.append(
                f"[{i}] (nguồn: {source})\n{result['content']}"
            )

        context = "\n\n".join(context_parts)

        # 3. Build prompt with grounding constraints
        prompt = (
            "Dựa trên các ngữ cảnh được cung cấp bên dưới, "
            "hãy trả lời câu hỏi. "
            "Trích dẫn số nguồn [1], [2], ... khi trả lời. "
            "Nếu ngữ cảnh không chứa câu trả lời, "
            "hãy nói rõ là không tìm thấy thông tin.\n\n"
            f"NGỮ CẢNH:\n{context}\n\n"
            f"CÂU HỎI: {question}\n\n"
            "TRẢ LỜI:"
        )

        return self.llm_fn(prompt)
