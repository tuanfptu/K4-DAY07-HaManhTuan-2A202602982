"""Prompt construction and answer generation for RAG."""

from __future__ import annotations

from typing import Any, Callable


class PromptBuilder:
    """Build prompts for the RAG pipeline.

    Constructs structured prompts with numbered context chunks,
    source citations, and grounding constraints.
    """

    SYSTEM_TEMPLATE = (
        "Bạn là trợ lý tư vấn sinh viên chính thức của Trường Đại học FPT TP.HCM. "
        "Nhiệm vụ của bạn là giải đáp câu hỏi của sinh viên dựa HOÀN TOÀN và CHÍNH XÁC trên tài liệu được cung cấp.\n\n"
        "CÁC QUY TẮC BẮT BUỘC:\n"
        "1. Trả lời trực tiếp, đầy đủ và chính xác vào câu hỏi, nêu rõ các thông tin quan trọng.\n"
        "2. BẮT BUỘC giữ nguyên và trích dẫn đầy đủ các con số, tỷ lệ phần trăm (%), thời hạn, số ngày, số tín chỉ, mức học phí, điểm GPA, số điện thoại hotline, địa chỉ email, tên phòng ban và chức danh từ ngữ cảnh.\n"
        "3. Tuyệt đối KHÔNG tự bịa đặt, suy đoán hay sử dụng thông tin ngoài ngữ cảnh được cung cấp.\n"
        "4. Kèm trích dẫn nguồn số [1], [2] tương ứng với từng ý trả lời."
    )

    def build_rag_prompt(
        self,
        query: str,
        contexts: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> str:
        """Build a complete RAG prompt with context and question.

        Args:
            query: User's question.
            contexts: List of retrieved chunk dicts with 'content', 'metadata'.
            system_prompt: Optional custom system prompt.

        Returns:
            Complete prompt string for the LLM.
        """
        system = system_prompt or self.SYSTEM_TEMPLATE
        context_str = self._format_contexts(contexts)

        if not contexts:
            return (
                f"{system}\n\n"
                "Không tìm thấy ngữ cảnh liên quan trong cơ sở tri thức.\n\n"
                f"CÂU HỎI: {query}\n\n"
                "TRẢ LỜI: Xin lỗi, tôi không tìm thấy thông tin liên quan "
                "trong bộ tài liệu hiện tại để trả lời câu hỏi này."
            )

        return (
            f"{system}\n\n"
            f"NGỮ CẢNH:\n{context_str}\n\n"
            f"CÂU HỎI: {query}\n\n"
            "TRẢ LỜI:"
        )

    def _format_contexts(self, contexts: list[dict[str, Any]]) -> str:
        """Format retrieved contexts with source citations."""
        parts: list[str] = []
        for i, ctx in enumerate(contexts, start=1):
            # Extract source info
            metadata = ctx.get("metadata", ctx.get("chunk_metadata", {}))
            source_parts: list[str] = []

            doc_id = metadata.get("doc_id", "unknown")
            source_parts.append(doc_id)

            if chapter := metadata.get("chapter"):
                source_parts.append(chapter)
            if article := metadata.get("article"):
                source_parts.append(article)

            source_ref = " | ".join(source_parts)
            content = ctx.get("content", "")
            score = ctx.get("score", ctx.get("rerank_score", ""))
            score_str = f" (score: {score:.3f})" if isinstance(score, float) else ""

            parts.append(
                f"[{i}] [Source: {source_ref}]{score_str}\n{content}"
            )

        return "\n\n".join(parts)


class CitationBuilder:
    """Build structured citations from retrieval results."""

    def build_citations(
        self,
        contexts: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Build citation objects from contexts.

        Returns:
            List of citation dicts with 'index', 'source', 'section', 'preview'.
        """
        citations: list[dict[str, str]] = []
        for i, ctx in enumerate(contexts, start=1):
            metadata = ctx.get("metadata", ctx.get("chunk_metadata", {}))
            content = ctx.get("content", "")

            citation = {
                "index": str(i),
                "source": metadata.get("doc_id", "unknown"),
                "section": self._build_section_ref(metadata),
                "preview": content[:200].replace("\n", " "),
            }
            citations.append(citation)

        return citations

    def _build_section_ref(self, metadata: dict) -> str:
        """Build a human-readable section reference."""
        parts: list[str] = []
        if chapter := metadata.get("chapter"):
            parts.append(chapter)
        if article := metadata.get("article"):
            parts.append(article)
        if section := metadata.get("section"):
            parts.append(section)
        return " > ".join(parts) if parts else "N/A"

    def format_citations_text(
        self,
        citations: list[dict[str, str]],
    ) -> str:
        """Format citations as a readable text block."""
        if not citations:
            return ""

        lines: list[str] = ["\n--- Nguồn tham khảo ---"]
        for c in citations:
            lines.append(
                f"[{c['index']}] {c['source']}"
                + (f" | {c['section']}" if c['section'] != 'N/A' else "")
            )
        return "\n".join(lines)


def get_gemini_llm(model_name: str = "gemini-3.6-flash") -> Callable[[str], str] | None:
    """Initialize a callable Gemini LLM generator function if API key is present."""
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    try:
        from google import genai
        client = genai.Client(api_key=api_key)

        def _generate(prompt: str) -> str:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            return response.text or ""

        return _generate
    except Exception:
        return None


class AnswerGenerator:
    """Generate answers using retrieved context and LLM.

    Combines prompt building, LLM invocation, and citation formatting.
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str] | None = None,
        prompt_builder: PromptBuilder | None = None,
        citation_builder: CitationBuilder | None = None,
        use_gemini: bool = True,
    ) -> None:
        if llm_fn is not None:
            self.llm_fn = llm_fn
        elif use_gemini:
            self.llm_fn = get_gemini_llm()
        else:
            self.llm_fn = None

        self.prompt_builder = prompt_builder or PromptBuilder()
        self.citation_builder = citation_builder or CitationBuilder()

    def generate(
        self,
        query: str,
        contexts: list[dict[str, Any]],
        include_citations: bool = True,
    ) -> dict[str, Any]:
        """Generate an answer with citations.

        Returns:
            Dict with 'answer', 'citations', 'prompt'.
        """
        prompt = self.prompt_builder.build_rag_prompt(query, contexts)

        if self.llm_fn:
            try:
                answer = self.llm_fn(prompt)
            except Exception as e:
                answer = f"Lỗi khi gọi LLM: {e}"
        else:
            answer = self._mock_answer(query, contexts)

        citations = (
            self.citation_builder.build_citations(contexts)
            if include_citations
            else []
        )

        if include_citations and citations:
            citation_text = self.citation_builder.format_citations_text(citations)
            answer = f"{answer}\n{citation_text}"

        return {
            "answer": answer,
            "citations": citations,
            "prompt": prompt,
        }

    def _mock_answer(
        self,
        query: str,
        contexts: list[dict[str, Any]],
    ) -> str:
        """Generate a well-grounded factual answer synthesizing top retrieved contexts."""
        if not contexts:
            return (
                "Không tìm thấy thông tin liên quan trong "
                "cơ sở tri thức để trả lời câu hỏi này."
            )

        import re
        q_words = set(re.findall(r"\w+", query.lower()))
        stopwords = {
            "là", "gì", "như", "thế", "nào", "bao", "nhiêu", "có", "không", "cho",
            "của", "tại", "ở", "và", "được", "sinh", "viên", "phải", "để", "khi",
            "các", "một", "những", "với", "trong", "theo", "đến", "trường"
        }
        content_words = {w for w in q_words if w not in stopwords and len(w) > 1}

        scored_sentences: list[tuple[int, str]] = []
        for ctx in contexts[:5]:
            content = ctx.get("content", "")
            lines = [
                line.strip()
                for line in re.split(r"(?<=[.!?\n])\s+", content)
                if len(line.strip()) > 15
            ]
            for line in lines:
                l_lower = line.lower()
                overlap = sum(1 for w in content_words if w in l_lower)
                scored_sentences.append((overlap, line))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences: list[str] = []
        seen: set[str] = set()
        for score, line in scored_sentences:
            if line not in seen:
                seen.add(line)
                top_sentences.append(line)
                if len(top_sentences) >= 8:
                    break

        if top_sentences:
            body = "\n\n".join(f"- {s}" for s in top_sentences)
        else:
            top_content = contexts[0].get("content", "")
            body = top_content[:1500]

        return (
            f"Dựa trên tài liệu quy định và chính sách chính thức của Trường Đại học FPT TP.HCM, "
            f"thông tin giải đáp cho câu hỏi '{query}' như sau:\n\n{body}"
        )
