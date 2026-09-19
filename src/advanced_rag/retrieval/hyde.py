"""Hypothetical Document Embeddings (HyDE) retriever."""

from __future__ import annotations

from typing import Callable

from .dense_retriever import DenseRetriever


class HyDERetriever:
    """Hypothetical Document Embeddings (HyDE) retriever.

    Transforms a user query into a hypothetical regulatory or policy document passage
    before performing dense semantic retrieval, bridging the vocabulary and stylistic gap
    between short user questions and formal handbook texts.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        llm_fn: Callable[[str], str] | None = None,
    ) -> None:
        """Initialize HyDERetriever with a DenseRetriever and optional LLM generator.

        Args:
            dense_retriever: DenseRetriever instance storing chunk embeddings.
            llm_fn: Optional callable taking a prompt and generating text passage.
        """
        self.dense_retriever = dense_retriever
        self.llm_fn = llm_fn

    def _generate_hypothetical(self, query: str) -> str:
        """Generate a template-based hypothetical passage using query keywords and policy patterns.

        Args:
            query: User query string.

        Returns:
            A synthetic document passage matching the formal style of university regulations.
        """
        clean_q = query.strip().rstrip("?.!")
        q_lower = clean_q.lower()

        if any(term in q_lower for term in ("ojt", "thực tập", "doanh nghiệp")):
            return (
                f"Căn cứ Quy định đào tạo và Hướng dẫn thực tập tốt nghiệp (OJT) tại Đại học FPT: "
                f"Sinh viên tham gia học phần OJT phải tích lũy đủ số tín chỉ quy định của chuyên ngành, "
                f"đạt chuẩn ngoại ngữ đầu ra theo lộ trình và hoàn thành tuần lễ định hướng (Orientation). "
                f"Đối với vấn đề {clean_q}, sinh viên thực hiện đăng ký qua cổng thông tin và nộp hồ sơ theo đúng hạn thông báo."
            )

        if any(term in q_lower for term in ("học phí", "tiền học", "nộp tiền", "tài chính")):
            return (
                f"Theo Quy định tài chính sinh viên Đại học FPT: "
                f"Mức học phí từng học kỳ được quy định căn cứ theo chương trình đào tạo và lộ trình môn học. "
                f"Sinh viên có nghĩa vụ hoàn tất học phí đúng thời hạn thông báo trên cổng thông tin đào tạo FAP "
                f"để được xếp lớp chính thức. Về nội dung {clean_q}, sinh viên theo dõi thông báo từ phòng Kế toán."
            )

        if any(term in q_lower for term in ("học bổng", "xét học bổng", "duy trì học bổng")):
            return (
                f"Căn cứ Quy chế cấp và duy trì học bổng tại Đại học FPT: "
                f"Sinh viên nhận học bổng cần duy trì điểm trung bình học kỳ (GPA) và điểm rèn luyện theo ngưỡng chuẩn, "
                f"không vi phạm kỷ luật và hoàn thành đầy đủ số tín chỉ mỗi kỳ. "
                f"Các điều kiện và thủ tục liên quan đến {clean_q} được phòng Dịch vụ sinh viên xét duyệt hàng kỳ."
            )

        if any(term in q_lower for term in ("thi lại", "học lại", "phúc khảo", "điểm danh", "vắng", "chuyên cần")):
            return (
                f"Theo Quy chế đào tạo và khảo thí Đại học FPT: "
                f"Sinh viên cần tham gia tối thiểu 80% thời lượng các buổi học để đạt điều kiện chuyên cần dự thi cuối kỳ. "
                f"Trường hợp chưa đạt hoặc muốn cải thiện kết quả liên quan đến {clean_q}, sinh viên nộp đơn yêu cầu trên FAP "
                f"theo đúng thời hạn quy định và hoàn thành thủ tục tài chính nếu có."
            )

        # General university policy template
        return (
            f"Căn cứ Sổ tay sinh viên và Quy chế đào tạo đại học chính quy tại Đại học FPT: "
            f"Nhà trường quy định chi tiết quyền lợi, điều kiện thực hiện và quy trình hỗ trợ đối với {clean_q}. "
            f"Sinh viên tra cứu thông tin chính thức trên cổng thông tin đào tạo FAP và liên hệ phòng ban chức năng để được giải đáp."
        )

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Retrieve top_k documents using Hypothetical Document Embeddings.

        Args:
            query: User query string.
            top_k: Number of ranked results to return (default: 10).

        Returns:
            List of (chunk_index, score) pairs from dense retrieval.
        """
        if not query or not query.strip() or top_k <= 0:
            return []

        clean_query = query.strip()
        hypothetical_passage = ""

        if self.llm_fn is not None:
            try:
                prompt = (
                    f"Bạn là chuyên gia soạn thảo quy chế đào tạo đại học.\n"
                    f"Hãy viết một đoạn văn bản quy định hoặc thông báo học vụ ngắn (2-3 câu) "
                    f"bằng tiếng Việt giải thích hoặc hướng dẫn cụ thể cho vấn đề sau:\n"
                    f"\"{clean_query}\"\n"
                    f"Văn phong hành chính, chuẩn mực, tập trung vào điều kiện và quy trình thực hiện."
                )
                response = self.llm_fn(prompt)
                if isinstance(response, str) and response.strip():
                    hypothetical_passage = response.strip()
            except Exception:
                hypothetical_passage = ""

        if not hypothetical_passage:
            hypothetical_passage = self._generate_hypothetical(clean_query)

        return self.dense_retriever.search(hypothetical_passage, top_k=top_k)
