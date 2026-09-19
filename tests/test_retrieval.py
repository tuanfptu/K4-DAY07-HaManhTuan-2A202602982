"""Unit tests for the Advanced RAG retrieval system."""

import unittest

from src.advanced_rag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HyDERetriever,
    HybridRetriever,
    MultiQueryRetriever,
    ParentChildRetriever,
    filter_by_audience,
    filter_chunks,
    reciprocal_rank_fusion,
)


class TestBM25Retriever(unittest.TestCase):
    def setUp(self):
        self.retriever = BM25Retriever(k1=1.5, b=0.75)
        self.sample_chunks = [
            {
                "id": "c1",
                "content": "Quy chế đào tạo đại học chính quy tại Đại học FPT.",
                "retrieval_content": "Quy chế đào tạo đại học chính quy tại Đại học FPT bao gồm các quy định về học phần và tín chỉ.",
                "chunk_metadata": {"audience": "student", "category": "academic"},
            },
            {
                "id": "c2",
                "content": "Hướng dẫn nộp học phí và chính sách tài chính cơ sở HCM.",
                "retrieval_content": "Hướng dẫn nộp học phí sinh viên và chính sách tài chính tại campus TP Hồ Chí Minh.",
                "chunk_metadata": {"audience": "student", "category": "tuition"},
            },
            {
                "id": "c3",
                "content": "Quy định tham gia thực tập doanh nghiệp OJT dành cho sinh viên.",
                "retrieval_content": "Quy định tham gia thực tập doanh nghiệp OJT dành cho sinh viên hoàn thành đủ tín chỉ chuyên ngành.",
                "chunk_metadata": {"audience": "student", "category": "ojt"},
            },
        ]

    def test_tokenize_vietnamese(self):
        text = "Sinh viên FPT, học phí & học bổng (2026)!"
        tokens = self.retriever._tokenize(text)
        self.assertIn("sinh", tokens)
        self.assertIn("viên", tokens)
        self.assertIn("fpt", tokens)
        self.assertIn("học", tokens)
        self.assertIn("phí", tokens)
        self.assertIn("bổng", tokens)
        self.assertIn("2026", tokens)

    def test_index_and_search(self):
        self.retriever.index(self.sample_chunks)
        results = self.retriever.search("học phí sinh viên", top_k=2)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        # c2 is about tuition / hoc phi
        top_idx, top_score = results[0]
        self.assertEqual(top_idx, 1)
        self.assertGreater(top_score, 0.0)

    def test_empty_query_or_corpus(self):
        self.assertEqual(self.retriever.search(""), [])
        empty_retriever = BM25Retriever()
        empty_retriever.index([])
        self.assertEqual(empty_retriever.search("test"), [])


class TestDenseRetriever(unittest.TestCase):
    def setUp(self):
        self.retriever = DenseRetriever()  # uses MockEmbedder
        self.sample_chunks = [
            {
                "id": "c1",
                "content": "Chương trình trao đổi quốc tế cho sinh viên FPT.",
                "retrieval_content": "Chương trình trao đổi quốc tế cho sinh viên FPT tại campus HCM.",
            },
            {
                "id": "c2",
                "content": "Đăng ký học bổng tài năng Đại học FPT năm 2026.",
                "retrieval_content": "Đăng ký học bổng tài năng Đại học FPT năm 2026.",
            },
        ]

    def test_index_and_search(self):
        self.retriever.index(self.sample_chunks)
        results = self.retriever.search("học bổng FPT", top_k=2)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        idx, score = results[0]
        self.assertIsInstance(idx, int)
        self.assertIsInstance(score, float)

    def test_custom_embedder(self):
        # Deterministic dummy embedder
        custom_embedder = lambda text: [1.0, 0.0] if "học bổng" in text else [0.0, 1.0]
        retriever = DenseRetriever(embedding_fn=custom_embedder)
        retriever.index(self.sample_chunks)
        results = retriever.search("học bổng", top_k=1)
        self.assertEqual(results[0][0], 1)  # c2 contains học bổng


class TestReciprocalRankFusion(unittest.TestCase):
    def test_empty_rankings(self):
        self.assertEqual(reciprocal_rank_fusion([]), [])
        self.assertEqual(reciprocal_rank_fusion([[], []]), [])

    def test_rrf_scoring(self):
        # List 1: doc 0 (rank 1), doc 1 (rank 2)
        # List 2: doc 1 (rank 1), doc 2 (rank 2)
        ranking1 = [(0, 10.0), (1, 5.0)]
        ranking2 = [(1, 8.0), (2, 4.0)]
        fused = reciprocal_rank_fusion([ranking1, ranking2], k=60)

        # Expected scores:
        # doc 1: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
        # doc 0: 1/(60+1) = 1/61 = 0.016393
        # doc 2: 1/(60+2) = 1/62 = 0.016129
        self.assertEqual(fused[0][0], 1)
        self.assertEqual(fused[1][0], 0)
        self.assertEqual(fused[2][0], 2)
        self.assertAlmostEqual(fused[0][1], (1 / 61) + (1 / 62), places=5)


class TestHybridRetriever(unittest.TestCase):
    def setUp(self):
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()
        self.hybrid = HybridRetriever(bm25=self.bm25, dense=self.dense, rrf_k=60)
        self.sample_chunks = [
            {"id": "c0", "retrieval_content": "Thủ tục xin cấp bảng điểm tại FAP."},
            {"id": "c1", "retrieval_content": "Hướng dẫn nộp học phí qua ngân hàng."},
            {"id": "c2", "retrieval_content": "Quy chế thi lại và phúc khảo bài thi."},
        ]

    def test_index_and_search(self):
        self.hybrid.index(self.sample_chunks)
        results = self.hybrid.search("học phí", bm25_top_k=5, dense_top_k=5, final_top_k=2)
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)
        self.assertEqual(self.hybrid.get_chunk(1)["id"], "c1")
        self.assertIsNone(self.hybrid.get_chunk(99))


class TestParentChildRetriever(unittest.TestCase):
    def test_child_search_expands_and_deduplicates_parents(self):
        children = [
            {"id": "p1__c1", "parent_id": "p1", "content": "học phí AI",
             "parent_content": "Toàn bộ bảng học phí AI và các ngành.",
             "retrieval_content": "học phí AI", "chunk_metadata": {"doc_id": "fees"}},
            {"id": "p1__c2", "parent_id": "p1", "content": "31.600.000 đồng",
             "parent_content": "Toàn bộ bảng học phí AI và các ngành.",
             "retrieval_content": "học phí AI 31.600.000", "chunk_metadata": {"doc_id": "fees"}},
            {"id": "p2__c1", "parent_id": "p2", "content": "quy chế OJT",
             "parent_content": "Toàn bộ quy chế OJT.",
             "retrieval_content": "quy chế OJT", "chunk_metadata": {"doc_id": "ojt"}},
        ]
        hybrid = HybridRetriever(BM25Retriever(), DenseRetriever())
        retriever = ParentChildRetriever(hybrid)
        retriever.index(children)
        results = retriever.search("học phí AI", child_top_k=3, parent_top_k=3)

        self.assertEqual(results[0]["parent_id"], "p1")
        self.assertEqual(results[0]["content"], "Toàn bộ bảng học phí AI và các ngành.")
        self.assertEqual(len([r for r in results if r["parent_id"] == "p1"]), 1)
        self.assertIn("matched_child_id", results[0])


class TestMetadataFilter(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"id": "c1", "chunk_metadata": {"audience": "student", "category": "ojt"}},
            {"id": "c2", "chunk_metadata": {"audience": "faculty", "category": "salary"}},
            {"id": "c3", "chunk_metadata": {"audience": "student", "category": "tuition"}},
            {"id": "c4", "metadata": {"audience": "staff", "category": "admin"}},
        ]

    def test_filter_none(self):
        res = filter_chunks(self.chunks, None)
        self.assertEqual(len(res), 4)

    def test_filter_matching(self):
        res = filter_chunks(self.chunks, {"audience": "student"})
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["id"], "c1")
        self.assertEqual(res[1]["id"], "c3")

    def test_filter_multiple_criteria(self):
        res = filter_chunks(self.chunks, {"audience": "student", "category": "ojt"})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["id"], "c1")

    def test_filter_by_audience(self):
        students = filter_by_audience(self.chunks, "student")
        self.assertEqual(len(students), 2)
        staff = filter_by_audience(self.chunks, "staff")
        self.assertEqual(len(staff), 1)
        self.assertEqual(staff[0]["id"], "c4")


class TestMultiQueryRetriever(unittest.TestCase):
    def setUp(self):
        self.dense = DenseRetriever()
        self.dense.index([
            {"id": "c0", "retrieval_content": "Quy định tham gia OJT tại Đại học FPT."},
            {"id": "c1", "retrieval_content": "Thủ tục xin cấp thẻ sinh viên."},
        ])

    def test_deterministic_variants(self):
        mq = MultiQueryRetriever(retriever=self.dense)
        queries = mq.generate_queries("điều kiện đi OJT")
        self.assertGreaterEqual(len(queries), 3)
        self.assertEqual(queries[0], "điều kiện đi OJT")

    def test_llm_query_generation(self):
        mock_llm = lambda prompt: "Quy định thực tập OJT FPT\nTiêu chuẩn xét duyệt OJT\nHướng dẫn đăng ký OJT"
        mq = MultiQueryRetriever(retriever=self.dense, llm_fn=mock_llm)
        queries = mq.generate_queries("điều kiện đi OJT")
        self.assertGreaterEqual(len(queries), 4)
        self.assertEqual(queries[0], "điều kiện đi OJT")

    def test_search(self):
        mq = MultiQueryRetriever(retriever=self.dense)
        results = mq.search("OJT", top_k=2)
        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 1)


class TestHyDERetriever(unittest.TestCase):
    def setUp(self):
        self.dense = DenseRetriever()
        self.dense.index([
            {"id": "c0", "retrieval_content": "Quy định đào tạo thực tập doanh nghiệp OJT Đại học FPT."},
            {"id": "c1", "retrieval_content": "Chính sách tài chính học phí sinh viên Đại học FPT."},
        ])

    def test_generate_hypothetical_fallback(self):
        hyde = HyDERetriever(dense_retriever=self.dense)
        passage = hyde._generate_hypothetical("Điều kiện tham gia OJT")
        self.assertIn("OJT", passage)
        self.assertIn("FPT", passage)

    def test_search_with_and_without_llm(self):
        hyde_fallback = HyDERetriever(dense_retriever=self.dense)
        results_fallback = hyde_fallback.search("học phí cơ sở HCM", top_k=2)
        self.assertIsInstance(results_fallback, list)
        self.assertEqual(len(results_fallback), 2)

        mock_llm = lambda p: "Theo quy định tài chính Đại học FPT, học phí sinh viên được nộp theo từng kỳ."
        hyde_llm = HyDERetriever(dense_retriever=self.dense, llm_fn=mock_llm)
        results_llm = hyde_llm.search("học phí", top_k=2)
        self.assertIsInstance(results_llm, list)
        self.assertEqual(len(results_llm), 2)


if __name__ == "__main__":
    unittest.main()
