"""Tests for the supplied multi-source benchmark schema and seven metrics."""

import unittest

from scripts.run_advanced_eval import evaluate_query


class TestGoldEvaluation(unittest.TestCase):
    def test_multi_source_recall_full_evidence_and_answer_criteria(self):
        query = {
            "id": "Q1",
            "query": "Điều kiện OJT?",
            "source_doc_ids": ["academic", "ojt"],
            "expected_audience": "student",
            "evidence": [
                {"id": "credits", "phrases": ["90% tín chỉ"]},
                {"id": "orientation", "phrases": ["Orientation bắt buộc"]},
            ],
            "answer_criteria": [
                {"id": "credits", "all_terms": ["90%", "tín chỉ"]},
                {"id": "orientation", "all_terms": ["Orientation", "bắt buộc"]},
            ],
        }
        chunks = [
            {"content": "Sinh viên hoàn thành 90% tín chỉ.",
             "chunk_metadata": {"doc_id": "academic", "audience": "student"}},
            {"content": "Orientation bắt buộc trước OJT.",
             "chunk_metadata": {"doc_id": "ojt", "audience": "student"}},
        ]
        result = evaluate_query(
            query,
            chunks,
            "Sinh viên cần đủ 90% tín chỉ và tham gia Orientation bắt buộc.",
            1.0,
        )
        self.assertEqual(result.recall_at_1, 0.5)
        self.assertEqual(result.recall_at_5, 1.0)
        self.assertEqual(result.full_evidence_at_5, 1.0)
        self.assertEqual(result.faithfulness_score, 1.0)
        self.assertEqual(result.audience_match_rate, 1.0)

    def test_full_evidence_requires_every_group(self):
        query = {
            "id": "Q",
            "source_doc_ids": ["ojt"],
            "evidence": [
                {"phrases": ["90% tín chỉ"]},
                {"phrases": ["Orientation bắt buộc"]},
            ],
        }
        chunks = [{"content": "90% tín chỉ", "chunk_metadata": {"doc_id": "ojt", "audience": "student"}}]
        result = evaluate_query(query, chunks, "90% tín chỉ", 1.0)
        self.assertEqual(result.full_evidence_at_5, 0.0)


if __name__ == "__main__":
    unittest.main()
