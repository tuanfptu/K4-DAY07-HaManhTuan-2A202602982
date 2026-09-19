"""
Tests for advanced chunking strategies for Vietnamese university policy documents.
"""

from __future__ import annotations

import unittest

from src.advanced_rag.chunking import (
    ContextualChunker,
    HeadingChunker,
    ParentChildChunker,
)
from src.advanced_rag.chunking.utils import (
    clean_heading,
    parse_number_or_roman,
    remove_vietnamese_accents,
    roman_to_int,
    slugify,
)


class TestChunkingUtilities(unittest.TestCase):
    """Test suite for utility functions in utils.py."""

    def test_roman_to_int(self) -> None:
        self.assertEqual(roman_to_int("I"), 1)
        self.assertEqual(roman_to_int("II"), 2)
        self.assertEqual(roman_to_int("III"), 3)
        self.assertEqual(roman_to_int("IV"), 4)
        self.assertEqual(roman_to_int("VI"), 6)
        self.assertEqual(roman_to_int("IX"), 9)
        self.assertEqual(roman_to_int("XII"), 12)
        self.assertEqual(roman_to_int("XX"), 20)
        self.assertIsNone(roman_to_int("ABC"))
        self.assertIsNone(roman_to_int(""))

    def test_parse_number_or_roman(self) -> None:
        self.assertEqual(parse_number_or_roman("6"), "6")
        self.assertEqual(parse_number_or_roman("II"), "2")
        self.assertEqual(parse_number_or_roman("12a"), "12a")
        self.assertEqual(parse_number_or_roman("IV"), "4")

    def test_remove_vietnamese_accents(self) -> None:
        raw = "Quy chế đào tạo đại học chính quy"
        expected = "quy che dao tao dai hoc chinh quy"
        self.assertEqual(remove_vietnamese_accents(raw), expected)

    def test_slugify(self) -> None:
        self.assertEqual(slugify("QUY CHẾ ĐÀO TẠO"), "quy_che_dao_tao")
        self.assertEqual(slugify("Điều 6. Học phần"), "dieu_6_hoc_phan")
        self.assertEqual(slugify("---"), "sec")

    def test_clean_heading(self) -> None:
        self.assertEqual(clean_heading("#### **CHƯƠNG I: QUY ĐỊNH**"), "CHƯƠNG I: QUY ĐỊNH")
        self.assertEqual(clean_heading("**Điều 1. Phạm vi áp dụng**"), "Điều 1. Phạm vi áp dụng")
        self.assertEqual(clean_heading("[Toggle](#)"), "Toggle")


class TestHeadingChunker(unittest.TestCase):
    """Test suite for HeadingChunker."""

    def setUp(self) -> None:
        self.sample_doc = (
            "# QUY CHẾ ĐÀO TẠO ĐẠI HỌC CHÍNH QUY\n\n"
            "## CHƯƠNG II: TỔ CHỨC ĐÀO TẠO\n\n"
            "### Điều 6. Học phần và tín chỉ\n\n"
            "1. Tín chỉ là đơn vị quy chuẩn để lượng hóa khối lượng kiến thức.\n"
            "2. Một tín chỉ được quy định bằng 15 tiết học lý thuyết hoặc 30 tiết thực hành.\n\n"
            "### Điều 7. Đăng ký học phần\n\n"
            "Sinh viên thực hiện đăng ký học phần qua cổng FAP đúng thời hạn quy định."
        )

    def test_empty_input(self) -> None:
        chunker = HeadingChunker()
        self.assertEqual(chunker.chunk(""), [])
        self.assertEqual(chunker.chunk("   \n\t  "), [])

    def test_chunk_output_structure(self) -> None:
        chunker = HeadingChunker(chunk_size=1000)
        chunks = chunker.chunk(self.sample_doc, metadata={"doc_id": "reg_01", "title": "QUY CHẾ ĐÀO TẠO"})
        self.assertGreater(len(chunks), 0)

        first = chunks[0]
        self.assertIn("id", first)
        self.assertIn("chunk_id", first)
        self.assertIn("content", first)
        self.assertIn("retrieval_content", first)
        self.assertIn("chunk_metadata", first)

        meta = first["chunk_metadata"]
        self.assertEqual(meta["doc_id"], "reg_01")
        self.assertIn("section_path", meta)
        self.assertIn("part_index", meta)

    def test_structured_chunk_id_format(self) -> None:
        chunker = HeadingChunker(chunk_size=1000)
        chunks = chunker.chunk(self.sample_doc, metadata={"doc_id": "reg_01", "title": "QUY CHẾ ĐÀO TẠO"})

        # First chunk is Điều 6 under Chapter II
        first_id = chunks[0]["chunk_id"]
        self.assertEqual(first_id, "reg_01__chapter_2__article_6__part_1")

        # Second chunk is Điều 7 under Chapter II
        second_id = chunks[1]["chunk_id"]
        self.assertEqual(second_id, "reg_01__chapter_2__article_7__part_1")

    def test_context_prefix_format(self) -> None:
        chunker = HeadingChunker(chunk_size=1000, include_context_prefix=True)
        chunks = chunker.chunk(self.sample_doc, metadata={"doc_id": "reg_01", "title": "QUY CHẾ ĐÀO TẠO"})

        first = chunks[0]
        retrieval = first["retrieval_content"]
        content = first["content"]

        # Content has original text
        self.assertTrue(content.startswith("1. Tín chỉ là đơn vị quy chuẩn"))
        self.assertNotIn("[SECTION]", content)

        # Retrieval content has prefix
        self.assertTrue(retrieval.startswith("[SECTION]"))
        self.assertIn("Document: QUY CHẾ ĐÀO TẠO", retrieval)
        self.assertIn("CHƯƠNG II", retrieval)
        self.assertIn("Điều 6", retrieval)
        self.assertTrue(retrieval.endswith(content))

    def test_recursive_split_oversized_body(self) -> None:
        # Create a large body under a section that exceeds chunk_size
        long_paragraph = "Khoản này chứa nội dung chi tiết về quy định học vụ tại trường. " * 30
        doc = (
            "# QUY CHẾ ĐÀO TẠO\n\n"
            "## CHƯƠNG II: TỔ CHỨC ĐÀO TẠO\n\n"
            f"### Điều 6. Học phần\n\n{long_paragraph}"
        )

        chunker = HeadingChunker(chunk_size=500, include_context_prefix=True)
        chunks = chunker.chunk(doc, metadata={"doc_id": "reg_01", "title": "QUY CHẾ ĐÀO TẠO"})

        self.assertGreater(len(chunks), 1)
        part_indices = [c["chunk_metadata"]["part_index"] for c in chunks]
        self.assertEqual(part_indices, list(range(1, len(chunks) + 1)))

        # All sub-chunks must preserve the same heading context and prefix
        for i, c in enumerate(chunks, start=1):
            self.assertEqual(c["chunk_id"], f"reg_01__chapter_2__article_6__part_{i}")
            self.assertIn("[SECTION]", c["retrieval_content"])
            self.assertIn("CHƯƠNG II", c["retrieval_content"])
            self.assertIn("Điều 6", c["retrieval_content"])

    def test_vietnamese_bold_policy_patterns(self) -> None:
        # Pattern often used in Vietnamese policy documents: #### **CHƯƠNG I...** and **Điều 1...**
        doc = (
            "#### **CHƯƠNG I: NHỮNG VẤN ĐỀ CHUNG**\n\n"
            "**Điều 1. Phạm vi áp dụng và đối tượng áp dụng**\n\n"
            "1. Quy chế này cụ thể hóa một số quy định về tổ chức đào tạo.\n\n"
            "**Điều 2. Sinh viên và khóa sinh viên**\n\n"
            "1. Sinh viên là người đáp ứng yêu cầu đầu vào."
        )

        chunker = HeadingChunker(chunk_size=1000)
        chunks = chunker.chunk(doc, metadata={"doc_id": "policy_fpt", "title": "Quy chế"})

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["chunk_id"], "policy_fpt__chapter_1__article_1__part_1")
        self.assertEqual(chunks[1]["chunk_id"], "policy_fpt__chapter_1__article_2__part_1")

        meta1 = chunks[0]["chunk_metadata"]
        self.assertEqual(meta1["chapter_number"], 1)
        self.assertEqual(meta1["article_number"], 1)

    def test_khoan_clause_pattern(self) -> None:
        doc = (
            "### Điều 10. Học bổng\n\n"
            "#### Khoản 1. Điều kiện xét học bổng khuyến khích\n\n"
            "Sinh viên có điểm GPA từ 8.0 trở lên.\n\n"
            "#### Khoản 2. Mức học bổng\n\n"
            "Mức học bổng tối đa là 100% học phí."
        )

        chunker = HeadingChunker(chunk_size=1000)
        chunks = chunker.chunk(doc, metadata={"doc_id": "scholarship", "title": "Học bổng"})

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["chunk_id"], "scholarship__article_10__clause_1__part_1")
        self.assertEqual(chunks[1]["chunk_id"], "scholarship__article_10__clause_2__part_1")


class TestParentChildChunker(unittest.TestCase):
    """Test suite for ParentChildChunker."""

    def setUp(self) -> None:
        self.doc = (
            "# QUY CHẾ ĐÀO TẠO\n\n"
            "## CHƯƠNG II: TỔ CHỨC ĐÀO TẠO\n\n"
            "### Điều 6. Học phần\n\n"
            "1. Tín chỉ là đơn vị quy chuẩn để lượng hóa khối lượng học tập. "
            "Một tín chỉ được quy định bằng 15 tiết lý thuyết hoặc 30 tiết thực hành. "
            "Để hoàn thành khối lượng kiến thức của 1 tín chỉ, sinh viên cần tự học tối thiểu 30 tiết. "
            "Tổng số tín chỉ tối đa của chương trình được xác định theo từng ngành đào tạo cụ thể.\n\n"
            "2. Học phần là khối lượng kiến thức tương đối trọn vẹn, thuận tiện cho sinh viên tích lũy. "
            "Mỗi học phần được quy định một số lượng tín chỉ nhất định và có mã học phần riêng."
        )

    def test_empty_input(self) -> None:
        chunker = ParentChildChunker()
        self.assertEqual(chunker.chunk(""), [])

    def test_parent_child_chunks(self) -> None:
        chunker = ParentChildChunker(
            parent_chunk_size=2000,
            child_chunk_size=150,
            child_overlap=20,
        )
        chunks = chunker.chunk(self.doc, metadata={"doc_id": "reg_01", "title": "QUY CHẾ ĐÀO TẠO"})

        self.assertGreater(len(chunks), 1)

        # Check required keys
        for c in chunks:
            self.assertIn("id", c)
            self.assertIn("chunk_id", c)
            self.assertIn("parent_id", c)
            self.assertIn("content", c)
            self.assertIn("parent_content", c)
            self.assertIn("retrieval_content", c)
            self.assertIn("chunk_metadata", c)

            # Child content must be smaller than or equal to parent content
            self.assertLessEqual(len(c["content"]), len(c["parent_content"]))
            # Parent content contains child content
            self.assertIn(c["content"][:30], c["parent_content"])

    def test_extract_parents_helper(self) -> None:
        chunker = ParentChildChunker(child_chunk_size=120)
        chunks = chunker.chunk(self.doc, metadata={"doc_id": "reg_01", "title": "QUY CHẾ"})
        parents = chunker.extract_parents(chunks)

        self.assertIsInstance(parents, dict)
        self.assertEqual(len(parents), 1)  # All children belong to the single Điều 6 parent
        pid = next(iter(parents))
        self.assertTrue(pid.startswith("reg_01__chapter_2__article_6"))


class TestContextualChunker(unittest.TestCase):
    """Test suite for ContextualChunker."""

    def setUp(self) -> None:
        self.doc = (
            "# QUY CHẾ ĐÀO TẠO ĐẠI HỌC\n\n"
            "## CHƯƠNG II: TỔ CHỨC ĐÀO TẠO\n\n"
            "### Điều 6. Học phần\n\n"
            "Sinh viên phải đăng ký học phần đúng tiến độ."
        )

    def test_empty_input(self) -> None:
        chunker = ContextualChunker()
        self.assertEqual(chunker.chunk(""), [])

    def test_deterministic_context_format(self) -> None:
        chunker = ContextualChunker(chunk_size=800)
        metadata = {
            "title": "QUY CHẾ ĐÀO TẠO ĐẠI HỌC",
            "doc_id": "reg_01",
            "audience": "sinh viên chính quy",
        }
        chunks = chunker.chunk(self.doc, metadata=metadata)

        self.assertEqual(len(chunks), 1)
        first = chunks[0]

        original_text = first["content"]
        self.assertEqual(original_text, "Sinh viên phải đăng ký học phần đúng tiến độ.")

        retrieval = first["retrieval_content"]
        # Format: "Document: {title}. Chapter: {chapter}. Article: {article}. Audience: {audience}.\n\n{original_text}"
        expected_header = (
            "Document: QUY CHẾ ĐÀO TẠO ĐẠI HỌC. "
            "Chapter: CHƯƠNG II: TỔ CHỨC ĐÀO TẠO. "
            "Article: Điều 6. Học phần. "
            "Audience: sinh viên chính quy."
        )
        self.assertEqual(first["context_header"], expected_header)
        self.assertEqual(retrieval, f"{expected_header}\n\n{original_text}")

    def test_missing_fields_defaults(self) -> None:
        # Document without chapter or article headings
        plain_doc = "Thông báo chung về lịch nghỉ Tết Nguyên Đán năm 2026."
        chunker = ContextualChunker(chunk_size=800)
        chunks = chunker.chunk(plain_doc, metadata={"id": "tet_2026", "title": "Thông báo nghỉ Tết"})

        self.assertEqual(len(chunks), 1)
        header = chunks[0]["context_header"]

        # Chapter and Article default cleanly to "N/A"
        self.assertIn("Document: Thông báo nghỉ Tết.", header)
        self.assertIn("Chapter: N/A.", header)
        self.assertIn("Article: N/A.", header)
        self.assertIn("Audience: sinh viên.", header)


if __name__ == "__main__":
    unittest.main()
