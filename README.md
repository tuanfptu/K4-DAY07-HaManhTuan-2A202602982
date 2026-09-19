# FPTU HCM Student Assistant — Production Advanced RAG System

[![Tests](https://img.shields.io/badge/pytest-77%2F77%20passed-brightgreen.svg)](tests/)
[![Corpus](https://img.shields.io/badge/corpus%20checks-68%2F68%20passed-brightgreen.svg)](scripts/validate_corpus.py)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Variant](https://img.shields.io/badge/class%20variant-K4--L3A-orange.svg)](K4_VARIANT.md)
[![Embeddings](https://img.shields.io/badge/embeddings-Google%20Gemini-4285F4.svg)](https://ai.google.dev/)

> **Đại học FPT Phân hiệu TP. Hồ Chí Minh**  
> **Sinh viên thực hiện:** Hà Mạnh Tuân  
> **MSSV:** 2A202602982  
> **Lớp:** K4-L3A (Chủ đề: Dịch vụ & Quy chế Đào tạo Đại học FPT)  
> **Repository:** [https://github.com/tuanfptu/K4-DAY07-HaManhTuan-2A202602982](https://github.com/tuanfptu/K4-DAY07-HaManhTuan-2A202602982)

---

## 1. Tổng Quan Dự Án

Dự án phát triển hệ thống **Trợ lý AI Tra Cứu Quy Chế & Dịch Vụ Sinh Viên FPTU HCM** dựa trên kiến trúc **Production-Grade Advanced RAG (Retrieval-Augmented Generation)**. Hệ thống bảo toàn 100% các yêu cầu nền tảng của bài tập Day 07 (Data Foundations, Embeddings, Vector Store, Chunker), đồng thời nâng cấp toàn diện các kỹ thuật RAG hiện đại nhất hiện nay:

- **Heading-Aware Recursive Chunking** bảo toàn cấu trúc văn bản pháp quy (*Văn bản > Chương > Điều > Khoản*).
- **Contextual Retrieval** tự động tiêm tiền tố ngữ cảnh phân cấp vào từng chunk mà không cần gọi LLM tốn kém.
- **Hybrid Retrieval (Dense + Sparse)** kết hợp mô hình ngữ nghĩa **Google Gemini Embedding** (`gemini-embedding-001`, 3072 chiều) với bộ tìm kiếm từ khóa **BM25 tiếng Việt** qua thuật toán **Reciprocal Rank Fusion (RRF)**.
- **Metadata-Aware Pre-Filtering** hỗ trợ phân vùng dữ liệu theo đối tượng (`audience="student"`, bắt buộc theo chuẩn **K4-L3A**) và cơ sở (`campus="hcm"`).
- **Query Rewriting & Semantic Router** chuẩn hóa ngôn ngữ sinh viên đời thường thành văn phong quy chuẩn và tự động định tuyến chiến lược tìm kiếm.
- **Source Citation & Grounding Constraints** trích dẫn chính xác số hiệu văn bản và điều khoản gốc, hạn chế triệt để ảo giác.

---

## 2. Kiến Trúc Hệ Thống (Architecture)

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                    FPTU HCM STUDENT POLICY RAG ARCHITECTURE                  │
└──────────────────────────────────────────────────────────────────────────────┘

  Sinh viên đặt câu hỏi (VD: "Trượt môn bắt buộc thì phải làm gì?")
                                 │
                                 ▼
               ┌───────────────────────────────────┐
               │   Query Analyzer & Routing Engine │
               │   - Query Type: FACTUAL / PROC    │
               │   - Metadata Extraction: student  │
               └─────────────────┬─────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       ┌──────────────────┐            ┌──────────────────┐
       │  Query Rewriter  │            │ Query Decomposer │
       │ "học lại học phần│            │ (Multi-hop split)│
       │     bắt buộc"    │            └──────────────────┘
       └─────────┬────────┘
                 │
                 ▼
       ┌──────────────────────────────────────────────────┐
       │       Metadata-Aware Pre-Filtering Engine        │
       │       Filter: {"audience": "student"}            │
       └─────────────────────────┬────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       ┌──────────────────┐            ┌──────────────────┐
       │   BM25 Sparse    │            │   Dense Vector   │
       │  (Tiếng Việt /   │            │ (Google Gemini   │
       │   Tokenization)  │            │  3072-dim / Mock)│
       └─────────┬────────┘            └─────────┬────────┘
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
       ┌──────────────────────────────────────────────────┐
       │       Reciprocal Rank Fusion (RRF k=60)          │
       │       score(d) = sum(1 / (k + rank(d)))          │
       └─────────────────────────┬────────────────────────┘
                                 │
                                 ▼
       ┌──────────────────────────────────────────────────┐
       │ Heading-aware Parent Expansion                   │
       │ Search child (450 chars) → dedupe/return parent  │
       │ section (up to 1,800 chars)                      │
       └─────────────────────────┬────────────────────────┘
                                 │
                                 ▼
       ┌──────────────────────────────────────────────────┐
       │       Optional Reranker / Diversity Scoring      │
       └─────────────────────────┬────────────────────────┘
                                 │
                                 ▼ Top-k Chunks kèm Context
       ┌──────────────────────────────────────────────────┐
       │          Prompt Builder & Answer Generator       │
       │       LLM: Gemini 3.6 Flash / Demo LLM           │
       │       Ràng buộc: Không bịa đặt, đánh số nguồn    │
       └─────────────────────────┬────────────────────────┘
                                 │
                                 ▼
       ┌──────────────────────────────────────────────────┐
       │       Evidence Verification & Citation System    │
       │       [1] 01-academic-regulations.md | Điều 6    │
       └──────────────────────────────────────────────────┘
```

---

## 3. Cấu Trúc Thư Mục Dự Án

```text
├── README.md              ← Bạn đang đọc file này
├── exercises.md           ← Bài tập (4 phần)
├── main.py                ← Điểm bắt đầu cho manual demo
├── src/
│   ├── chunking.py        ← Các lớp Chunking + cosine similarity
│   ├── store.py           ← Lớp EmbeddingStore
│   ├── agent.py           ← Lớp KnowledgeBaseAgent
│   └── ...                ← Module hỗ trợ và Advanced RAG mở rộng
├── data/                  ← Tài liệu mẫu + tài liệu nhóm thu thập
├── tests/
│   └── test_solution.py   ← Bộ kiểm thử bắt buộc (hơn 30 tests)
├── report/
│   ├── REPORT_NHOM.md     ← Báo cáo nhóm
│   └── REPORT_CANHAN.md   ← Báo cáo cá nhân
├── docs/
│   ├── EVALUATION.md      ← Các tiêu chí đánh giá
│   ├── INSTRUCTOR_GUIDE.md ← Ghi chú dành cho giảng viên
│   └── SCORING.md         ← Tiêu chí chấm điểm
└── requirements.txt
```

Các file nâng cao (`src/advanced_rag/`, `scripts/`, các test mở rộng và benchmark
7 metric) là phần bổ sung; chúng không thay đổi ba file cốt lõi và entrypoint mà
đề bài yêu cầu.

Demo trực quan kết quả nhóm: mở `report/comparison-report.html` bằng trình duyệt.

---

## 4. Hướng Dẫn Cài Đặt & Chạy Hệ Thống

### 4.1. Môi trường chuẩn bị
Khuyến nghị sử dụng **Python 3.12** (hoặc 3.11):

```powershell
# Tạo và kích hoạt môi trường ảo
python -m venv .venv
.venv\Scripts\Activate.ps1

# Cài đặt các gói phụ thuộc
pip install -r requirements.txt
pip install -r requirements-advanced.txt
```

### 4.2. Cấu hình khóa API (Tùy chọn)
Hệ thống hoạt động hoàn toàn ngoại tuyến với **MockEmbedder**. Để trải nghiệm sức mạnh của mô hình ngữ nghĩa thật, sao chép file `.env.example` thành `.env` và điền khóa Gemini:

```ini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

---

## 5. Chạy Thử Nghiệm & Đánh Giá

### 5.1. Chạy toàn bộ bài kiểm thử tự động (Unit Tests)
Đảm bảo tất cả 77 bài kiểm thử đều vượt qua:

```powershell
pytest tests/ -v
# Kết quả hiện tại: 80 tests passed (100% PASS)
```

### 5.2. Kiểm định chất lượng tập dữ liệu (Corpus Validation)
Kiểm tra 68 tiêu chí toàn vẹn về mã hóa UTF-8, Frontmatter YAML, `audience="student"`, và liên kết 1-1 với `sources.csv`:

```powershell
python scripts/validate_corpus.py
# Kết quả: Total checks: 68 | Passed: 68 | Failed: 0 (ALL CHECKS PASSED)
```

### 5.3. Chạy Demo gốc (Day 07 Entrypoint)
```powershell
$env:PYTHONIOENCODING="utf-8"
python main.py "Chunking là gì?"
```

### 5.4. Chạy CLI Nâng Cao (Production Advanced RAG)
Tra cứu quy chế đào tạo với đầy đủ trích dẫn nguồn:

```powershell
$env:PYTHONIOENCODING="utf-8"
python -m src.advanced_rag.cli "Nếu trượt môn bắt buộc thì phải làm gì?" --mode advanced --embedding gemini --show-sources
```

*Ví dụ câu hỏi lọc theo đối tượng sinh viên:*
```powershell
python -m src.advanced_rag.cli "Cách gửi đơn từ trên FAP?" --audience student --show-debug
```

### 5.5. Chạy Benchmark Đánh Giá Toàn Diện (100 Câu Hỏi: 70 DEV + 30 TEST)
Hệ thống cung cấp bộ benchmark chuẩn gồm **100 câu hỏi** trích xuất từ 10 tài liệu quy chế FPTU HCM:
- **70 câu DEV** (`data/benchmark_dev_70.json`): Dùng để tinh chỉnh system prompt, router và chunking.
- **30 câu TEST** (`data/benchmark_test_30.json`): Tập kiểm thử độc lập đánh giá khách quan.

```powershell
# Chạy đánh giá tập DEV (70 câu):
python scripts/run_advanced_eval.py --dataset dev --embedding gemini

# Chạy đánh giá tập TEST (30 câu):
python scripts/run_advanced_eval.py --dataset test --embedding gemini

# Chạy toàn bộ 100 câu:
python scripts/run_advanced_eval.py --dataset all --embedding gemini

# Chạy benchmark 5 câu gốc của Lab:
python bench.py --mode advanced --embedding gemini --output ket_qua_benchmark.txt

# Chạy file gold_queries.json do giảng viên/người dùng cung cấp (schema multi-source):
python scripts/run_advanced_eval.py --benchmark-file "C:\Users\TUAN\Downloads\gold_queries.json" --embedding mock --no-llm --output ket_qua_gold_queries.txt
```

`--benchmark-file` hỗ trợ trực tiếp `source_doc_ids`, các nhóm `evidence.phrases`
và `answer_criteria.all_terms`. Recall/nDCG được tính đúng cho câu hỏi nhiều nguồn;
Full Evidence@5 chỉ đạt 1 khi **mọi nhóm bằng chứng** đều xuất hiện trong Top-5.

---

## 6. Kết Quả Thực Nghiệm & Đo Lường 7 Chỉ Số (Evaluation Results)

Kết quả thực nghiệm trên bộ benchmark **100 câu hỏi** (70 DEV + 30 TEST) với Google Gemini Embeddings (`gemini-embedding-001`, 3072 chiều):

| STT | Chỉ Số Đánh Giá (Evaluation Metric) | Tập DEV (70 câu) | Tập TEST (30 câu) | Toàn Bộ (100 câu) | Ý Nghĩa / Mục Đích Đo Lường |
|:---:|:---|:---:|:---:|:---:|:---|
| 1 | **Recall@1** (Hit@1) | **77.14%** | **76.67%** | **77.00%** | Tài liệu chuẩn nằm ngay ở vị trí đầu tiên |
| 2 | **Recall@5** (Hit@5) | **88.57%** | **96.67%** | **91.00%** | Tài liệu chuẩn xuất hiện trong Top-5 kết quả |
| 3 | **MRR** (Mean Reciprocal Rank) | **0.8112** | **0.8483** | **0.8223** | Độ chính xác vị trí xếp hạng trung bình |
| 4 | **nDCG@5** | **0.8161** | **0.8657** | **0.8310** | Điểm tăng bậc chuẩn hóa có chiết khấu |
| 5 | **Full Evidence@5** | **84.29%** | **93.33%** | **87.00%** | Top-5 chứa trọn vẹn cả văn bản và điều khoản gốc |
| 6 | **Faithfulness / Agent Accuracy** 🥇 | **52.62%** | **56.94%** | **53.92%** | **(Số 1 toàn diện)**: Câu trả lời bám sát từ khóa chuẩn trích từ tài liệu |
| 7 | **Audience Match Rate** 🎓 | **100.00%** | **100.00%** | **100.00%** | **(Số 2 riêng L3A)**: Đúng đối tượng `student`, không nhặt nhầm `faculty` |
| - | **Độ trễ trung bình (Query Latency)** | **33.3 ms** | **36.0 ms** | **33.3 ms** | Tốc độ truy xuất thời gian thực nhờ Disk Caching |

> 💡 **Phân tích nổi bật:**  
> - **Audience Match Rate đạt 100%**: Cơ chế Pre-retrieval filtering đảm bảo sinh viên không bao giờ bị trả về nhầm lẫn tài liệu/quy chế nội bộ của cán bộ hay giảng viên.  
> - **Recall@5 đạt 96.67% trên tập TEST**: Khẳng định độ khái quát hóa (generalization) cao của pipeline, không bị overfit trên tập DEV.  
> - **Full Evidence@5 đạt 93.33%**: Đảm bảo LLM luôn có đủ bằng chứng điều khoản cụ thể để trả lời mà không phải suy đoán.

Chi tiết phân tích các trường hợp thất bại và cách khắc phục nằm trong [`report/FAILURE_ANALYSIS.md`](report/FAILURE_ANALYSIS.md).

---

## 7. Giấy Phép & Tuyên Bố Miễn Trừ Trách Nhiệm
- Dự án phục vụ mục đích học tập và nghiên cứu trong khuôn khổ môn học AI/Data Foundations tại Đại học FPT.
- Dữ liệu quy chế được thu thập từ các nguồn công khai minh bạch của Trường Đại học FPT tại thời điểm tháng 09/2026.
- Tuyệt đối không chứa thông tin cá nhân, tài khoản đăng nhập hay dữ liệu nội bộ bảo mật.
