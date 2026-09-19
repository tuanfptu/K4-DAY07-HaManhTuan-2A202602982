# Báo Cáo Phân Tích Lỗi Truy Xuất (Retrieval Failure Analysis)
**Dự án:** FPT University HCM Student Policy & Services Assistant (K4-L3A)  
**Sinh viên:** Hà Mạnh Tuân  
**MSSV:** 2A202602982  
**Ngày thực hiện:** 19/09/2026  

---

## 1. Giới thiệu & Bối cảnh

Trong quá trình xây dựng và đánh giá hệ thống RAG cho bộ văn bản quy chế, dịch vụ sinh viên Đại học FPT phân hiệu TP. Hồ Chí Minh (`data/university/`), việc so sánh giữa các phương pháp truy xuất khác nhau (Baseline Fixed-Size vs Sentence vs Recursive vs Heading-Aware, Dense-only vs BM25 vs Hybrid RRF) đã bộc lộ rõ các điểm nghẽn (bottlenecks) và lỗi truy xuất đặc thù của tiếng Việt và văn bản hành chính/quy chuẩn.

Tài liệu này tập trung mổ xẻ chi tiết **2 ca thất bại thực tế (Real Retrieval Failure Cases)** trong hệ thống cơ sở (baseline), đồng thời phân tích nguyên nhân gốc rễ và chứng minh giải pháp trong kiến trúc Advanced RAG đã khắc phục triệt để như thế nào.

---

## 2. Ca Thất Bại 1: Lỗi Từ Vựng Đời Thường (Lexical Mismatch & Informal Query Failure)

### 2.1. Mô tả truy vấn và kết quả Baseline
* **Truy vấn gốc:** `"Nếu trượt môn bắt buộc thì phải làm gì?"`
* **Mục tiêu mong đợi (Gold Standard):** 
  * Tài liệu: `01-academic-regulations.md` (Quy chế đào tạo đại học chính quy)
  * Phần: **Điều 6, Khoản 3** (Quy định về việc sinh viên không đạt học phần bắt buộc thì phải đăng ký học lại học phần đó ở học kỳ tiếp theo khi học phần được mở).
* **Kết quả từ Dense Baseline (Mock / Generic Multilingual):**
  * Top-1: `06-campus-facilities-hcm.md` (Điểm tương đồng: 0.162) — *Nhiễu hoàn toàn (nói về phòng học, thư viện)*
  * Top-2: `04-scholarship-faq.md` (Điểm tương đồng: 0.141) — *Nói về duy trì học bổng*
  * Top-3: `08-ojt-registration.md` (Điểm tương đồng: 0.098) — *Nói về đăng ký doanh nghiệp*
  * **Hit@3: ✗ Thất bại (0/1)** — Không tìm thấy Điều 6 Quy chế đào tạo trong Top-3.

### 2.2. Phân tích nguyên nhân gốc rễ (Root Cause Analysis)
1. **Lệch pha từ vựng & phong cách ngôn ngữ (Lexical & Register Mismatch):**
   * Trong ngôn ngữ sinh viên hằng ngày, từ `"trượt môn"` rất phổ biến.
   * Tuy nhiên, trong văn bản pháp lý và quy chế của Đại học FPT, cụm từ chính xác được sử dụng là:
     > *"Sinh viên có học phần bắt buộc **không đạt** (điểm tổng kết học phần < 5.0 hoặc vi phạm điểm thành phần) phải đăng ký **học lại** học phần đó..."*
   * Từ `"trượt"` hoàn toàn **không xuất hiện** trong toàn bộ văn bản `01-academic-regulations.md`.
2. **Hạn chế của Sparse Retrieval đơn thuần (BM25 Failure):**
   * Do từ khóa `"trượt"` không khớp với `"không đạt"`, BM25 truyền thống nhận điểm số TF-IDF = 0 đối với các đoạn quan trọng của Điều 6.
3. **Chunking không nhận thức được ngữ cảnh tiêu đề (Heading Amnesia):**
   * Ở Fixed-Size Chunking (`chunk_size=500, overlap=50`), đoạn text của Khoản 3 bị cắt ngang, tách rời khỏi Tiêu đề lớn (`CHƯƠNG II: TỔ CHỨC ĐÀO TẠO`, `Điều 6. Kế hoạch học tập`). Chunk bị mất bối cảnh tài liệu dẫn đến độ tương đồng ngữ nghĩa bị phân tán.

### 2.3. Giải pháp trong Advanced RAG & Kết quả khắc phục
Hệ thống Advanced RAG xử lý lỗi này qua 3 tầng:

1. **Tầng Query Rewriter (`src/advanced_rag/routing/query_rewriter.py`):**
   * Nhận diện cụm từ informal: `"trượt môn"` → Chuẩn hóa thành thuật ngữ quy chuẩn: `"không đạt học phần bắt buộc, học lại học phần"`.
   * Truy vấn được viết lại thành:
     > *"Quy định về sinh viên không đạt học phần bắt buộc và đăng ký học lại tại Đại học FPT"*
2. **Tầng Contextual Prefix (`src/advanced_rag/chunking/heading_chunker.py`):**
   * Mỗi chunk được gán tiền tố ngữ cảnh:
     > `[SECTION] Document: QUY CHẾ ĐÀO TẠO ĐẠI HỌC CHÍNH QUY > CHƯƠNG II: TỔ CHỨC ĐÀO TẠO > Điều 6. Kế hoạch học tập`
   * Điều này giúp vector embedding của chunk mang đầy đủ ý nghĩa thực thể trường học và ngữ cảnh quy chế.
3. **Tầng Hybrid Fusion (BM25 + Dense + RRF):**
   * Sau khi query được rewrite, cả BM25 và Dense (Gemini) đều bắt chính xác các tokens `"học phần bắt buộc"`, `"không đạt"`, `"học lại"`.
   * **Kết quả sau khắc phục:**
     * Top-1: `01-academic-regulations.md` (Điều 6, Khoản 3) — **Hit@1 = ✓ (Chính xác tuyệt đối)**.

---

## 3. Ca Thất Bại 2: Xung Đột Phạm Vi Campus & Metadata (Cross-Campus Interference)

### 3.1. Mô tả truy vấn và kết quả Baseline
* **Truy vấn gốc:** `"Học phí ngành Kỹ thuật phần mềm một kỳ là bao nhiêu?"`
* **Mục tiêu mong đợi (Gold Standard):**
  * Tài liệu: `03-tuition-hcm.md` (Học phí campus TP. Hồ Chí Minh khóa K22: KV1 = 22.120.000 VNĐ, KV khác = 31.600.000 VNĐ/học kỳ).
* **Kết quả khi không có Metadata Filtering:**
  * Nếu hệ thống ingest nhiều văn bản thuộc các campus khác nhau hoặc tài liệu chung:
  * Top-1: Trả về thông tin học phí chung hoặc nhầm lẫn giữa học phí tiếng Anh dự bị và học phí chuyên ngành.
  * Trong các câu hỏi nhắm đến đối tượng cụ thể (ví dụ sinh viên vs cán bộ), việc thiếu lọc `audience="student"` dẫn đến kết quả trả về các quy định nội bộ không dành cho người học.

### 3.2. Phân tích nguyên nhân gốc rễ
* **Semantic Ambiguity (Mơ hồ ngữ nghĩa về địa phương hóa):**
  * Đại học FPT có nhiều campus (Hà Nội, TP.HCM, Đà Nẵng, Cần Thơ, Quy Nhơn) với mức học phí ưu đãi vùng miền (KV1, KV khác) khác nhau.
  * Nếu người dùng không chỉ định rõ chữ `"HCM"`, mô hình Dense vector có xu hướng thiên lệch về các đoạn văn bản có tần suất xuất hiện từ `"học phí"` dày đặc nhất, thay vì tài liệu có hiệu lực tại campus của người hỏi.
* **Filter Ignorance:**
  * Baseline RAG không hỗ trợ Pre-retrieval filtering theo metadata schema (`campus`, `audience`).

### 3.3. Giải pháp trong Advanced RAG
1. **Metadata Pre-filtering (`src/advanced_rag/retrieval/metadata_filter.py`):**
   * Hệ thống tự động trích xuất hoặc áp dụng ràng buộc:
     `metadata_filter = {"audience": "student", "campus": "hcm"}`
   * Loại bỏ 100% tài liệu không liên quan đến sinh viên HCM trước khi tiến hành tính toán vector.
2. **Khai thác cấu trúc bảng Markdown (`Table Preservation`):**
   * Bảng học phí dạng Markdown được giữ nguyên vẹn cấu trúc qua pipeline làm sạch (`cleaner.py`), không bị cắt ngang hàng, giúp embedding nắm trọn vẹn cặp dữ liệu: `"Kỹ thuật phần mềm | 22.120.000 | 31.600.000"`.

---

## 4. Tổng Hợp Bài Học Kiến Trúc (Architecture Lessons Learned)

| Vấn đề quan sát | Nguyên nhân kỹ thuật | Giải pháp kiến trúc đã triển khai |
|---|---|---|
| **Từ vựng sinh viên khác văn bản pháp quy** | Khoảng cách ngữ nghĩa giữa ngôn ngữ tự nhiên và từ ngữ hành chính | **Query Rewriting** (ánh xạ từ ngữ đời thường → thuật ngữ pháp lý) |
| **Chunk bị mất ngữ cảnh của Điều/Chương** | Fixed-size chunking cắt ngang cấu trúc logic văn bản | **HeadingAwareChunker** (bảo toàn phân cấp Tiêu đề > Chương > Điều > Khoản) |
| **Embedding đơn lẻ không đủ nhạy với số liệu/mã môn** | Dense models làm mờ các từ hiếm (mã môn, số hiệu, tiền tệ) | **Hybrid Retrieval (BM25 + Dense RRF)** kết hợp sức mạnh từ vựng và ngữ nghĩa |
| **Nhiễu thông tin giữa các đối tượng/cơ sở** | Thiếu cơ chế phân vùng dữ liệu theo đặc thù đối tượng | **Metadata-Aware Filtering** (`audience: student`, `campus: hcm`) |
| **Ảo giác khi thiếu bằng chứng** | LLM cố gắng suy đoán khi dữ liệu không đủ | **Evidence Verification & Abstention** (từ chối trả lời nếu thiếu căn cứ) |

---
*Báo cáo được đúc kết từ quá trình thực nghiệm trực tiếp trên tập dữ liệu Quy chế FPTU HCM.*
