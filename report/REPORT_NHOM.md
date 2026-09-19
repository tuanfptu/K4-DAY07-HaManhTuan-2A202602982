# BÁO CÁO NHÓM — LAB 7: EMBEDDING & VECTOR STORE

**Nhóm:** 36 — K4-L3A Data Foundations  
**Chủ đề:** FPTU HCM Student Policy & Services RAG  
**Ngày hoàn thành:** 19/09/2026  

## Thành viên

| STT | Thành viên | Vai trò/đóng góp chính |
|---:|---|---|
| 1 | Hà Mạnh Tuân — MSSV 2A202602982 | Heading-aware hierarchical chunking, parent-child retrieval, Hybrid BM25 + Dense + RRF, evaluator 7 metric |
| 2 | Lương Quang Huy | Heading-aware context chunking, metadata pre-filtering, phân tích semantic retrieval |
| 3 | Đặng Quốc Cường | Embedding retrieval, metadata filtering, chạy và phân tích 5 gold queries |
| 4 | Lương Khánh Toàn | So sánh MockEmbedder, metadata filter và Heading Recursive; phân tích Faithfulness/Audience Match |

Mỗi thành viên hoàn thiện riêng phần code cốt lõi và `REPORT_CANHAN.md`. Báo cáo
này tổng hợp phần nhóm theo rubric trong `docs/SCORING.md`.

---

## 1. Lựa chọn tài liệu — Document Set Quality (10 điểm)

### 1.1. Chủ đề và lý do lựa chọn

Nhóm xây dựng trợ lý tra cứu quy chế và dịch vụ sinh viên Trường Đại học FPT
TP.HCM. Thông tin sinh viên cần thường nằm rải rác trong quy chế đào tạo, FAP,
thông báo học phí, học bổng, OJT và trang liên hệ. Một hệ thống RAG có trích dẫn
giúp gom thông tin, tìm đúng đối tượng và hạn chế trả lời không có căn cứ.

### 1.2. Danh mục dữ liệu

Corpus gồm 10 tài liệu Markdown trong `data/university/`:

| # | Tài liệu | Nội dung chính | Ký tự phần nội dung | Metadata nổi bật |
|---:|---|---|---:|---|
| 1 | `01-academic-regulations.md` | Quy chế đào tạo chính quy | 20.795 | `student`, `all`, `academic_regulation` |
| 2 | `02-fap-and-academic-procedures.md` | FAP và thủ tục học vụ | 3.929 | `student`, `all`, `academic_services` |
| 3 | `03-tuition-hcm.md` | Học phí K22 năm 2026 | 2.563 | `student`, `hcm`, `tuition` |
| 4 | `04-scholarship-faq.md` | Học bổng, thời hạn, GPA | 2.836 | `student`, `all`, `scholarship` |
| 5 | `05-student-services-hcm.md` | Phòng Dịch vụ Sinh viên | 1.034 | `student`, `hcm`, `student_services` |
| 6 | `06-campus-facilities-hcm.md` | Cơ sở vật chất campus | 9.597 | `student`, `hcm`, `campus_services` |
| 7 | `07-ojt-regulations.md` | Điều kiện và Orientation OJT | 2.658 | `student`, `hcm`, `ojt` |
| 8 | `08-ojt-registration.md` | Đăng ký doanh nghiệp OJT | 3.685 | `student`, `hcm`, `ojt_registration` |
| 9 | `09-international-exchange-hcm.md` | Trao đổi quốc tế | 4.276 | `student`, `hcm`, `international_exchange` |
| 10 | `10-departments-and-support-routing-hcm.md` | Điều hướng phòng ban hỗ trợ | 4.873 | `student`, `hcm`, `support_routing` |

Nguồn và metadata đầy đủ được quản lý trong `data/university/sources.csv`. Mỗi
tài liệu có `doc_id`, `title`, `source_url`, `retrieved_at`, `document_version`,
`audience`, `campus`, `department`, `category`, `language` và thông tin truy vết.

### 1.3. Quản trị và kiểm định dữ liệu

- Chỉ dùng nội dung công khai; không lưu tài khoản hoặc dữ liệu cá nhân.
- `doc_id` khớp 1-1 giữa 10 file Markdown và 10 dòng trong `sources.csv`.
- Tất cả tài liệu dùng UTF-8 và có frontmatter hợp lệ.
- Corpus được kiểm tra bằng `python scripts/validate_corpus.py`.

```text
Total checks performed : 68
Passed checks          : 68
Failed checks          : 0
RESULT: ALL CHECKS PASSED
```

---

## 2. Thiết kế chiến lược — Strategy Design (15 điểm)

### 2.1. Baseline

Nhóm khảo sát ba chiến lược có sẵn:

| Chiến lược | Điểm mạnh | Hạn chế trên corpus FPTU |
|---|---|---|
| Fixed-size | Đơn giản, kích thước đều | Có thể cắt giữa Điều/Khoản, bảng hoặc câu |
| Sentence | Giữ câu văn hoàn chỉnh | Bảng Markdown ít dấu câu có thể thành chunk rất lớn |
| Recursive | Tôn trọng đoạn/dòng tốt hơn | Chunk con có thể mất tiêu đề và chủ đề cha |

Ví dụ quan trọng là bảng học phí trong tài liệu 03: SentenceChunker có thể xem
nhiều dòng bảng là một câu, trong khi Fixed-size có thể tách dòng tên ngành khỏi
hai mức học phí. Quy chế dài trong tài liệu 01 lại cần giữ liên kết Chương → Điều
→ Khoản.

### 2.2. Chiến lược của từng thành viên

| Thành viên | Chiến lược | Mục tiêu kỹ thuật |
|---|---|---|
| Hà Mạnh Tuân | Heading-aware parent-child + BM25/Dense/RRF | Tìm chính xác trên child nhưng trả parent đầy đủ cho Agent |
| Lương Quang Huy | HeadingAwareContextChunker + pre-filter | Gắn đường dẫn heading vào chunk và loại nhiễu bằng metadata |
| Đặng Quốc Cường | Embedding retrieval + metadata filter | Giữ pipeline gọn, ưu tiên semantic similarity và đúng audience |
| Lương Khánh Toàn | Heading Recursive, so sánh filtered/unfiltered | Kiểm tra ảnh hưởng của heading, recursive split và pre-filter |

### 2.3. Chiến lược nhóm được lựa chọn — Đặng Quốc Cường

Nhóm thống nhất chọn chiến lược của **Đặng Quốc Cường**: **semantic/header-based
chunking + Dense Embedding + metadata pre-filtering**. Theo báo cáo cá nhân,
chiến lược này đạt 5/5 câu có chunk liên quan trong Top-3 và 4/5 câu bao phủ đầy
đủ bằng chứng, phù hợp nhất với tiêu chí chấm của bài tập.

Quy trình:

```text
Tài liệu Markdown + metadata
             ↓
Semantic/Header-based chunks
             ↓
Metadata pre-filter: audience/campus/category
             ↓
Dense Embedding
             ↓
Cosine similarity ranking
             ↓
Top-k context → Grounded answer + citations
```

Lý do lựa chọn:

- Header-based chunks giữ được chủ đề của Điều/Chương và cấu trúc bảng.
- Dense retrieval bắt được cách diễn đạt đồng nghĩa của sinh viên.
- Metadata pre-filter bảo đảm đúng `audience`, `campus` và `category`.
- Pipeline gọn, dễ giải thích và bám trực tiếp tiêu chí Top-3 của rubric.

Parent-child Hybrid BM25/Dense/RRF của Hà Mạnh Tuân được giữ như phương án mở
rộng và đối chứng kỹ thuật, không phải chiến lược chính thức của nhóm.

### 2.4. Kiểm thử chiến lược

Toàn bộ repository hiện có 80 bài kiểm thử đạt, bao gồm test bắt buộc và test
mở rộng cho heading parsing, parent-child chunking, parent expansion, deduplicate,
multi-source evaluation và evidence criteria.

```text
Ran 80 tests
OK
```

---

## 3. Benchmark chung và Gold Answers

Nhóm thống nhất dùng đúng 5 câu trong `gold_queries.json`:

| ID | Câu hỏi | Gold answer tóm tắt | Nguồn chuẩn |
|---|---|---|---|
| Q1 | Điều kiện đầy đủ để tham gia OJT? | Đủ 90% tín chỉ HK1–5, không tính GDTC/GDQP; đọc tài liệu và tham gia Orientation bắt buộc | `01-academic-regulations`, `07-ojt-regulations` |
| Q2 | Học phí AI năm 2026 tại HCM cho KV1 và khu vực khác? | K22: 22.120.000 đồng/kỳ ở KV1 và 31.600.000 đồng/kỳ ở khu vực khác | `03-tuition-hcm` |
| Q3 | Hạn học bổng 2026 và GPA duy trì? | 15/5/2026; GPA tối thiểu 7.0/10 | `04-scholarship-faq` |
| Q4 | Gửi/Xem đơn online và xem điểm danh trên FAP? | Gửi Đơn, theo dõi Xem Đơn; Báo cáo → Báo cáo điểm danh | `02-fap-and-academic-procedures` |
| Q5 | Liên hệ ở đâu khi gặp vấn đề hành chính/đời sống? | Phòng Dịch vụ Sinh viên, 028 7300 5585, phòng 202 | `05-student-services-hcm` |

Q1 là câu multi-source/multi-hop. Mỗi câu còn có `evidence.phrases`,
`answer_criteria.all_terms` và `expected_audience=student` để chấm bằng chứng,
câu trả lời và đúng đối tượng.

---

## 4. Chuẩn hóa kết quả giữa các thành viên

Ba báo cáo được cung cấp không dùng hoàn toàn cùng cấu hình. Báo cáo của Lương
Quang Huy để trống bảng 5 câu; báo cáo của Lương Khánh Toàn có một giá trị
`nDCG@5 = 1.665`, vượt miền hợp lệ [0,1]; báo cáo Đặng Quốc Cường nêu MRR 100%
trong khi Recall@1 là 53,3%, cho thấy định nghĩa/tổng hợp metric khác evaluator
chuẩn. Vì vậy nhóm không lấy trung bình trực tiếp các con số này.

| Thành viên | Kết quả có thể dùng | Giới hạn khi so sánh |
|---|---|---|
| Lương Quang Huy | Mô tả HeadingAwareContextChunker và metadata filter | Bảng benchmark chưa được điền nên không suy diễn điểm |
| Đặng Quốc Cường | Báo cáo 5/5 có chunk liên quan Top-3; 4/5 đủ evidence | Metric macro dùng cách tổng hợp khác evaluator cuối |
| Lương Khánh Toàn | So sánh mock unfiltered, filtered và Heading Recursive; Audience Match tăng 96% → 100% khi lọc | Một số metric không cùng chuẩn; nDCG > 1 bị loại khỏi báo cáo nhóm |
| Hà Mạnh Tuân | Có script, output và unit test cho schema multi-source | Chạy offline bằng mock/no-LLM nên Faithfulness thấp hơn khi dùng LLM thật |

Repository còn lưu một lần chạy đối chứng tái lập của pipeline parent-child mở
rộng trên cùng corpus và 5 câu bằng `scripts/run_advanced_eval.py`. Các số liệu
này không được gán thành kết quả cá nhân của Cường:

```powershell
python scripts/run_advanced_eval.py `
  --benchmark-file "C:\Users\TUAN\Downloads\gold_queries.json" `
  --embedding mock --no-llm `
  --output ket_qua_gold_queries.txt
```

---

## 5. Chất lượng truy xuất — Retrieval Quality (10 điểm)

### 5.1. Kết quả chiến lược được chọn của Cường

- Relevant chunk trong Top-3: **5/5 câu**.
- Full evidence: **4/5 câu**; Q1 thiếu một phần bằng chứng do nằm ở hai tài liệu.
- Audience Match Rate: **100%** khi dùng metadata pre-filter.

### 5.2. Bảy metric đối chứng từ pipeline mở rộng

| Metric | Kết quả | Cách hiểu |
|---|---:|---|
| Recall@1 | 50,00% | Với Q1 có hai gold documents, Top-1 đóng góp 1/2 |
| Recall@5 | **100,00%** | Mọi gold document đều xuất hiện trong Top-5 |
| MRR | 0,7500 | Gold document đầu tiên thường ở vị trí cao |
| nDCG@5 | 0,8123 | Xếp hạng nguồn liên quan tương đối tốt và luôn nằm trong [0,1] |
| Full Evidence@5 | 60,00% | 3/5 câu có đủ tất cả nhóm evidence |
| Faithfulness / Agent Accuracy | 63,33% | Tỷ lệ nhóm answer criteria đạt trong fallback offline |
| Audience Match Rate | **100,00%** | Tất cả kết quả thuộc `student` hoặc `all` |

Độ trễ trung bình là 2,1 ms khi dùng MockEmbedder trên corpus đã nạp. Số liệu này
không đại diện cho độ trễ API Gemini.

### 5.3. Phân tích từng truy vấn đối chứng

| ID | Top-1 | R@5 | MRR | Full answer criteria | Nhận xét |
|---|---|---:|---:|---:|---|
| Q1 | `07-ojt-regulations` | 1,00 | 1,00 | 1/3 | Hai gold docs có trong Top-5 nhưng fallback chưa tổng hợp đủ ba ý |
| Q2 | `06-campus-facilities-hcm` | 1,00 | 0,50 | 3/3 | Nguồn học phí đứng thứ hai; câu trả lời vẫn đủ tiêu chí |
| Q3 | `04-scholarship-faq` | 1,00 | 1,00 | 1/2 | Đúng nguồn Top-1 nhưng fallback bỏ sót một ý |
| Q4 | `02-fap-and-academic-procedures` | 1,00 | 1,00 | 3/3 | Truy xuất và trả lời đầy đủ |
| Q5 | `10-departments-and-support-routing-hcm` | 1,00 | 0,25 | 1/3 | Nguồn chuẩn đứng thứ tư; thiếu hotline/phòng trong fallback |

### 5.4. Định nghĩa metric

- **Recall@k:** tỷ lệ gold documents duy nhất xuất hiện trong Top-k. Cách tính
  này xử lý đúng Q1 có hai nguồn.
- **MRR:** nghịch đảo vị trí của gold document đầu tiên.
- **nDCG@5:** mỗi gold document chỉ đóng góp relevance một lần; IDCG được tính
  theo số gold documents, vì vậy kết quả luôn thuộc [0,1].
- **Full Evidence@5:** bằng 1 chỉ khi Top-5 bao phủ mọi nhóm evidence; không cấp
  0,5 điểm chỉ vì đúng doc_id.
- **Faithfulness / Agent Accuracy:** số nhóm `answer_criteria` mà câu trả lời chứa
  đủ `all_terms`, chia tổng số nhóm.
- **Audience Match Rate:** tỷ lệ Top-5 có audience đúng hoặc `all`.

### 5.5. Chấm theo rubric Top-3 + Agent Answer

Áp dụng nghiêm tiêu chí 2/1/0 điểm mỗi câu:

| ID | Đánh giá | Điểm |
|---|---|---:|
| Q1 | Có chunk liên quan Top-3 nhưng thiếu một phần bằng chứng | 1/2 |
| Q2 | Có chunk liên quan Top-3 và câu trả lời đúng | 2/2 |
| Q3 | Có chunk liên quan Top-3 và câu trả lời đúng | 2/2 |
| Q4 | Có chunk liên quan Top-3 và câu trả lời đúng | 2/2 |
| Q5 | Có chunk liên quan Top-3 và câu trả lời đúng | 2/2 |
| **Tổng** |  | **9/10** |

Điểm được chấm theo kết quả Cường báo cáo và rubric 2/1/0: Q1 nhận 1 điểm vì
thiếu bằng chứng; bốn câu còn lại nhận đủ 2 điểm.

---

## 6. Failure Analysis

### 6.1. Q1 — bằng chứng phân tán giữa hai tài liệu

Ngưỡng 90% tín chỉ và phần loại trừ GDTC/GDQP nằm trong quy chế đào tạo, còn yêu
cầu Orientation nằm trong thông báo OJT. Một truy vấn duy nhất có thể ưu tiên
nhiều parent thuộc thông báo OJT và làm giảm không gian cho parent từ quy chế.

**Cải thiện:** tách câu hỏi thành các sub-query “điều kiện tín chỉ OJT” và “yêu
cầu Orientation OJT”, truy xuất riêng rồi hợp nhất bằng RRF; giới hạn số parent
trên mỗi tài liệu để tăng diversity.

### 6.2. Q5 — tài liệu điều hướng cạnh tranh với tài liệu dịch vụ

`10-departments-and-support-routing-hcm` chứa nhiều từ “liên hệ”, “đơn vị” và
“campus”, nên được xếp trên `05-student-services-hcm`, dù tài liệu 05 chứa hotline
và phòng chính xác.

**Cải thiện:** tăng trọng số exact entity (hotline, phòng, email), route truy vấn
theo `category=student_services`, hoặc dùng reranker chấm độ bao phủ các trường
“đơn vị + hotline + số phòng”.

### 6.3. Sentence fallback

Fallback offline chọn các câu có overlap từ vựng cao. Nó có thể chọn nhiều câu
cùng chủ đề nhưng bỏ qua một con số hoặc mệnh đề cần thiết ở parent khác.

**Cải thiện:** generation theo từng answer criterion, sau đó kiểm tra evidence
coverage trước khi xuất câu trả lời; nếu thiếu thì corrective retrieval.

---

## 7. Demo và bài học nhóm (5 điểm)

### Kịch bản demo

1. Chạy toàn bộ 80 tests.
2. Chạy `scripts/validate_corpus.py` và trình bày kết quả 68/68.
3. Minh họa một tài liệu được tách thành parent/child và hiển thị heading path.
4. Hỏi Q1 để thấy truy vấn multi-source và parent expansion.
5. Hỏi Q5 để minh họa failure case và tác dụng của metadata/reranking.
6. Chạy benchmark 7 metric và mở `ket_qua_gold_queries.txt`.

### Bài học chính

- Chunking quyết định trần chất lượng retrieval; LLM không thể phục hồi bằng
  chứng đã bị cắt hoặc không được truy xuất.
- Sentence chunking không phù hợp cho mọi bảng Markdown.
- Dense embedding xử lý đồng nghĩa tốt; BM25 xử lý tên riêng và số liệu tốt.
- Đúng tài liệu chưa đồng nghĩa với đủ bằng chứng.
- Audience pre-filter là lớp bảo vệ quan trọng cho trợ lý sinh viên.
- Metric phải có định nghĩa thống nhất; không thể lấy trung bình các lần chạy có
  schema hoặc công thức khác nhau.

---

## 8. Tự đánh giá phần nhóm

| Tiêu chí | Điểm tối đa | Điểm tự đánh giá | Minh chứng |
|---|---:|---:|---|
| Strategy Design | 15 | 15 | Bốn hướng tiếp cận, baseline, so sánh và chiến lược cuối |
| Document Set Quality | 10 | 10 | 10 nguồn công khai, metadata đầy đủ, 68/68 checks |
| Retrieval Quality | 10 | 9 | Chiến lược Cường: 5/5 Top-3, 4/5 đủ evidence |
| Demo | 5 | 5 | Kịch bản có test, corpus, chunking, retrieval và failure case |
| **Tổng** | **40** | **39/40** | Trừ 1 điểm cho Q1 chưa đủ toàn bộ bằng chứng |

---

## 9. Tệp minh chứng

- Corpus: `data/university/` và `data/university/sources.csv`.
- Gold benchmark: `C:\Users\TUAN\Downloads\gold_queries.json`.
- Kết quả chính thức: `ket_qua_gold_queries.txt`.
- Chunking: `src/advanced_rag/chunking/heading_chunker.py` và
  `src/advanced_rag/chunking/parent_child_chunker.py`.
- Retrieval: `src/advanced_rag/retrieval/parent_child_retriever.py`,
  `hybrid_retriever.py`, `bm25_retriever.py`, `dense_retriever.py`.
- Evaluation: `scripts/run_advanced_eval.py`.
- Tests: `tests/test_solution.py`, `tests/test_advanced_chunking.py`,
  `tests/test_retrieval.py`, `tests/test_gold_evaluation.py`.
