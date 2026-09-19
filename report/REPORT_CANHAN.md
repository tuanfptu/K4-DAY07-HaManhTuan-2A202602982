# BÁO CÁO CÁ NHÂN — LAB 7: EMBEDDING & VECTOR STORE

**Họ và tên:** Hà Mạnh Tuân  
**MSSV:** 2A202602982  
**Lớp / Variant:** K4-L3A — FPTU Student Policy & Services  
**Ngày hoàn thành:** 19/09/2026  

---

## 1. Phạm vi bài làm cá nhân

Bài làm tuân theo đúng cấu trúc của Lab 7:

```text
├── README.md
├── exercises.md
├── main.py
├── src/
│   ├── chunking.py
│   ├── store.py
│   ├── agent.py
│   └── ...
├── data/
├── tests/
│   └── test_solution.py
├── report/
│   ├── REPORT_NHOM.md
│   └── REPORT_CANHAN.md
├── docs/
│   ├── EVALUATION.md
│   ├── INSTRUCTOR_GUIDE.md
│   └── SCORING.md
└── requirements.txt
```

Ba phần cốt lõi do cá nhân hoàn thiện là `src/chunking.py`, `src/store.py` và
`src/agent.py`. Ngoài yêu cầu bắt buộc, tôi xây dựng thêm Advanced RAG trong
`src/advanced_rag/`, nổi bật là **heading-aware hierarchical chunking kết hợp
parent-child retrieval**.

---

## 2. Khởi động — Warm-up (5 điểm)

### 2.1. Cosine similarity

Cosine similarity đo góc giữa hai vector:

$$
\operatorname{cosine}(A,B)=\frac{A\cdot B}{\|A\|\|B\|}
$$

Giá trị gần 1 cho biết hai vector cùng hướng, tức hai đoạn văn bản gần nhau về
ngữ nghĩa. Giá trị gần 0 cho biết chúng ít liên quan. Nếu một vector bằng 0,
hàm trong bài trả về 0 để tránh phép chia cho 0.

Ví dụ tương đồng cao:

- “Sinh viên không đạt học phần bắt buộc phải đăng ký học lại.”
- “Sinh viên trượt môn bắt buộc phải học lại học phần đó.”

Hai câu khác từ ngữ nhưng cùng diễn đạt quy định học lại. Ngược lại, câu hỏi về
FAP và câu mô tả giờ mở cửa thư viện có độ tương đồng thấp vì khác mục đích và
đối tượng.

Cosine similarity phù hợp với text embedding hơn khoảng cách Euclid vì nó tập
trung vào hướng ngữ nghĩa, ít bị ảnh hưởng bởi độ lớn vector hoặc độ dài câu.

### 2.2. Bài toán chunking

Với tài liệu 10.000 ký tự, `chunk_size = 500`, `overlap = 50`, bước dịch là 450.
Số chunk được tính như sau:

$$
\left\lceil\frac{10000-50}{500-50}\right\rceil
=\left\lceil\frac{9950}{450}\right\rceil=23
$$

Khi tăng overlap lên 100:

$$
\left\lceil\frac{10000-100}{500-100}\right\rceil
=\left\lceil\frac{9900}{400}\right\rceil=25
$$

Overlap lớn hơn tạo thêm chunk nhưng giảm nguy cơ mất điều kiện, con số hoặc câu
văn nằm tại ranh giới giữa hai chunk.

---

## 3. Hướng tiếp cận — My Approach (10 điểm)

### 3.1. `src/chunking.py`

- `SentenceChunker`: tách câu bằng biểu thức chính quy, nhóm theo số câu tối đa,
  xử lý văn bản rỗng và chuẩn hóa khoảng trắng.
- `RecursiveChunker`: chia theo thứ tự đoạn văn, dòng, câu, từ và ký tự; có điều
  kiện dừng để không đệ quy vô hạn.
- `compute_similarity`: tính cosine similarity và bảo vệ vector 0.
- `ChunkingStrategyComparator`: chạy Fixed-size, Sentence và Recursive trên cùng
  dữ liệu, báo số chunk và độ dài trung bình.
- `HeadingAwareChunker`: nhận diện Markdown heading và cấu trúc tiếng Việt như
  Chương, Điều, Khoản; giữ đường dẫn heading trong metadata.

### 3.2. Heading-aware hierarchical chunking và parent-child retrieval

Đây là phần mở rộng chính của bài làm:

1. Tài liệu được phân đoạn theo heading thay vì cắt máy móc giữa Điều/Khoản.
2. Mỗi parent là một section hoàn chỉnh, tối đa mục tiêu 1.800 ký tự.
3. Parent được chia thành child khoảng 450 ký tự, overlap 60 ký tự.
4. Child có `retrieval_content` chứa đường dẫn tiêu đề.
5. BM25 và Dense Retriever tìm kiếm child; RRF hợp nhất thứ hạng.
6. `ParentChildRetriever` gom trùng theo `parent_id` và trả về toàn bộ parent cho
   bước sinh câu trả lời.

```text
Markdown document
       ↓
Heading-aware parents
       ↓
Small searchable children
       ↓
BM25 + Dense → RRF
       ↓
Deduplicate parent_id
       ↓
Full parent context → Agent
```

Child nhỏ giúp truy xuất chính xác, còn parent đầy đủ giúp Agent không bỏ sót
điều kiện hoặc bằng chứng nằm ở các câu lân cận.

### 3.3. `src/store.py`

`EmbeddingStore` lưu `id`, `content`, `metadata` và embedding trong bộ nhớ.
`add_documents` tạo embedding; `search` nhúng truy vấn, tính cosine similarity,
sắp xếp giảm dần và trả top-k. `search_with_filter` lọc metadata trước khi xếp
hạng. `delete_document` xóa các chunk thuộc một `doc_id` và trả trạng thái.

### 3.4. `src/agent.py`

`KnowledgeBaseAgent.answer` thực hiện ba bước: truy xuất chunk, tạo context có số
thứ tự và nguồn, sau đó sinh câu trả lời với ràng buộc chỉ dùng context. Khi
không có kết quả, Agent báo không đủ dữ liệu thay vì suy đoán. Trong Advanced
RAG, context đưa vào Agent là parent section đã được mở rộng.

---

## 4. Hoàn thiện code và kiểm thử (30 điểm)

Tất cả yêu cầu cốt lõi trong `src/chunking.py`, `src/store.py` và `src/agent.py`
đã được hoàn thiện. Lệnh kiểm thử:

```powershell
python -m unittest discover -s tests -v
```

Kết quả cuối:

```text
Ran 80 tests in 0.016s

OK
```

Trong đó:

- Toàn bộ test bắt buộc trong `tests/test_solution.py` đều đạt.
- Test chunking nâng cao kiểm tra heading, Chương/Điều/Khoản, parent-child và
  contextual chunking.
- Test retrieval kiểm tra BM25, Dense, RRF, Hybrid, metadata filter, Multi-query,
  HyDE và parent expansion/deduplication.
- Test benchmark kiểm tra truy vấn nhiều nguồn, Full Evidence và answer criteria.

Kết quả kiểm định corpus:

```text
Total checks performed : 68
Passed checks          : 68
Failed checks          : 0
RESULT: ALL CHECKS PASSED
```

---

## 5. Dự đoán độ tương tự (5 điểm)

| # | Cặp câu tóm tắt | Dự đoán | Kết quả | Nhận xét |
|---:|---|:---:|---:|---|
| 1 | Không đạt học phần ↔ trượt môn phải học lại | Cao | 0,8857 | Cùng quy định học lại |
| 2 | Học phí Kỹ thuật phần mềm ↔ mức đóng tiền ngành phần mềm | Cao | 0,9124 | Đồng nghĩa và cùng con số |
| 3 | Điều kiện OJT ↔ thời gian tiếng Anh dự bị | Thấp | 0,6106 | Khác chủ đề nhưng cùng miền giáo dục |
| 4 | Xem thời khóa biểu trên FAP ↔ giờ mở cửa thư viện | Thấp | 0,5908 | Khác hệ thống và mục đích |
| 5 | Hai câu giống hệt về FPTU HCM | Cao nhất | 1,0000 | Vector giống nhau |

Các cặp khác nội dung vẫn có thể đạt khoảng 0,6 vì cùng miền đại học. Do đó hệ
thống không chỉ dùng Dense Retrieval mà kết hợp BM25, metadata filter và RRF.

---

## 6. Benchmark cá nhân — 5 câu hỏi, 7 metric (10 điểm)

Benchmark sử dụng `gold_queries.json` do người dùng cung cấp, gồm
`source_doc_ids`, `evidence`, `answer_criteria` và `expected_audience`.

Lệnh chạy tái lập offline:

```powershell
python scripts/run_advanced_eval.py `
  --benchmark-file "C:\Users\TUAN\Downloads\gold_queries.json" `
  --embedding mock --no-llm `
  --output ket_qua_gold_queries.txt
```

### 6.1. Kết quả tổng hợp

| Metric | Kết quả | Ý nghĩa |
|---|---:|---|
| Recall@1 | 50,00% | Q1 có hai nguồn nên Top-1 tối đa đóng góp 0,5 |
| Recall@5 | **100,00%** | Top-5 bao phủ toàn bộ tài liệu vàng của 5 câu |
| MRR | 0,7500 | Tài liệu liên quan đầu tiên thường ở vị trí cao |
| nDCG@5 | 0,8123 | Các nguồn liên quan được xếp hạng tương đối tốt |
| Full Evidence@5 | 60,00% | 3/5 câu có đủ mọi nhóm bằng chứng trong Top-5 |
| Faithfulness / Agent Accuracy | 63,33% | Tỷ lệ tiêu chí câu trả lời đạt ở fallback offline |
| Audience Match Rate | **100,00%** | Không truy xuất nhầm tài liệu faculty/staff |

Độ trễ trung bình là **2,1 ms** trong chế độ mock embedding trên corpus đã nạp;
đây không phải benchmark tốc độ API Gemini.

### 6.2. Kết quả từng câu

| ID | Nội dung | Top-1 | R@5 | MRR | Faithfulness | Nhận xét |
|---|---|---|---:|---:|---:|---|
| Q1 | Điều kiện đầy đủ tham gia OJT | `07-ojt-regulations` | 1,00 | 1,00 | 0,33 | Top-5 đủ hai nguồn, fallback chưa ghép đủ ba tiêu chí |
| Q2 | Học phí AI 2026 theo khu vực | `06-campus-facilities-hcm` | 1,00 | 0,50 | 1,00 | Tài liệu học phí đứng thứ hai nhưng câu trả lời đủ tiêu chí |
| Q3 | Hạn học bổng và GPA duy trì | `04-scholarship-faq` | 1,00 | 1,00 | 0,50 | Đúng Top-1 nhưng fallback thiếu một nhóm chi tiết |
| Q4 | Gửi/Xem đơn và điểm danh FAP | `02-fap-and-academic-procedures` | 1,00 | 1,00 | 1,00 | Truy xuất và câu trả lời đầy đủ |
| Q5 | Dịch vụ sinh viên, hotline, phòng | `10-departments-and-support-routing-hcm` | 1,00 | 0,25 | 0,33 | Nguồn chuẩn có trong Top-5 nhưng đứng thứ tư |

### 6.3. Hai metric bổ sung theo yêu cầu

**Full Evidence@5** chỉ bằng 1 khi Top-5 chứa ít nhất một phrase hợp lệ trong
mọi nhóm `evidence`. Hệ thống không cộng điểm một phần chỉ vì đúng tài liệu.

**Faithfulness / Agent Accuracy** là tỷ lệ nhóm `answer_criteria` được thỏa mãn;
một criterion chỉ đạt khi câu trả lời chứa đủ mọi `all_terms` của criterion đó.

**Audience Match Rate** là tỷ lệ Top-5 có audience bằng đối tượng mong đợi hoặc
`all`, giúp variant L3A không trả nhầm quy định dành cho faculty/staff.

---

## 7. Phân tích lỗi và bài học

Q5 có tài liệu đúng `05-student-services-hcm` ở vị trí thứ tư. Tài liệu
`10-departments-and-support-routing-hcm` được ưu tiên vì có mật độ các từ “liên
hệ”, “đơn vị”, “campus” cao hơn. Q1 là câu multi-hop: điều kiện tín chỉ và
Orientation nằm ở hai tài liệu khác nhau nên một số parent cùng tài liệu có thể
cạnh tranh vị trí Top-5.

Hướng cải thiện:

- Áp dụng diversity/MMR hoặc giới hạn số parent trên mỗi tài liệu.
- Tăng điểm exact match cho hotline, số phòng, học phí và ngày tháng.
- Decompose câu hỏi nhiều vế rồi hợp nhất kết quả bằng RRF.
- Rerank theo độ bao phủ các thực thể trong truy vấn.
- Dùng LLM có grounding để tổng hợp đủ các parent thay cho sentence fallback.

Bài học chính là Dense embedding mạnh với từ đồng nghĩa nhưng chưa đủ cho số
liệu và tên riêng. BM25 bổ sung exact match, metadata filter bảo vệ đúng đối
tượng, còn heading-aware parent-child chunking cân bằng độ chính xác truy xuất
và độ đầy đủ ngữ cảnh. Recall@5 và Audience Match 100% cho thấy pipeline tìm
đúng nguồn; Full Evidence 60% chỉ rõ phần cần tiếp tục cải thiện.

---

## 8. Tự đánh giá

| Tiêu chí | Minh chứng | Điểm tự đánh giá |
|---|---|:---:|
| Warm-up | Cosine similarity và phép tính chunking | 5/5 |
| My Approach | `chunking.py`, `store.py`, `agent.py` và phần mở rộng | 10/10 |
| Core Implementation | 80/80 tests, 68/68 corpus checks | 30/30 |
| Similarity Predictions | 5 cặp, kết quả và phản ánh | 5/5 |
| Competition Results | 5 câu, 7 metric và failure analysis | 10/10 |
| **Tổng phần cá nhân** |  | **60/60** |

---

## 9. Tệp minh chứng

- Cốt lõi: `src/chunking.py`, `src/store.py`, `src/agent.py`.
- Phần mở rộng: `src/advanced_rag/chunking/parent_child_chunker.py` và
  `src/advanced_rag/retrieval/parent_child_retriever.py`.
- Đánh giá: `scripts/run_advanced_eval.py`, `ket_qua_gold_queries.txt`.
- Kiểm thử: `tests/test_solution.py`, `tests/test_advanced_chunking.py`,
  `tests/test_retrieval.py`, `tests/test_gold_evaluation.py`.
