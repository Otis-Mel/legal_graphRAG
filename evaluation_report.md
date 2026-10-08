# Báo cáo Evaluation — Legal GraphRAG

**Thời gian:** 2026-06-14T11:10:12.376433  
**Câu hỏi:** 70 (70 trong tài liệu, 0 ngoài domain)

---

## 1. Retrieval Metrics

| Metric | Kết quả | Ý nghĩa |
|--------|---------|---------|
| Precision@1 | **0.9857** | Chunk đầu tiên đúng file |
| Precision@3 | **0.9429** | Top-3 có chunk đúng |
| Precision@5 | **0.9343** | Top-5 có chunk đúng |
| MRR | **0.9929** | Vị trí trung bình chunk đúng |

## 2. Answer Quality Metrics

| Metric | Kết quả | Phương pháp | Ý nghĩa |
|--------|---------|-------------|---------|
| BERTScore F1 | **0.8865** | XLM-RoBERTa-large | Tương đồng ngữ nghĩa (nhận biết paraphrase) |
| Answer Relevancy | **0.9615** | LLM + BERTScore | Trả lời đúng trọng tâm câu hỏi |

## 3. Hallucination Control (LLM-as-judge, claim-level)

| Metric | Kết quả | Ý nghĩa |
|--------|---------|---------|
| Avg Score | **0.5296** | Tỷ lệ claims được hỗ trợ bởi context (cao = tốt) |
| PASS rate | **20.0%** | Câu trả lời có ≥85% claims OK |
| PARTIAL rate | **42.9%** | Câu trả lời có 50–84% claims OK |
| FAIL rate | **37.1%** | Câu trả lời có <50% claims OK (hallucinate nặng) |
| Phân bổ verdict | **PASS=14 / PARTIAL=30 / FAIL=26** | Số câu theo từng mức |

> **Phương pháp:** LLM phân tách answer thành các mệnh đề pháp lý độc lập (claim extraction), sau đó verify từng mệnh đề với context PDF được retrieve. Phân loại: **SUPPORTED** (có trong context), **UNSUPPORTED** (bịa đặt/sai), **GENERAL** (kiến thức chung hợp lệ). Score = tỷ lệ SUPPORTED + GENERAL trên tổng claims.

## 4. Kết quả theo Domain

| Domain | P@3 | BERTScore | AR | Hallucination | Câu |
|--------|-----|-----------|-----|---------------|-----|
| Luật Dân sự | 0.97 | 0.90 | 0.97 | 0.60 | 10 |
| Luật An toàn thực phẩm | 1.00 | 0.89 | 0.96 | 0.73 | 10 |
| Luật Hôn nhân Gia đình | 0.97 | 0.88 | 0.96 | 0.50 | 10 |
| Sở hữu trí tuệ | 1.00 | 0.88 | 0.95 | 0.52 | 10 |
| Luật Đất đai | 0.97 | 0.88 | 0.99 | 0.44 | 10 |
| Luật Giao thông | 0.73 | 0.89 | 0.96 | 0.41 | 10 |
| Luật Thuế TNCN | 0.97 | 0.88 | 0.96 | 0.51 | 10 |

## 5. Kết quả theo Độ khó

| Độ khó | P@3 | BERTScore | AR |
|--------|-----|-----------|-----|
| easy | 0.94 | 0.91 | 0.95 |
| medium | 0.95 | 0.88 | 0.97 |
| hard | 0.94 | 0.88 | 0.96 |

## 6. Câu trả lời cần cải thiện (hallucination cao nhất)

**attp_04** — Giấy chứng nhận cơ sở đủ điều kiện an toàn thực phẩm có thời hạn bao l...
- Verdict: **FAIL** (score=0.00, ?/? claims OK)
  - ⚠ `Giấy chứng nhận cơ sở đủ điều kiện an toàn thực phẩm đạt yêu cầu Thực hành sản xuất tốt (GMP) thực phẩm bảo vệ`

**hn_01** — Điều kiện về độ tuổi kết hôn theo luật hôn nhân gia đình là bao nhiêu?...
- Verdict: **FAIL** (score=0.00, ?/? claims OK)
  - ⚠ `Nam từ đủ 20 tuổi trở lên mới được kết hôn.`
  - ⚠ `Nữ từ đủ 18 tuổi trở lên mới được kết hôn.`
  - ⚠ `Điều kiện về độ tuổi kết hôn được quy định tại Điều 8.`

**hn_02** — Khi ly hôn tòa án giải quyết vấn đề con chung dựa trên nguyên tắc nào?...
- Verdict: **FAIL** (score=0.00, ?/? claims OK)
  - ⚠ `Tòa án sẽ công nhận thuận tình ly hôn nếu thỏa thuận về việc trông nom, nuôi dưỡng, chăm sóc, giáo dục con bảo`
  - ⚠ `Nếu không thỏa thuận được hoặc có thỏa thuận nhưng không bảo đảm quyền lợi chính đáng của vợ và con thì Tòa án`
  - ⚠ `Nguồn: luathonnhanvagiadinh.pdf, trang 18, Điều 55`

**shtt_06** — Xâm phạm quyền bảo vệ sự toàn vẹn của tác phẩm là hành vi gì?...
- Verdict: **FAIL** (score=0.00, ?/? claims OK)
  - ⚠ `Xâm phạm quyền bảo vệ sự toàn vẹn của tác phẩm là hành vi xuyên tạc tác phẩm.`
  - ⚠ `Xâm phạm quyền bảo vệ sự toàn vẹn của tác phẩm là hành vi sửa đổi tác phẩm gây phương hại đến danh dự và uy tí`
  - ⚠ `Xâm phạm quyền bảo vệ sự toàn vẹn của tác phẩm là hành vi cắt xén tác phẩm gây phương hại đến danh dự và uy tí`

**shtt_07** — Trích dẫn hợp lý tác phẩm phải đáp ứng điều kiện gì?...
- Verdict: **FAIL** (score=0.00, ?/? claims OK)
  - ⚠ `Phần trích dẫn chỉ nhằm mục đích giới thiệu, bình luận hoặc làm sáng tỏ vấn đề được đề cập trong tác phẩm của `
  - ⚠ `Phần trích dẫn từ tác phẩm được sử dụng để trích dẫn không gây thiệt hại một cách bất hợp lý đến lợi ích hợp p`
  - ⚠ `Phần trích dẫn từ tác phẩm được sử dụng để trích dẫn phù hợp với tính chất, đặc điểm của loại hình tác phẩm đư`


## 7. Giải thích metrics

- **Precision@K**: Tỷ lệ top-K chunks retrieved có đúng file nguồn
- **MRR**: Mean Reciprocal Rank — chunk đúng ở vị trí nào trong kết quả search
- **BERTScore**: Dùng XLM-RoBERTa-large embedding so sánh ngữ nghĩa — nhận biết paraphrase, từ đồng nghĩa tiếng Việt
- **Answer Relevancy**: Sinh lại câu hỏi từ câu trả lời, đo BERTScore với câu hỏi gốc
- **Hallucination Score**: LLM tách answer thành claims → verify từng claim với context → score = tỷ lệ claims được hỗ trợ (SUPPORTED hoặc GENERAL)