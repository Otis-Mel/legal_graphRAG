# Legal GraphRAG với Neo4j + Gemini + LangChain

## 1) Chuẩn bị môi trường trên Mac + VSCode

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Mở file `.env` và điền `GEMINI_API_KEY`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`.

## 2) Thêm PDF

Chép các file PDF pháp luật vào thư mục `pdfs/`.

## 3) Chạy ứng dụng

```bash
python app.py
```

Mở trình duyệt tại:

```text
http://127.0.0.1:5000
```

## 4) Cách hoạt động

- PDF được nạp bằng LangChain `PyPDFLoader`
- Văn bản được chia chunk bằng `RecursiveCharacterTextSplitter`
- Mỗi chunk được tạo embedding bằng Gemini
- Chunk được lưu vào Neo4j với vector index
- Gemini trích xuất triplet để tạo `Entity` và quan hệ `RELATES`
- Khi hỏi đáp, hệ thống kết hợp:
  - Vector Retrieval trên `Chunk`
  - Graph lookup trên `Entity` / `RELATES`
  - Gemini tổng hợp câu trả lời có tham chiếu nguồn

## 5) Neo4j

Bạn cần chạy Neo4j trước khi start app.
Nếu dùng Neo4j Desktop, hãy đảm bảo DB đang ở trạng thái Running.

## 6) Gợi ý demo với thầy

- Giới thiệu graph schema: `Document -> Chunk -> Entity -> RELATES`
- Demo ingest 1-2 PDF luật
- Đặt câu hỏi về điều kiện, hành vi bị cấm, xử phạt, đối tượng áp dụng
- Mở Neo4j Browser để cho thấy các node `Chunk`, `Entity`, và cạnh `RELATES`
