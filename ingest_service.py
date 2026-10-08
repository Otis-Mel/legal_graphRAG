# ingest_service.py — IMPROVED VERSION
# Thay đổi chính:
#   1. ingest_all_pdfs(): check file đã tồn tại → SKIP (tiết kiệm cost)
#   2. force_reingest(): xóa + nạp lại 1 file cụ thể
#   3. list_status(): xem file nào đã ingest, file nào chưa
#   4. delete_document(): xóa 1 file khỏi database

import traceback
from pathlib import Path
from typing import Any

from config import PDF_DIR
from embedding import get_embedding
from llm_functions import extract_triplets
from neo4j_store import Neo4jStore
from pdf_loader import find_pdf_files, load_pdf_documents


class IngestService:
    def __init__(self, store: Neo4jStore) -> None:
        self.store = store

   

    def ingest_all_pdfs(self) -> dict[str, Any]:
        """
        Scan thư mục PDF, chỉ ingest file CHƯA có trong database.
        File đã tồn tại → skip (tiết kiệm embedding cost).
        
        Muốn nạp lại 1 file cụ thể → dùng force_reingest(filename).
        """
        pdf_files = find_pdf_files(PDF_DIR)

        if not pdf_files:
            return {
                "status": "warning",
                "message": f"Không có file PDF nào trong thư mục {PDF_DIR}",
                "processed_files": 0,
                "processed_chunks": 0,
                "skipped_files": [],
                "failed_files": [],
            }

        # ============================================================
        # CHECK FILE ĐÃ TỒN TẠI TRONG NEO4J
        # Query 1 lần duy nhất → O(1) lookup cho mỗi file
        # ============================================================
        existing_docs = self.store.get_ingested_documents()
        print(f"[INGEST] Database hiện có {len(existing_docs)} documents:")
        for name, cnt in existing_docs.items():
            print(f"  ✓ {name} ({cnt} chunks)")

        processed_files = 0
        processed_chunks = 0
        skipped_files: list[str] = []
        failed_files: list[dict[str, str]] = []

        for pdf_path in pdf_files:
            filename = pdf_path.name

            # ============================================================
            # SKIP nếu file đã tồn tại
            # ============================================================
            if filename in existing_docs:
                chunk_count = existing_docs[filename]
                print(f"\n[SKIP] {filename} — đã có {chunk_count} chunks trong DB")
                skipped_files.append(filename)
                continue

            # File mới → ingest
            print(f"\n[NEW] Đang nạp file mới: {filename}")
            try:
                new_chunks = self._ingest_single_file(pdf_path)
                processed_files += 1
                processed_chunks += new_chunks
                print(f"[DONE] {filename}: {new_chunks} chunks đã nạp ✓")

            except Exception as e:
                failed_files.append({"file": filename, "error": str(e)})
                print(f"[ERROR] Lỗi file {filename}: {e}")
                traceback.print_exc()

        # Summary
        msg_parts = []
        if processed_files:
            msg_parts.append(f"Nạp mới {processed_files} file ({processed_chunks} chunks)")
        if skipped_files:
            msg_parts.append(f"Bỏ qua {len(skipped_files)} file đã tồn tại")
        if failed_files:
            msg_parts.append(f"Lỗi {len(failed_files)} file")

        return {
            "status": "success" if processed_files > 0 or skipped_files else "error",
            "message": " | ".join(msg_parts) if msg_parts else "Không có file nào để xử lý",
            "processed_files": processed_files,
            "processed_chunks": processed_chunks,
            "skipped_files": skipped_files,
            "failed_files": failed_files,
        }

    # ════════════════════════════════════════════
    # 2. FORCE RE-INGEST — nạp lại 1 file cụ thể
    # ════════════════════════════════════════════

    def force_reingest(self, filename: str) -> dict[str, Any]:
        """
        Xóa file cũ khỏi database rồi nạp lại từ đầu.
        Dùng khi: PDF được cập nhật nội dung mới.
        
        Usage:
            svc.force_reingest("luatdansu.pdf")
        """
        pdf_path = PDF_DIR / filename
        if not pdf_path.exists():
            return {
                "status": "error",
                "message": f"File không tồn tại: {pdf_path}",
            }

        # Xóa data cũ
        deleted_chunks = self.store.delete_document(filename)
        print(f"[REINGEST] Đã xóa {deleted_chunks} chunks cũ của {filename}")

        # Nạp lại
        try:
            new_chunks = self._ingest_single_file(pdf_path)
            return {
                "status": "success",
                "message": f"Re-ingest {filename}: xóa {deleted_chunks} chunks cũ → nạp {new_chunks} chunks mới",
                "deleted_chunks": deleted_chunks,
                "new_chunks": new_chunks,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Lỗi khi re-ingest {filename}: {e}",
            }

    # ════════════════════════════════════════════
    # 3. XÓA 1 FILE khỏi database
    # ════════════════════════════════════════════

    def delete_document(self, filename: str) -> dict[str, Any]:
        """
        Xóa 1 document + chunks + entities liên quan.
        
        Usage:
            svc.delete_document("luatdansu.pdf")
        """
        deleted = self.store.delete_document(filename)
        return {
            "status": "success",
            "message": f"Đã xóa {filename} ({deleted} chunks)",
            "deleted_chunks": deleted,
        }

    # ════════════════════════════════════════════
    # 4. XEM TRẠNG THÁI — file nào đã nạp, file nào chưa
    # ════════════════════════════════════════════

    def list_status(self) -> dict[str, Any]:
        """
        Liệt kê tất cả PDF trong thư mục + trạng thái (đã nạp / chưa nạp).
        
        Usage:
            status = svc.list_status()
            for f in status["files"]:
                print(f"{f['status']} {f['name']} ({f['chunks']} chunks)")
        """
        pdf_files = find_pdf_files(PDF_DIR)
        existing_docs = self.store.get_ingested_documents()

        files = []
        for pdf_path in pdf_files:
            name = pdf_path.name
            if name in existing_docs:
                files.append({
                    "name": name,
                    "status": "✓ ingested",
                    "chunks": existing_docs[name],
                })
            else:
                files.append({
                    "name": name,
                    "status": "✗ chưa nạp",
                    "chunks": 0,
                })

        # File trong DB nhưng không còn trong thư mục (đã xóa PDF?)
        orphans = []
        pdf_names = {p.name for p in pdf_files}
        for name in existing_docs:
            if name not in pdf_names:
                orphans.append(name)

        return {
            "files": files,
            "total_in_folder": len(pdf_files),
            "total_ingested": len(existing_docs),
            "new_files": sum(1 for f in files if f["status"] == "✗ chưa nạp"),
            "orphan_documents": orphans,
        }

    # ════════════════════════════════════════════
    # INTERNAL — nạp 1 file (không check tồn tại)
    # ════════════════════════════════════════════

    def _ingest_single_file(self, pdf_path: Path) -> int:
        """Nạp 1 file PDF vào Neo4j. Trả về số chunks đã nạp."""
        filename = pdf_path.name
        chunks = load_pdf_documents(pdf_path)
        print(f"[INGEST] {filename}: {len(chunks)} chunks")

        ingested = 0
        for idx, chunk in enumerate(chunks, start=1):
            content = (chunk.get("content") or "").strip()
            if not content:
                continue

            # 1. Embedding
            emb = get_embedding(content, task_type="retrieval_document")

            # 2. Triplet extraction (20 chunks đầu)
            summary = ""
            triplets = []
            try:
                if idx <= 20:
                    extracted = extract_triplets(content)
                    summary = extracted.get("summary", "")
                    triplets = extracted.get("triplets", [])
                    triplets = [
                        t for t in triplets
                        if t.get("relation") and t["relation"].upper() != "LIEN_QUAN"
                    ]
            except Exception as e:
                print(f"[WARN] Bỏ qua triplet chunk {idx}: {e}")

            # 3. Lưu vào Neo4j
            try:
                self.store.ingest_chunk(
                    chunk=chunk,
                    embedding=emb,
                    summary=summary,
                    triplets=triplets,
                )
                ingested += 1
            except Exception as e:
                print(f"[ERROR] Không lưu chunk {idx}: {e}")
                traceback.print_exc()
                continue

            if idx % 10 == 0 or idx == len(chunks):
                print(f"[INGEST] {filename}: {idx}/{len(chunks)}")

        return ingested