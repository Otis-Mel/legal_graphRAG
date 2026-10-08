# rag_service.py
from __future__ import annotations

import re
from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from memory_service import MemoryService

from config import TOP_K
from embedding import get_embedding
from llm_functions import answer_with_context
from neo4j_store import Neo4jStore


class GraphRAGService:
    def __init__(self, store: Neo4jStore) -> None:
        self.store = store

    # ─────────────────────────────────────────────
    # TERM EXTRACTION
    # ─────────────────────────────────────────────

    def _extract_terms(self, text: str) -> list[str]:
        stopwords = {
            "là", "gì", "về", "và", "của", "cho", "có", "theo", "được",
            "trong", "khi", "với", "hay", "để", "các", "những", "một",
            "nào", "ra", "sao", "bao", "nhiêu", "tại", "đến", "từ",
            "thì", "mà", "bị", "do", "trên", "dưới", "này", "đó",
        }
        cleaned = re.sub(r"[^\wÀ-ỹ]", " ", text)
        terms = []
        for t in cleaned.split():
            v = t.lower().strip()
            if len(v) >= 2 and v not in stopwords and v not in terms:
                terms.append(v)
        return terms

    # ─────────────────────────────────────────────
    # RANKING & FILTER
    # ─────────────────────────────────────────────

    def _score_chunk(self, chunk: dict, terms: list[str]) -> float:
        content = (chunk.get("content") or "").lower()
        return sum(3.0 for t in terms if t in content)

    def _rank_chunks(self, question: str, chunks: list[dict]) -> list[dict]:
        terms = self._extract_terms(question)
        return sorted(
            chunks,
            key=lambda c: float(c.get("score", 0)) * 20 + self._score_chunk(c, terms),
            reverse=True,
        )

    def _filter_chunks(self, question: str, chunks: list[dict]) -> list[dict]:
        terms = self._extract_terms(question)
        if not terms:
            return chunks
        filtered = [
            c for c in chunks
            if any(t in (c.get("content") or "").lower() for t in terms)
        ]
        return filtered if filtered else chunks

    # ─────────────────────────────────────────────
    # CONTEXT BUILDER
    # ─────────────────────────────────────────────

    def _build_context(self, chunks: List[dict], max_chunks: int = 5) -> str:
        """
        Build context string từ chunks, đưa metadata Điều/Khoản vào header
        để LLM tạo citation chính xác.
        """
        parts = []
        for i, c in enumerate(chunks[:max_chunks], start=1):
            content = (c.get("content") or "").strip()
            if len(content) < 30:
                continue

            legal_refs = [
                c[k] for k in ("dieu", "khoan", "chuong")
                if c.get(k)
            ]
            legal_str = " | ".join(legal_refs) if legal_refs else ""

            header = (
                f"[NGUỒN {i}] "
                f"File: {c.get('source', 'unknown')} | "
                f"Trang: {c.get('page', '?')}"
            )
            if legal_str:
                header += f" | {legal_str}"

            parts.append(f"{header}\n{content[:1200]}")

        return "\n\n".join(parts)

    # ─────────────────────────────────────────────
    # GRAPH
    # ─────────────────────────────────────────────

    def _build_answer_graph(self, question: str) -> dict[str, Any]:
        try:
            graph = self.store.graph_search(question, top_k=80)
            if graph.get("nodes"):
                return graph
            return self.store.get_graph_data(limit=120)
        except Exception as e:
            print(f"[GRAPH] Error: {e}")
            return self.store.get_graph_data(limit=120)

    # ─────────────────────────────────────────────
    # QUERY EXPANSION
    # ─────────────────────────────────────────────

    def _expand_query(self, question: str) -> List[str]:
        """
        Dùng LLM viết lại câu hỏi thành 2-3 cách diễn đạt khác nhau.
        Mục đích: tăng recall khi vector search bằng cách cover nhiều
        cách phrasing khác nhau của cùng một câu hỏi pháp lý.
        """
        prompt = f"""Bạn là chuyên gia pháp luật Việt Nam.
Hãy viết lại câu hỏi dưới đây thành 2 cách diễn đạt khác nhau.
Mỗi cách dùng từ ngữ pháp lý khác nhau nhưng cùng ý nghĩa.
Chỉ trả về 2 câu, mỗi câu một dòng, không đánh số, không giải thích.

Câu hỏi gốc: {question}"""

        try:
            from llm_functions import _generate_with_fallback
            raw = _generate_with_fallback(
                prompt,
                temperature=0.3,
                max_output_tokens=200,
            )
            expansions = [
                line.strip() for line in raw.strip().split("\n")
                if line.strip() and len(line.strip()) > 10
            ][:2]  # lấy tối đa 2 câu
            print(f"[QE] Expanded: {expansions}")
            return [question] + expansions
        except Exception as e:
            print(f"[QE] Failed: {e}")
            return [question]  # fallback về câu gốc

    def _rrf_merge(self, ranked_lists: List[List[dict]], k: int = 60) -> List[dict]:
        """
        Reciprocal Rank Fusion: merge nhiều ranked list thành 1.
        Score = sum(1 / (k + rank_i)) cho mỗi chunk_id.
        Chuẩn trong production RAG.
        """
        scores: dict[str, float] = {}
        chunk_map: dict[str, dict] = {}

        for ranked in ranked_lists:
            for rank, chunk in enumerate(ranked, start=1):
                cid = chunk.get("chunk_id", "")
                if not cid:
                    continue
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
                if cid not in chunk_map:
                    chunk_map[cid] = chunk

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        merged = []
        for cid in sorted_ids:
            chunk = chunk_map[cid].copy()
            chunk["rrf_score"] = round(scores[cid], 6)
            merged.append(chunk)
        return merged

    # ─────────────────────────────────────────────
    # CHUNK RETRIEVAL
    # ─────────────────────────────────────────────

    def _get_chunks_for_question(self, question: str) -> List[dict]:
        """
        Hybrid search với Query Expansion + RRF:
        1. Viết lại câu hỏi thành 2-3 cách (query expansion)
        2. Vector search cho từng cách diễn đạt
        3. Merge kết quả bằng RRF (Reciprocal Rank Fusion)
        4. Bổ sung keyword search nếu thiếu
        """
        # Bước 1: Query expansion
        queries = self._expand_query(question)

        # Bước 2: Vector search cho từng query
        all_ranked_lists = []
        existing_ids: set[str] = set()

        for q in queries:
            try:
                emb = get_embedding(q, task_type="retrieval_query")
                results = self.store.vector_search(emb, top_k=TOP_K + 3)
                if results:
                    all_ranked_lists.append(results)
                    print(f"[RAG] Query '{q[:40]}...' → {len(results)} chunks")
            except Exception as e:
                print(f"[WARN] Vector search failed for '{q[:30]}': {e}")

        # Bước 3: RRF merge
        if all_ranked_lists:
            chunks = self._rrf_merge(all_ranked_lists)
            print(f"[RRF] Merged → {len(chunks)} unique chunks")
        else:
            chunks = []

        # Bước 4: Keyword search bổ sung nếu ít kết quả
        if len(chunks) < 3:
            try:
                kw_chunks = self.store.keyword_search(question, limit=TOP_K)
                existing_ids = {c.get("chunk_id") for c in chunks}
                added = 0
                for kc in kw_chunks:
                    if kc.get("chunk_id") not in existing_ids:
                        chunks.append(kc)
                        added += 1
                print(f"[RAG] Keyword search bổ sung: {added} chunks")
            except Exception as e:
                print(f"[WARN] Keyword search failed: {e}")

        if chunks:
            chunks = self._rank_chunks(question, chunks)
            chunks = self._filter_chunks(question, chunks)

        return chunks

    # ─────────────────────────────────────────────
    # MAIN ASK
    # ─────────────────────────────────────────────

    def ask(self, question: str, memory: Optional[MemoryService] = None) -> dict[str, Any]:
        question = (question or "").strip()
        if not question:
            return {
                "answer": "Vui lòng nhập câu hỏi.",
                "chunks": [],
                "graph_facts": [],
                "answer_graph": {"nodes": [], "edges": []},
                "has_pdf_data": False,
            }

        # 1. Lấy chunks
        chunks = self._get_chunks_for_question(question)

        # 2. Graph visualization
        answer_graph = self._build_answer_graph(question)

        # 3. Build context string (bao gồm metadata Điều/Khoản)
        context = self._build_context(chunks, max_chunks=5)

        # 4. Gọi LLM — nhận tuple (answer, has_data)
        try:
            result = answer_with_context(
                question=question,
                context=context,
                graph_facts=[],
                memory=memory,
                chunks=chunks,
            )
            # answer_with_context trả về tuple (answer, has_data)
            if isinstance(result, tuple):
                answer, has_pdf_data = result
            else:
                answer, has_pdf_data = result, bool(chunks)
        except Exception as e:
            print(f"[ERROR] RAG ask error: {e}")
            answer = "Có lỗi xảy ra khi xử lý câu hỏi."
            has_pdf_data = False

        return {
            "answer": answer,
            # Chỉ trả chunks về frontend khi thực sự dùng PDF
            "chunks": chunks[:5] if has_pdf_data else [],
            "graph_facts": [],
            "answer_graph": answer_graph,
            "has_pdf_data": has_pdf_data,
        }