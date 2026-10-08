# memory_service.py
from __future__ import annotations

from typing import List, Dict, Optional
from datetime import datetime
import threading


class MemoryService:
    """
    Two-Tier Memory Architecture cho Legal GraphRAG Chatbot.

    Tầng 1 — Short-term Buffer (RAM):
        - Giữ N cặp Q&A gần nhất trong RAM
        - Mục đích: ngữ cảnh hội thoại liên tục (conversation continuity)
        - Bị xóa khi restart app (ephemeral)
        - Capped tại `short_term_limit` (mặc định 30)

    Tầng 2 — Long-term Memory (Neo4j):
        a) Memory nodes: Mọi cặp Q&A được lưu vĩnh viễn vào Neo4j
           → KHÔNG giới hạn số lượng, KHÔNG xóa node cũ
           → Vector search tìm memory liên quan (semantic retrieval)
        b) UserFact nodes: Sau mỗi hội thoại, LLM trích xuất các sự kiện/
           thông tin quan trọng → lưu vĩnh viễn, deduplicate tự động
           → Ví dụ: "User quan tâm đến luật đất đai", "User hỏi nhiều về thuế TNCN"

    Khi cần context:
        - Short-term: lấy 3-5 Q&A gần nhất từ RAM (recency)
        - Long-term:  vector search Memory + UserFact liên quan đến câu hỏi hiện tại
        - Kết hợp cả hai để inject vào system prompt
    """

    def __init__(self, short_term_limit: int = 30, store=None):
        self.short_term_limit = short_term_limit
        self.store = store                  # Neo4jStore instance
        self._short_term: List[Dict] = []   # Tầng 1: RAM buffer

        if self.store:
            self._setup_memory_schema()

    # ══════════════════════════════════════════════
    # SCHEMA SETUP
    # ══════════════════════════════════════════════

    def _setup_memory_schema(self):
        """Tạo vector index cho Memory và UserFact nodes."""
        try:
            with self.store.driver.session() as session:
                # Index cho Memory nodes (Q&A pairs)
                session.run(
                    "CREATE VECTOR INDEX memory_embeddings IF NOT EXISTS "
                    "FOR (m:Memory) ON m.embedding "
                    "OPTIONS { indexConfig: { "
                    "`vector.dimensions`: 3072, "
                    "`vector.similarity_function`: 'cosine' "
                    "} }"
                )
                # Index cho UserFact nodes (extracted facts)
                session.run(
                    "CREATE VECTOR INDEX userfact_embeddings IF NOT EXISTS "
                    "FOR (f:UserFact) ON f.embedding "
                    "OPTIONS { indexConfig: { "
                    "`vector.dimensions`: 3072, "
                    "`vector.similarity_function`: 'cosine' "
                    "} }"
                )
                print("[MEMORY] Vector indexes ready (Memory + UserFact)")
        except Exception as e:
            print(f"[MEMORY] Vector index setup skipped: {e}")

    # ══════════════════════════════════════════════
    # TẦNG 1: SHORT-TERM BUFFER (RAM)
    # ══════════════════════════════════════════════

    def _add_short_term(self, question: str, answer: str):
        """Thêm vào RAM buffer, cap tại short_term_limit."""
        self._short_term.append({
            "question": question,
            "answer": answer,
            "created_at": datetime.utcnow().isoformat(),
        })
        if len(self._short_term) > self.short_term_limit:
            self._short_term = self._short_term[-self.short_term_limit:]

    def get_short_term_context(self, n: int = 3) -> str:
        """Lấy N cặp Q&A gần nhất từ RAM."""
        recent = self._short_term[-n:]
        if not recent:
            return ""
        lines = []
        for i, m in enumerate(recent, start=1):
            q = m.get("question", "").strip()
            a = m.get("answer", "").strip()
            if q and a:
                a_short = a[:300] + "..." if len(a) > 300 else a
                lines.append(f"[Gần đây {i}] Hỏi: {q}\nTrả lời: {a_short}")
        return "\n\n".join(lines)

    # ══════════════════════════════════════════════
    # TẦNG 2a: LONG-TERM MEMORY (Neo4j - Memory nodes)
    # ══════════════════════════════════════════════

    def _save_memory_node(self, question: str, answer: str):
        """
        Lưu cặp Q&A thành Memory node trong Neo4j.
        KHÔNG giới hạn số lượng — đây là long-term, lưu vĩnh viễn.
        """
        try:
            from embedding import get_embedding
            emb = get_embedding(question, task_type="retrieval_document")

            with self.store.driver.session() as session:
                session.run(
                    """
                    CREATE (m:Memory {
                        question:   $question,
                        answer:     $answer,
                        created_at: $created_at,
                        embedding:  $embedding
                    })
                    """,
                    question=question.strip(),
                    answer=answer.strip(),
                    created_at=datetime.utcnow().isoformat(),
                    embedding=emb,
                )
            print(f"[MEMORY] Saved Memory node: '{question[:50]}...'")
        except Exception as e:
            print(f"[MEMORY] Save Memory node failed: {e}")

    def _search_memory_nodes(self, question: str, top_k: int = 3) -> List[dict]:
        """Vector search tìm Memory nodes liên quan đến câu hỏi."""
        try:
            from embedding import get_embedding
            emb = get_embedding(question, task_type="retrieval_query")

            with self.store.driver.session() as session:
                results = session.run(
                    "CALL db.index.vector.queryNodes('memory_embeddings', $top_k, $embedding) "
                    "YIELD node, score "
                    "WHERE score > 0.72 "
                    "RETURN node.question AS question, node.answer AS answer, "
                    "       node.created_at AS created_at, score "
                    "ORDER BY score DESC",
                    top_k=top_k,
                    embedding=emb,
                )
                return [dict(r) for r in results]
        except Exception as e:
            print(f"[MEMORY] Memory vector search failed: {e}")
            return []

    # ══════════════════════════════════════════════
    # TẦNG 2b: LONG-TERM FACTS (Neo4j - UserFact nodes)
    # ══════════════════════════════════════════════

    def _extract_and_save_facts(self, question: str, answer: str):
        """
        Dùng LLM trích xuất thông tin/sự kiện quan trọng từ hội thoại.
        Lưu thành UserFact nodes trong Neo4j (vĩnh viễn, deduplicated).

        Ví dụ facts:
        - "Người dùng quan tâm đến quy định về thừa kế trong Bộ luật Dân sự"
        - "Người dùng hỏi về mức phạt vi phạm giao thông"
        - "Người dùng cần tìm hiểu thuế thu nhập cá nhân cho freelancer"
        """
        try:
            from llm_functions import _generate_with_fallback
            from embedding import get_embedding

            prompt = f"""Phân tích đoạn hội thoại pháp lý sau và trích xuất các THÔNG TIN QUAN TRỌNG cần nhớ lâu dài.

Hỏi: {question}
Trả lời: {answer[:500]}

Trích xuất 1-3 facts ngắn gọn (mỗi fact 1 dòng). Chỉ trích xuất nếu có thông tin đáng nhớ.
Facts nên bao gồm: chủ đề pháp lý user quan tâm, lĩnh vực luật liên quan, tình huống cụ thể của user.
Nếu hội thoại quá chung chung (chào hỏi, cảm ơn...) → trả về NONE.

Chỉ trả về các dòng fact, không đánh số, không giải thích. Hoặc trả về NONE."""

            raw = _generate_with_fallback(
                prompt,
                temperature=0.1,
                max_output_tokens=300,
            )

            if not raw or "NONE" in raw.upper():
                return

            facts = [
                line.strip().lstrip("- ").strip()
                for line in raw.strip().split("\n")
                if line.strip() and len(line.strip()) > 10 and "NONE" not in line.upper()
            ]

            for fact in facts[:3]:  # tối đa 3 facts per conversation
                self._save_fact_node(fact)

        except Exception as e:
            print(f"[MEMORY] Fact extraction failed: {e}")

    def _save_fact_node(self, fact_text: str):
        """
        Lưu 1 UserFact node vào Neo4j.
        Deduplicate: nếu đã có fact tương tự (cosine > 0.92), cập nhật thay vì tạo mới.
        """
        try:
            from embedding import get_embedding
            emb = get_embedding(fact_text, task_type="retrieval_document")

            with self.store.driver.session() as session:
                # Check duplicate bằng vector similarity
                existing = session.run(
                    "CALL db.index.vector.queryNodes('userfact_embeddings', 1, $embedding) "
                    "YIELD node, score "
                    "WHERE score > 0.92 "
                    "RETURN elementId(node) AS node_id, node.fact AS fact, score "
                    "LIMIT 1",
                    embedding=emb,
                ).single()

                if existing:
                    # Fact tương tự đã tồn tại → cập nhật timestamp + tăng count
                    session.run(
                        """
                        MATCH (f:UserFact) WHERE elementId(f) = $node_id
                        SET f.last_seen = $now,
                            f.hit_count = coalesce(f.hit_count, 1) + 1
                        """,
                        node_id=existing["node_id"],
                        now=datetime.utcnow().isoformat(),
                    )
                    print(f"[MEMORY] UserFact updated (duplicate): '{fact_text[:50]}...'")
                else:
                    # Fact mới → tạo node
                    session.run(
                        """
                        CREATE (f:UserFact {
                            fact:       $fact,
                            embedding:  $embedding,
                            created_at: $created_at,
                            last_seen:  $created_at,
                            hit_count:  1
                        })
                        """,
                        fact=fact_text,
                        embedding=emb,
                        created_at=datetime.utcnow().isoformat(),
                    )
                    print(f"[MEMORY] New UserFact saved: '{fact_text[:50]}...'")

        except Exception as e:
            print(f"[MEMORY] Save UserFact failed: {e}")

    def _search_fact_nodes(self, question: str, top_k: int = 3) -> List[dict]:
        """Vector search tìm UserFact nodes liên quan."""
        try:
            from embedding import get_embedding
            emb = get_embedding(question, task_type="retrieval_query")

            with self.store.driver.session() as session:
                results = session.run(
                    "CALL db.index.vector.queryNodes('userfact_embeddings', $top_k, $embedding) "
                    "YIELD node, score "
                    "WHERE score > 0.70 "
                    "RETURN node.fact AS fact, node.hit_count AS hit_count, score "
                    "ORDER BY score DESC",
                    top_k=top_k,
                    embedding=emb,
                )
                return [dict(r) for r in results]
        except Exception as e:
            print(f"[MEMORY] UserFact vector search failed: {e}")
            return []

    # ══════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════

    def add(self, question: str, answer: str):
        """
        Lưu hội thoại vào cả 2 tầng:
        1. Short-term RAM buffer (đồng bộ, nhanh)
        2. Long-term Neo4j Memory + UserFact extraction (async, background)
        """
        q = question.strip()
        a = answer.strip()
        if not q or not a:
            return

        # Tầng 1: RAM (instant)
        self._add_short_term(q, a)

        # Tầng 2: Neo4j (background thread để không block response)
        if self.store:
            thread = threading.Thread(
                target=self._save_long_term,
                args=(q, a),
                daemon=True,
            )
            thread.start()

    def _save_long_term(self, question: str, answer: str):
        """Background task: lưu Memory node + extract UserFacts."""
        self._save_memory_node(question, answer)
        self._extract_and_save_facts(question, answer)

    def get_context(self, question: str = "", n_short: int = 3, n_long: int = 3) -> str:
        """
        Lấy context từ CẢ HAI tầng memory:

        Returns:
            String gồm:
            - [SHORT-TERM] N hội thoại gần nhất (recency)
            - [LONG-TERM MEMORY] Hội thoại cũ liên quan (semantic)
            - [LONG-TERM FACTS] Các sự kiện đã biết về user (semantic)
        """
        parts = []

        # ── Tầng 1: Short-term (recency) ──
        short = self.get_short_term_context(n=n_short)
        if short.strip():
            parts.append(f"[HỘI THOẠI GẦN ĐÂY]\n{short}")

        # ── Tầng 2: Long-term (semantic search) ──
        if self.store and question.strip():
            # 2a: Memory nodes liên quan
            memories = self._search_memory_nodes(question, top_k=n_long)
            if memories:
                mem_lines = []
                for i, m in enumerate(memories, 1):
                    q = (m.get("question") or "").strip()
                    a = (m.get("answer") or "").strip()
                    if q and a:
                        a_short = a[:300] + "..." if len(a) > 300 else a
                        mem_lines.append(f"[Liên quan {i}] Hỏi: {q}\nTrả lời: {a_short}")
                if mem_lines:
                    parts.append(f"[TRÍ NHỚ DÀI HẠN]\n" + "\n\n".join(mem_lines))

            # 2b: UserFact nodes liên quan
            facts = self._search_fact_nodes(question, top_k=3)
            if facts:
                fact_lines = [f"- {f['fact']}" for f in facts if f.get("fact")]
                if fact_lines:
                    parts.append(f"[THÔNG TIN ĐÃ BIẾT VỀ NGƯỜI DÙNG]\n" + "\n".join(fact_lines))

        return "\n\n".join(parts)

    # Backward-compatible alias
    def get_recent_context(self, n: int = 5, question: str = "") -> str:
        """Alias cho get_context() — giữ tương thích với code cũ."""
        return self.get_context(question=question, n_short=min(n, 3), n_long=min(n, 3))

    # ══════════════════════════════════════════════
    # STATS & CLEAR
    # ══════════════════════════════════════════════

    def stats(self) -> dict:
        """Trả về thống kê memory cho API/debug."""
        result = {
            "short_term_count": len(self._short_term),
            "short_term_limit": self.short_term_limit,
        }
        if self.store:
            try:
                with self.store.driver.session() as session:
                    mem_count = session.run(
                        "MATCH (m:Memory) RETURN count(m) AS c"
                    ).single()["c"]
                    fact_count = session.run(
                        "MATCH (f:UserFact) RETURN count(f) AS c"
                    ).single()["c"]
                    result["long_term_memories"] = mem_count
                    result["long_term_facts"] = fact_count
            except Exception as e:
                result["long_term_error"] = str(e)
        return result

    def clear(self):
        """Xoá toàn bộ memory (RAM + Neo4j Memory + UserFact)."""
        self._short_term = []
        if self.store:
            try:
                with self.store.driver.session() as session:
                    session.run("MATCH (m:Memory) DETACH DELETE m")
                    session.run("MATCH (f:UserFact) DETACH DELETE f")
                print("[MEMORY] Cleared all memory (RAM + Neo4j)")
            except Exception as e:
                print(f"[MEMORY] Clear failed: {e}")

    def __len__(self):
        return len(self._short_term)