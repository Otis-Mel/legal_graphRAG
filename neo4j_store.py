from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from neo4j import GraphDatabase
from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, VECTOR_INDEX_NAME


@dataclass
class SearchResult:
    content: str
    source: str
    page: int
    score: float
    chunk_id: str


class Neo4jStore:
    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )

    def close(self) -> None:
        self.driver.close()

    # ========================
    # SCHEMA
    # ========================
    def setup_schema(self):
        with self.driver.session() as session:
            # Drop old text index
            session.run("DROP INDEX entity_name_index IF EXISTS;")

            # Entity unique constraint
            try:
                session.run("""
                    CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
                    FOR (e:Entity)
                    REQUIRE e.name IS UNIQUE;
                """)
            except Exception as e:
                print("[WARN] Constraint creation skipped:", e)

            # Vector index cho Chunk.embedding
            # gemini-embedding-001 = 3072 dims
            # Nếu báo lỗi dimension mismatch: chạy len(get_embedding("test")) để kiểm tra
            try:
                session.run(
                    "CREATE VECTOR INDEX " + VECTOR_INDEX_NAME + " IF NOT EXISTS "
                    "FOR (c:Chunk) ON c.embedding "
                    "OPTIONS { indexConfig: { "
                    "`vector.dimensions`: 3072, "
                    "`vector.similarity_function`: 'cosine' "
                    "} }"
                )
                print("[SCHEMA] Vector index '" + VECTOR_INDEX_NAME + "' ready.")
            except Exception as e:
                print("[WARN] Vector index creation skipped:", e)

    # ========================
    # INGEST
    # ========================
    def ingest_chunk(
        self,
        chunk: dict[str, Any],
        embedding: list[float],
        summary: str,
        triplets: list[dict[str, str]],
    ) -> None:
        with self.driver.session() as session:
            session.run(
                """
                MERGE (d:Document {name: $source})
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.content = $content,
                    c.page = $page,
                    c.source = $source,
                    c.embedding = $embedding,
                    c.summary = $summary
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                chunk_id=chunk["chunk_id"],
                content=chunk["content"],
                page=chunk["page"],
                source=chunk["source"],
                embedding=embedding,
                summary=summary,
            )

            for triplet in triplets:
                s = (triplet.get("source") or "").strip()
                t = (triplet.get("target") or "").strip()
                r = (triplet.get("relation") or "LIEN_QUAN").strip().upper()
                if not s or not t:
                    continue
                session.run(
                    """
                    MERGE (s:Entity {name: $s})
                    MERGE (t:Entity {name: $t})
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    MERGE (c)-[:MENTIONS]->(s)
                    MERGE (c)-[:MENTIONS]->(t)
                    MERGE (s)-[rel:RELATES {type: $r}]->(t)
                    """,
                    s=s, t=t, r=r, chunk_id=chunk["chunk_id"],
                )

    # ========================
    # VECTOR SEARCH
    # ========================
    def vector_search(self, embedding: list[float], top_k: int = 5):
        query = (
            "CALL db.index.vector.queryNodes('" + VECTOR_INDEX_NAME + "', $top_k, $embedding) "
            "YIELD node, score "
            "RETURN node.chunk_id AS chunk_id, "
            "       node.content  AS content, "
            "       node.source   AS source, "
            "       node.page     AS page, "
            "       score "
            "ORDER BY score DESC"
        )
        with self.driver.session() as session:
            return [dict(r) for r in session.run(query, top_k=top_k, embedding=embedding)]

    # ========================
    # KEYWORD SEARCH
    # ========================
    def keyword_search(self, text: str, limit: int = 5):
        terms = [t.strip() for t in text.split() if len(t.strip()) >= 2]
        query = """
        MATCH (c:Chunk)
        WHERE any(term IN $terms WHERE toLower(c.content) CONTAINS toLower(term))
        RETURN c.chunk_id AS chunk_id,
               c.content  AS content,
               c.source   AS source,
               c.page     AS page,
               0.5        AS score
        LIMIT $limit
        """
        # Keyword chunks dùng score=0.5 (giữa 2 ngưỡng) để được nhận vào context
        with self.driver.session() as session:
            return [dict(r) for r in session.run(query, terms=terms, limit=limit)]

    # ========================
    # FULL GRAPH DATA
    # ========================
    def get_graph_data(self, limit: int = 200):
        query = """
        MATCH (e1:Entity)-[r:RELATES]->(e2:Entity)
        RETURN e1.name AS source, r.type AS relation, e2.name AS target
        LIMIT $limit
        """
        with self.driver.session() as session:
            records = session.run(query, limit=limit)
            nodes = {}
            edges = []
            for r in records:
                s, t, rel = r["source"], r["target"], r["relation"] or "LIEN_QUAN"
                if s not in nodes:
                    nodes[s] = {"id": s, "label": s, "group": "entity"}
                if t not in nodes:
                    nodes[t] = {"id": t, "label": t, "group": "entity"}
                edges.append({"from": s, "to": t, "label": rel})
            return {"nodes": list(nodes.values()), "edges": edges}

    # ========================
    # GRAPH SEARCH (theo câu hỏi)
    # ========================
    def graph_search(self, question: str, top_k: int = 10):
        terms = [t.strip().lower() for t in question.split() if len(t.strip()) >= 2][:6]
        query = """
        MATCH (e1:Entity)-[r:RELATES]->(e2:Entity)
        WHERE any(term IN $terms WHERE toLower(e1.name) CONTAINS term
                                   OR toLower(e2.name) CONTAINS term)
        RETURN e1.name AS source, r.type AS relation, e2.name AS target
        LIMIT $top_k
        """
        with self.driver.session() as session:
            records = session.run(query, terms=terms, top_k=top_k)
            nodes = {}
            edges = []
            for r in records:
                s, t, rel = r["source"], r["target"], r["relation"] or "LIEN_QUAN"
                if s not in nodes:
                    nodes[s] = {"id": s, "label": s, "group": "seed"}
                if t not in nodes:
                    nodes[t] = {"id": t, "label": t, "group": "entity"}
                edges.append({"from": s, "to": t, "label": rel})
            return {"nodes": list(nodes.values()), "edges": edges}

    # ========================
    # ENTITY DETAIL
    # ========================
    def get_entity_context(self, entity_name: str, limit: int = 6):
        with self.driver.session() as session:
            chunks = session.run(
                """
                MATCH (e:Entity {name: $name})<-[:MENTIONS]-(c:Chunk)
                RETURN c.content AS content, c.source AS source, c.page AS page
                LIMIT $limit
                """,
                name=entity_name, limit=limit,
            )
            return {"entity": entity_name, "chunks": [dict(r) for r in chunks]}

    # ========================
    # STATISTICS
    # ========================
    def stats(self):
        with self.driver.session() as session:
            return {
                "documents": session.run("MATCH (d:Document) RETURN count(d) AS n").single()["n"],
                "chunks":    session.run("MATCH (c:Chunk)    RETURN count(c) AS n").single()["n"],
                "entities":  session.run("MATCH (e:Entity)   RETURN count(e) AS n").single()["n"],
            }

    # ========================
    # DOCUMENT MANAGEMENT — MỚI
    # ========================

    def get_ingested_documents(self) -> dict[str, int]:
        """
        Trả về dict {tên_file: số_chunks} của tất cả document đã ingest.
        Dùng để check file nào đã tồn tại → skip khi ingest incremental.
        """
        with self.driver.session() as session:
            results = session.run("""
                MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
                RETURN d.name AS name, count(c) AS chunk_count
                ORDER BY d.name
            """)
            return {r["name"]: r["chunk_count"] for r in results}

    def delete_document(self, doc_name: str) -> int:
        """
        Xóa 1 document + tất cả chunks + entity liên quan.
        Trả về số chunks đã xóa.
        Dùng khi cần re-ingest lại 1 file cụ thể.
        """
        with self.driver.session() as session:
            # Đếm chunks trước
            result = session.run(
                "MATCH (d:Document {name: $name})-[:HAS_CHUNK]->(c:Chunk) "
                "RETURN count(c) AS n",
                name=doc_name,
            )
            count = result.single()["n"]

            # Xóa entities chỉ liên quan đến document này
            # (entity được MENTIONS bởi chunk của document khác thì giữ lại)
            session.run("""
                MATCH (d:Document {name: $name})-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE NOT exists {
                    MATCH (other:Chunk)-[:MENTIONS]->(e)
                    WHERE other.source <> $name
                }
                DETACH DELETE e
            """, name=doc_name)

            # Xóa chunks + document
            session.run("""
                MATCH (d:Document {name: $name})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                DETACH DELETE c, d
            """, name=doc_name)

            print(f"[NEO4J] Đã xóa document '{doc_name}' + {count} chunks")
            return count