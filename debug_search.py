#!/usr/bin/env python3
# debug_search.py - chạy file này trong thư mục project để kiểm tra
# python3 debug_search.py

import sys
sys.path.insert(0, '.')

from embedding import get_embedding
from neo4j_store import Neo4jStore

store = Neo4jStore()

test_questions = [
    "Hồ sơ đăng ký chỉ định cơ sở kiểm nghiệm thực phẩm phục vụ quản lý nhà nước bao gồm những gì?",
    "Trường hợp người được giám hộ đã có năng lực hành vi dân sự đầy đủ thì trong thời hạn là bao nhiêu ngày?",
]

for q in test_questions:
    print(f"\n{'='*60}")
    print(f"CÂU HỎI: {q}")
    print(f"{'='*60}")
    
    try:
        emb = get_embedding(q, task_type="retrieval_query")
        results = store.vector_search(emb, top_k=5)
        
        if not results:
            print("  → Không có kết quả vector search!")
        else:
            for i, r in enumerate(results, 1):
                print(f"\n  [{i}] Score: {r.get('score', 0):.4f}")
                print(f"      File: {r.get('source', '?')} - Trang: {r.get('page', '?')}")
                print(f"      Chunk ID: {r.get('chunk_id', '?')}")
                content_preview = (r.get('content') or '')[:200].replace('\n', ' ')
                print(f"      Nội dung: {content_preview}...")
    except Exception as e:
        print(f"  → LỖI: {e}")

store.close()
print(f"\n{'='*60}")
print("Xong. Kiểm tra score và file nguồn ở trên.")