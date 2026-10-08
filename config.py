import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', '12345678')
PDF_DIR = Path(os.getenv('PDF_DIR', str(BASE_DIR / 'pdfs')))
VECTOR_INDEX_NAME = os.getenv('VECTOR_INDEX_NAME', 'legal_vector_index')

# ============================================================
# CẢI THIỆN 1: Tăng TOP_K từ 5 → 10
# Lý do: TOP_K=5 quá thấp, query expansion tạo 3 query nhưng mỗi
# query chỉ lấy 8 chunk (TOP_K+3). Với 10, mỗi query lấy 13 chunk,
# RRF merge sẽ có nhiều ứng viên hơn → recall tăng → P@K + MRR tăng.
# ============================================================
TOP_K = int(os.getenv('TOP_K', '10'))

# ============================================================
# CẢI THIỆN 2: Tăng MAX_CONTEXT_CHARS từ 12000 → 16000
# Lý do: Cho phép đưa nhiều context hơn vào LLM, giúp answer
# đầy đủ hơn → BERTScore tăng (vì bao phủ được nhiều thông tin
# trong expected_answer hơn).
# ============================================================
MAX_CONTEXT_CHARS = int(os.getenv('MAX_CONTEXT_CHARS', '16000'))

# ============================================================
# CẢI THIỆN 3 (YÊU CẦU RE-INGEST): Giảm CHUNK_SIZE + tăng OVERLAP
# CHUNK_SIZE: 1200 → 800
#   - Chunk nhỏ hơn = mỗi chunk tập trung vào 1 chủ đề
#   - Vector embedding sẽ specific hơn → cosine similarity chính xác hơn
#   - Giảm noise trong mỗi chunk → retrieval precision tăng
# CHUNK_OVERLAP: 200 → 300
#   - Overlap 37.5% (thay vì 16.7%) đảm bảo không cắt giữa câu/điều
#   - Thông tin ở ranh giới chunk không bị mất
# ============================================================
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '800'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '300'))

GEMINI_CHAT_MODEL = os.getenv('GEMINI_CHAT_MODEL', 'gemini-2.0-flash')
GEMINI_EMBED_MODEL = os.getenv('GEMINI_EMBED_MODEL', 'models/embedding-001')

PDF_DIR.mkdir(parents=True, exist_ok=True)