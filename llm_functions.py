# llm_functions.py
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional, List, Dict

from dotenv import load_dotenv
from google import genai
from google.genai import types
from memory_service import MemoryService

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Thiếu GEMINI_API_KEY trong file .env")

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
]

BAD_ENTITY_VALUES = {
    "của", "và", "là", "trong", "theo", "với", "hoặc", "cho", "được",
    "các", "những", "việc", "này", "đó", "khoản", "điều", "mục",
    "chương", "phần", "quy định", "liên quan", "lien_quan", "áp dụng",
    "trường hợp", "đối tượng", "nội dung", "căn cứ", "hành vi",
    "xử phạt", "pháp luật", "của pháp luật", "theo quy định",
    "người", "tổ chức", "cá nhân",
}

RELATION_NORMALIZATION = {
    "quy dinh": "QUY_DINH",
    "quy định": "QUY_DINH",
    "ap dung cho": "AP_DUNG_CHO",
    "áp dụng cho": "AP_DUNG_CHO",
    "xử phạt": "XU_PHAT",
    "xu phat": "XU_PHAT",
    "cấm": "CAM",
    "cam": "CAM",
    "yêu cầu": "YEU_CAU",
    "yeu cau": "YEU_CAU",
    "quản lý": "QUAN_LY",
    "quan ly": "QUAN_LY",
    "chịu trách nhiệm": "CHIU_TRACH_NHIEM",
    "chiu trach nhiem": "CHIU_TRACH_NHIEM",
    "liên quan": "IGNORE",
    "lien quan": "IGNORE",
    "có thời hạn": "CO_THOI_HAN",
    "hiệu lực": "HIEU_LUC",
}

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Ngưỡng vector score để xem chunk là thực sự liên quan
# Gemini embedding-001 thường cho score 0.5-0.8, không nên đặt quá cao
RELEVANCE_SCORE_THRESHOLD = 0.86

# Fallback: chấp nhận chunk có score thấp hơn nếu không có gì tốt hơn
RELEVANCE_SCORE_FALLBACK = 0.50

# Số ký tự tối thiểu của context PDF để xem là "có dữ liệu thực"
MIN_USEFUL_CONTEXT_CHARS = 200


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _normalize_spaces(text: str) -> str:
    return " ".join((text or "").strip().split())


def _strip_accidental_quotes(text: str) -> str:
    return (text or "").strip().strip('"').strip("'")


def _clean_json_text(text: str) -> str:
    return (text or "").replace("```json", "").replace("```", "").strip()


def _safe_parse_json(text: str) -> dict[str, Any]:
    text = _clean_json_text(text)
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {"summary": "", "triplets": []}


def _looks_like_bad_entity(text: str) -> bool:
    value = _normalize_spaces(_strip_accidental_quotes(text)).lower()
    if not value or value in BAD_ENTITY_VALUES or len(value) < 3:
        return True
    if value.replace(".", "").replace(",", "").replace("/", "").replace("-", "").isdigit():
        return True
    if len(value.split()) >= 8:
        return True
    return False


def _remove_vietnamese_accents(text: str) -> str:
    text = text.lower()
    replacements = {
        "á": "a", "à": "a", "ả": "a", "ã": "a", "ạ": "a",
        "ă": "a", "ắ": "a", "ằ": "a", "ẳ": "a", "ẵ": "a", "ặ": "a",
        "â": "a", "ấ": "a", "ầ": "a", "ẩ": "a", "ẫ": "a", "ậ": "a",
        "é": "e", "è": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
        "ê": "e", "ế": "e", "ề": "e", "ể": "e", "ễ": "e", "ệ": "e",
        "í": "i", "ì": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
        "ó": "o", "ò": "o", "ỏ": "o", "õ": "o", "ọ": "o",
        "ô": "o", "ố": "o", "ồ": "o", "ổ": "o", "ỗ": "o", "ộ": "o",
        "ơ": "o", "ớ": "o", "ờ": "o", "ở": "o", "ỡ": "o", "ợ": "o",
        "ú": "u", "ù": "u", "ủ": "u", "ũ": "u", "ụ": "u",
        "ư": "u", "ứ": "u", "ừ": "u", "ử": "u", "ữ": "u", "ự": "u",
        "ý": "y", "ỳ": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
        "đ": "d",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def _normalize_relation(text: str) -> str:
    raw = _normalize_spaces(_strip_accidental_quotes(text))
    lowered = raw.lower()
    lowered_ascii = _remove_vietnamese_accents(lowered)
    if lowered in RELATION_NORMALIZATION:
        return RELATION_NORMALIZATION[lowered]
    if lowered_ascii in RELATION_NORMALIZATION:
        return RELATION_NORMALIZATION[lowered_ascii]
    raw = raw.upper().replace(" ", "_")
    raw = re.sub(r"[^A-Z0-9_]", "", raw)
    return raw or "LIEN_QUAN"


def _dedupe_triplets(triplets: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    result = []
    for item in triplets:
        s, r, t = item["source"], item["relation"], item["target"]
        key = (s.lower(), r.upper(), t.lower())
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


# ─────────────────────────────────────────────
# RELEVANCE ASSESSMENT
# ─────────────────────────────────────────────

def assess_context_relevance(chunks: List[dict], question: str) -> dict:
    """
    Đánh giá xem các chunks trả về có thực sự liên quan đến câu hỏi không.

    Trả về dict:
        has_data   : bool  — True nếu có đủ dữ liệu PDF để trả lời
        confidence : str   — "high" | "medium" | "low"
    """
    if not chunks:
        print("[RELEVANCE] Không có chunk nào.")
        return {"has_data": False, "confidence": "low"}

    # Log score của tất cả chunks để dễ calibrate ngưỡng
    scores = [float(c.get("score", 0)) for c in chunks]
    print(f"[RELEVANCE] Scores: {[round(s, 3) for s in scores]}")

    # Lọc theo ngưỡng chính
    scored_chunks = [c for c in chunks if float(c.get("score", 0)) >= RELEVANCE_SCORE_THRESHOLD]

    # Fallback: nếu không chunk nào qua ngưỡng chính, thử ngưỡng fallback
    if not scored_chunks:
        scored_chunks = [c for c in chunks if float(c.get("score", 0)) >= RELEVANCE_SCORE_FALLBACK]
        if scored_chunks:
            print(f"[RELEVANCE] Dùng fallback threshold, {len(scored_chunks)} chunks.")
            # Vẫn có dữ liệu nhưng confidence thấp
            total_content = " ".join(c.get("content", "") for c in scored_chunks)
            if len(total_content.strip()) >= MIN_USEFUL_CONTEXT_CHARS:
                return {"has_data": True, "confidence": "low"}
        print("[RELEVANCE] Không có chunk đủ điểm → dùng kiến thức chung.")
        return {"has_data": False, "confidence": "low"}

    # Kiểm tra tổng content có đủ thực chất không
    total_content = " ".join(c.get("content", "") for c in scored_chunks)
    if len(total_content.strip()) < MIN_USEFUL_CONTEXT_CHARS:
        print("[RELEVANCE] Context quá ngắn → dùng kiến thức chung.")
        return {"has_data": False, "confidence": "low"}

    confidence = "high" if len(scored_chunks) >= 3 else "medium"
    print(f"[RELEVANCE] has_data=True, confidence={confidence}, chunks={len(scored_chunks)}")
    return {"has_data": True, "confidence": confidence}


# ─────────────────────────────────────────────
# EXTRACT TRIPLETS
# ─────────────────────────────────────────────

def extract_triplets(text: str) -> Dict[str, Any]:
    if not text.strip():
        return {"summary": "", "triplets": []}

    prompt = f"""
Bạn là hệ thống trích xuất knowledge graph cho văn bản pháp luật tiếng Việt.
Hãy trả về JSON THUẦN, không markdown, không giải thích.
Schema:
{{
  "summary": "Tóm tắt ngắn 1-2 câu",
  "triplets": [
    {{
      "source": "thực thể nguồn",
      "relation": "quan hệ",
      "target": "thực thể đích"
    }}
  ]
}}
Văn bản:
{text}
""".strip()

    try:
        raw_text = _generate_with_fallback(
            prompt, temperature=0.1, max_output_tokens=900, response_mime_type="application/json"
        )
        data = _safe_parse_json(raw_text)
    except Exception as e:
        print(f"[WARN] LLM extract_triplets error: {e}")
        return {"summary": "", "triplets": []}

    summary = _normalize_spaces(str(data.get("summary", "")))
    triplets = data.get("triplets", [])
    normalized = []
    for item in triplets:
        if not isinstance(item, dict):
            continue
        s = _normalize_spaces(_strip_accidental_quotes(str(item.get("source", ""))))
        r = _normalize_relation(str(item.get("relation", "")))
        t = _normalize_spaces(_strip_accidental_quotes(str(item.get("target", ""))))
        if _looks_like_bad_entity(s) or _looks_like_bad_entity(t):
            continue
        if s.lower() == t.lower() or r == "IGNORE":
            continue
        normalized.append({"source": s, "relation": r, "target": t})
    normalized = _dedupe_triplets(normalized)
    return {"summary": summary, "triplets": normalized}


# ─────────────────────────────────────────────
# GENERATE WITH FALLBACK
# ─────────────────────────────────────────────

def _generate_with_fallback(
    prompt: str,
    *,
    temperature: float,
    max_output_tokens: int,
    system_prompt: Optional[str] = None,
    response_mime_type: Optional[str] = None,
    max_retries: int = 3,
) -> str:
    import time
    last_error: Optional[Exception] = None
    for model in MODEL_CANDIDATES:
        for attempt in range(1, max_retries + 1):
            try:
                cfg_kwargs: dict[str, Any] = {
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                }
                if response_mime_type:
                    cfg_kwargs["response_mime_type"] = response_mime_type
                if system_prompt:
                    cfg_kwargs["system_instruction"] = system_prompt

                cfg = types.GenerateContentConfig(**cfg_kwargs)
                resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
                text = (resp.text or "").strip()
                if text:
                    return text
                # Trả về rỗng — thử lại
                print(f"[WARN] Model {model} trả về rỗng (attempt {attempt})")
            except Exception as e:
                err_str = str(e).lower()
                print(f"[WARN] Model {model} attempt {attempt} error: {e}")
                last_error = e
                # Rate limit → chờ rồi retry
                if "429" in err_str or "quota" in err_str or "rate" in err_str:
                    wait = 2 ** attempt  # 2s, 4s, 8s
                    print(f"[RETRY] Rate limit, chờ {wait}s...")
                    time.sleep(wait)
                    continue
                # Lỗi khác → chuyển model ngay
                break
    raise RuntimeError(f"Không gọi được model nào: {last_error}")


# ─────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────

def _build_system_prompt(memory: Optional[MemoryService], question: str = "") -> str:
    """
    System prompt = persona + 2-tier memory context.
    Memory đặt ở đây để KHÔNG làm nhiễu PDF context bên dưới.
    """
    memory_block = ""
    if memory:
        ctx = memory.get_context(question=question, n_short=3, n_long=3)
        if ctx.strip():
            memory_block = f"""

{ctx}

"""

    return f"""Bạn là trợ lý AI pháp lý tiếng Việt, chuyên tra cứu văn bản pháp luật.
Nhiệm vụ: Đọc kỹ tài liệu được cung cấp và trả lời câu hỏi dựa trên đó.

Quy tắc bắt buộc:
1. ĐỌC TOÀN BỘ nội dung tài liệu PDF trước khi kết luận có hay không có thông tin.
2. Nếu tìm thấy thông tin trong tài liệu: trả lời CHÍNH XÁC theo tài liệu, trích dẫn rõ ràng theo định dạng [Nguồn: tên_file, trang X, Điều Y (nếu có)].
3. Chỉ khi tài liệu THỰC SỰ không đề cập: trả lời từ kiến thức pháp luật Việt Nam chung mà không nhắc đến tài liệu.
4. KHÔNG được nói "tài liệu không có" nếu chưa đọc kỹ toàn bộ context.
5. KHÔNG bịa đặt con số, thời hạn, điều khoản không có trong tài liệu.
6. Câu trả lời mạch lạc, ngắn gọn, dùng đầu mục khi liệt kê.
7. QUAN TRỌNG: Chỉ dùng [Nguồn: ...] khi câu trả lời LẤY TRỰC TIẾP từ tài liệu. Nếu tài liệu không đề cập hoặc không rõ ràng, KHÔNG trích dẫn bất kỳ nguồn nào.{memory_block}"""


def _build_pdf_prompt(context: str, graph_facts: List[str], question: str, confidence: str) -> str:
    """Prompt khi CÓ dữ liệu PDF liên quan."""
    graph_text = "\n".join(graph_facts) if graph_facts else ""

    graph_block = f"""
=== TRI THỨC ĐỒ THỊ BỔ SUNG ===
{graph_text}
""" if graph_text else ""

    return f"""Dưới đây là các đoạn trích từ tài liệu pháp luật. Hãy đọc kỹ TỪNG ĐOẠN rồi trả lời câu hỏi.

=== NỘI DUNG TÀI LIỆU ===
{context}
{graph_block}
=== CÂU HỎI CẦN TRẢ LỜI ===
{question}

Yêu cầu:
- Tìm thông tin liên quan trong từng đoạn tài liệu ở trên.
- Trả lời dựa HOÀN TOÀN vào nội dung tài liệu, trích dẫn inline: [Nguồn: tên_file, trang X, Điều Y].
- Nếu nhiều đoạn đề cập cùng một vấn đề, tổng hợp lại thành câu trả lời đầy đủ.
- Nếu thực sự không tìm thấy thông tin trong tài liệu, hãy trả lời từ kiến thức pháp luật chung.""".strip()


def _build_general_prompt(question: str) -> str:
    """Prompt khi KHÔNG có dữ liệu PDF liên quan - trả lời tự nhiên."""
    return f"""
=== CÂU HỎI ===
{question}

Hãy trả lời câu hỏi trên dựa trên kiến thức pháp luật Việt Nam.
Trả lời tự nhiên, rõ ràng, ngắn gọn.
TUYỆT ĐỐI KHÔNG trích dẫn nguồn file, trang, điều khoản cụ thể.
Nếu không chắc chắn, nói rõ và khuyến nghị tham khảo văn bản pháp luật chính thức.
""".strip()


# ─────────────────────────────────────────────
# MAIN: ANSWER WITH CONTEXT
# ─────────────────────────────────────────────

def answer_question(
    context: str,
    question: str,
    graph_facts: Optional[List[str]] = None,
    memory: Optional[MemoryService] = None,
    chunks: Optional[List[dict]] = None,
) -> str:
    """
    Hàm trả lời chính.

    Luồng:
        1. Đánh giá relevance từ chunks (score-based).
        2. Build system prompt với memory tách biệt.
        3. Chọn prompt PDF hoặc kiến thức chung.
        4. Gọi LLM, lưu memory, trả về answer.
    """
    graph_facts = graph_facts or []

    # Đánh giá xem PDF context có hữu ích không
    if chunks is not None:
        relevance = assess_context_relevance(chunks, question)
        has_data = relevance["has_data"]
        confidence = relevance["confidence"]
    else:
        # Fallback nếu không truyền chunks: check độ dài context
        has_data = bool(context.strip()) and len(context.strip()) >= MIN_USEFUL_CONTEXT_CHARS
        confidence = "medium" if has_data else "low"

    system_prompt = _build_system_prompt(memory, question=question)

    if has_data and context.strip():
        user_prompt = _build_pdf_prompt(context, graph_facts, question, confidence)
        temperature = 0.05
        max_tokens = 3000
        mode = "PDF"
    else:
        user_prompt = _build_general_prompt(question)
        temperature = 0.25
        max_tokens = 2500
        mode = "GENERAL"

    print(f"[LLM] Mode={mode}, confidence={confidence}, context_len={len(context)}")

    try:
        answer = _generate_with_fallback(
            user_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_prompt=system_prompt,
        )
    except Exception as e:
        print(f"[ERROR] LLM answer error: {e}")
        answer = "Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi. Vui lòng thử lại."

    if memory:
        memory.add(question, answer)

    # Trả về tuple (answer, has_data) để rag_service biết ẩn sources khi general
    # Dùng attribute hack để không phá vỡ interface cũ
    answer_obj = answer
    return answer_obj, has_data


def answer_with_context(
    question: str,
    context: str,
    graph_facts: Optional[List[str]] = None,
    memory: Optional[MemoryService] = None,
    chunks: Optional[List[dict]] = None,
) -> tuple:
    """Alias giữ tương thích với rag_service.py. Trả về (answer, has_data)."""
    return answer_question(
        context=context,
        question=question,
        graph_facts=graph_facts,
        memory=memory,
        chunks=chunks,
    )