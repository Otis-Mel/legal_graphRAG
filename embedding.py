from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("Thiếu GEMINI_API_KEY trong file .env")

client = genai.Client(api_key=GEMINI_API_KEY)

EMBED_MODEL_CANDIDATES = [
    "gemini-embedding-001",
]


def get_embedding(text: str, task_type: str = "retrieval_document") -> List[float]:
    text = (text or "").strip()
    if not text:
        return []

    last_error: Exception | None = None

    for model_name in EMBED_MODEL_CANDIDATES:
        try:
            print(f" Using embedding model: {model_name}")

            response = client.models.embed_content(
                model=model_name,
                contents=text,
            )

            # SDK mới trả về embeddings là list các giá trị
            if hasattr(response, "embeddings") and response.embeddings:
                first = response.embeddings[0]
                if hasattr(first, "values") and first.values:
                    return list(first.values)

            # fallback nếu trả về embedding đơn (trường hợp cũ)
            if hasattr(response, "embedding") and response.embedding:
                emb = response.embedding
                if hasattr(emb, "values") and emb.values:
                    return list(emb.values)

            raise RuntimeError(f"Không lấy được vector từ model {model_name}")

        except Exception as e:
            print(f"[WARN] Embedding model {model_name} lỗi: {e}")
            last_error = e

    # Khi không thành công sau khi thử hết các model, log chi tiết lỗi
    error_message = f"Lỗi khi gọi Gemini Embedding: {last_error}"
    print(error_message)
    raise RuntimeError(error_message)