from __future__ import annotations

import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Nếu trang có ít hơn số ký tự này → coi là scanned → dùng OCR
OCR_TEXT_THRESHOLD = 50


# ─────────────────────────────────────────────
# OCR SUPPORT
# ─────────────────────────────────────────────

def _ocr_available() -> bool:
    """Kiểm tra pytesseract và pdf2image có được cài không."""
    try:
        import pytesseract
        import pdf2image
        return True
    except ImportError:
        return False


def _get_poppler_path() -> str | None:
    """Tìm poppler path trên macOS (Homebrew) hoặc Linux."""
    import shutil
    # macOS Homebrew Apple Silicon
    if Path("/opt/homebrew/bin/pdfinfo").exists():
        return "/opt/homebrew/bin"
    # macOS Homebrew Intel
    if Path("/usr/local/bin/pdfinfo").exists():
        return "/usr/local/bin"
    # Linux / PATH
    if shutil.which("pdfinfo"):
        return None  # đã có trong PATH
    return None


def _ocr_pdf_page(pdf_path: Path, page_num_0indexed: int) -> str:
    """
    OCR một trang PDF (0-indexed) bằng pytesseract.
    Trả về text đã nhận dạng.
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract

        poppler_path = _get_poppler_path()
        print(f"[OCR] poppler_path={poppler_path}, trang {page_num_0indexed + 1}")

        images = convert_from_path(
            str(pdf_path),
            first_page=page_num_0indexed + 1,
            last_page=page_num_0indexed + 1,
            dpi=300,
            poppler_path=poppler_path,
        )
        if not images:
            print(f"[OCR] Không convert được trang {page_num_0indexed + 1}")
            return ""

        text = pytesseract.image_to_string(
            images[0],
            lang="vie+eng",
            config="--psm 3",
        )
        result = text.strip()
        print(f"[OCR] Trang {page_num_0indexed + 1}: {len(result)} ký tự")
        return result

    except Exception as e:
        print(f"[OCR] Lỗi trang {page_num_0indexed + 1}: {e}")
        return ""


def _is_scanned_page(text: str) -> bool:
    """Trả về True nếu trang có quá ít text → khả năng là scanned."""
    return len((text or "").strip()) < OCR_TEXT_THRESHOLD


# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────

def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─────────────────────────────────────────────
# LEGAL METADATA EXTRACTION
# ─────────────────────────────────────────────

def _extract_legal_refs(text: str) -> dict:
    """
    Bắt Điều / Khoản / Chương / Mục đầu tiên trong chunk
    để hỗ trợ citation khi trả lời.
    """
    dieu_match  = re.search(r"(Điều\s+\d+\.?\s*[^\n]*)",   text, re.IGNORECASE)
    khoan_match = re.search(r"(Khoản\s+\d+\.?\s*[^\n]*)",  text, re.IGNORECASE)
    chuong_match = re.search(r"(Chương\s+[IVXLC\d]+\.?\s*[^\n]*)", text, re.IGNORECASE)
    muc_match   = re.search(r"(Mục\s+[IVXLC\d]+\.?\s*[^\n]*)",    text, re.IGNORECASE)

    return {
        "dieu":   dieu_match.group(1).strip()   if dieu_match   else "",
        "khoan":  khoan_match.group(1).strip()  if khoan_match  else "",
        "chuong": chuong_match.group(1).strip() if chuong_match else "",
        "muc":    muc_match.group(1).strip()    if muc_match    else "",
    }


# ─────────────────────────────────────────────
# SPLITTER
# ─────────────────────────────────────────────

def _make_splitter() -> RecursiveCharacterTextSplitter:
    """Ưu tiên cắt theo cấu trúc pháp luật trước, rồi đến đoạn/câu."""
    separators = [
        "\nĐiều ",
        "\nKhoản ",
        "\nMục ",
        "\nChương ",
        "\nPhần ",
        "\n\n",
        "\n",
        ". ",
        "; ",
        " ",
        "",
    ]
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )


# ─────────────────────────────────────────────
# MAIN LOADER
# ─────────────────────────────────────────────

def load_pdf_documents(pdf_path: str | Path) -> list[dict]:
    """
    Load PDF → chunks. Tự động phát hiện trang scanned và dùng OCR.

    Requires (chỉ khi có trang scanned):
        pip install pytesseract pdf2image
        brew install tesseract tesseract-lang   # macOS
        # hoặc: apt install tesseract-ocr tesseract-ocr-vie  # Ubuntu
    """
    pdf_path = Path(pdf_path)
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    use_ocr = _ocr_available()
    splitter = _make_splitter()
    results: list[dict] = []

    for page_doc in docs:
        page_num = int(page_doc.metadata.get("page", 0)) + 1
        raw_text = _clean_text(page_doc.page_content)

        # Phát hiện trang scanned → OCR
        if _is_scanned_page(raw_text):
            if use_ocr:
                print(f"[OCR] Trang {page_num} của {pdf_path.name} là scanned → dùng OCR")
                ocr_text = _ocr_pdf_page(pdf_path, page_num - 1)
                raw_text = _clean_text(ocr_text)
                if not raw_text:
                    print(f"[OCR] Trang {page_num}: OCR không ra text, bỏ qua")
                    continue
            else:
                print(
                    f"[WARN] Trang {page_num} của {pdf_path.name} có vẻ là scanned "
                    f"nhưng pytesseract/pdf2image chưa cài → bỏ qua trang này.\n"
                    f"       Để bật OCR: pip install pytesseract pdf2image\n"
                    f"       macOS: brew install tesseract tesseract-lang"
                )
                continue

        if not raw_text:
            continue

        split_docs = splitter.create_documents(
            [raw_text],
            metadatas=[{"page": page_num, "source": pdf_path.name}],
        )

        for local_idx, doc in enumerate(split_docs):
            content = _clean_text(doc.page_content)
            if not content:
                continue

            legal_refs = _extract_legal_refs(content)

            results.append({
                "chunk_id":     f"{pdf_path.stem}-p{page_num}-c{local_idx}",
                "content":      content,
                "page":         page_num,
                "source":       pdf_path.name,
                "document_name": pdf_path.stem,
                "dieu":         legal_refs["dieu"],
                "khoan":        legal_refs["khoan"],
                "chuong":       legal_refs["chuong"],
                "muc":          legal_refs["muc"],
                "is_ocr":       _is_scanned_page(page_doc.page_content or ""),
            })

    return results


# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────

def find_pdf_files(pdf_dir: str | Path) -> list[Path]:
    return sorted(Path(pdf_dir).glob("*.pdf"))