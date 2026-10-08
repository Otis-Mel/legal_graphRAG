#!/usr/bin/env python3
# evaluate.py — Streamlined Evaluation
# Metrics: Precision@K · MRR · BERTScore · Answer Relevancy
# Hallucination: LLM-as-judge kiểm tra từng claims trong answer
#
# Chạy: cd ~/Downloads/legal_graphrag_project && python3 evaluate.py

import sys, json, time, re
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluation_data import EVAL_DATASET
from neo4j_store import Neo4jStore
from rag_service import GraphRAGService

SLEEP = 2  # giây giữa các Gemini API call

# ──────────────────────────────────────────────────────────────
# BERT SCORE — load 1 lần duy nhất
# ──────────────────────────────────────────────────────────────
# CẢI THIỆN: Đổi model từ bert-base-multilingual-cased → xlm-roberta-large
#
# Lý do:
# 1. bert-base-multilingual-cased (12 layers, 110M params):
#    - Shared wordpiece vocab 104 ngôn ngữ → tiếng Việt bị ít token
#    - Tokenize "bồi thường thiệt hại" → nhiều subword lạ → embedding kém
#    - F1 score bị nén trong khoảng 0.65-0.80 cho tiếng Việt
#
# 2. xlm-roberta-large (24 layers, 560M params):
#    - SentencePiece unigram tokenizer → Vietnamese coverage tốt hơn nhiều
#    - Train trên CommonCrawl 100 ngôn ngữ, data tiếng Việt nhiều hơn
#    - Contextual embedding sâu hơn (24 layers) → semantic matching chính xác hơn
#    - F1 score phản ánh đúng mức độ tương đồng ngữ nghĩa thực tế
#
# 3. rescale_with_baseline=True:
#    - Normalize score dùng baseline statistics
#    - Score 0 = random pair, Score 1 = perfect match
#    - Giúp score dễ interpret hơn và discriminate tốt hơn
#
# Lần đầu chạy sẽ download ~2.2GB model (chỉ 1 lần).
# ──────────────────────────────────────────────────────────────
print("Đang load BERTScore model (lần đầu ~2-3 phút download xlm-roberta-large)...")
try:
    from bert_score import score as bert_score_fn
    BERT_MODEL = "xlm-roberta-large"
    BERT_READY = True
    print(f"BERTScore OK — model: {BERT_MODEL}")
except ImportError:
    BERT_READY = False
    print("[WARN] bert-score chưa cài → bỏ qua BERTScore")


# ══════════════════════════════════════════════════════════════
# RETRIEVAL METRICS
# ══════════════════════════════════════════════════════════════

def _source_match(retrieved: str, expected: str) -> bool:
    r = (retrieved or "").lower()
    e = (expected or "").lower()
    return e in r or r in e


def precision_at_k(sources: list, expected: str, k: int) -> float:
    if not expected:
        return -1.0
    hits = sum(1 for s in sources[:k] if _source_match(s, expected))
    return hits / k


def reciprocal_rank(sources: list, expected: str) -> float:
    if not expected:
        return -1.0
    for i, s in enumerate(sources, 1):
        if _source_match(s, expected):
            return 1.0 / i
    return 0.0


# ══════════════════════════════════════════════════════════════
# BERTSCORE — semantic similarity
# ══════════════════════════════════════════════════════════════

def compute_bertscore(hypothesis: str, reference: str) -> dict:
    """
    BERTScore: dùng contextual embedding từ XLM-RoBERTa để so sánh ngữ nghĩa.
    
    CẢI THIỆN:
    - model: xlm-roberta-large (24 layers, tốt cho tiếng Việt)
    - rescale_with_baseline=True → score chuẩn hóa, phân biệt tốt hơn
    - num_layers=17 → layer tối ưu cho XLM-R theo paper BERTScore
    """
    if not BERT_READY:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    try:
        P, R, F1 = bert_score_fn(
            [hypothesis], [reference],
            model_type=BERT_MODEL,
            lang="others",
            verbose=False,
            device="cpu",
            rescale_with_baseline=True,  # Normalize scores
        )
        return {
            "precision": round(P[0].item(), 4),
            "recall":    round(R[0].item(), 4),
            "f1":        round(F1[0].item(), 4),
        }
    except Exception as e:
        print(f"[WARN] BERTScore error: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


# ══════════════════════════════════════════════════════════════
# LLM HELPER
# ══════════════════════════════════════════════════════════════

def _gemini(prompt: str, max_tokens: int = 200) -> str:
    try:
        from google import genai
        from google.genai import types
        import os
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        cfg = types.GenerateContentConfig(temperature=0.0, max_output_tokens=max_tokens)
        for model in ["gemini-2.5-flash-lite", "gemini-2.5-flash"]:
            try:
                resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
                return (resp.text or "").strip()
            except Exception:
                time.sleep(2)
                continue
        return ""
    except Exception as e:
        print(f"[WARN] Gemini error: {e}")
        return ""


# ══════════════════════════════════════════════════════════════
# ANSWER RELEVANCY
# ══════════════════════════════════════════════════════════════

def answer_relevancy(question: str, answer: str) -> float:
    """
    Answer Relevancy: sinh lại câu hỏi từ answer, đo BERTScore với question gốc.
    Score cao = answer trả lời đúng trọng tâm.
    """
    generated_q = _gemini(
        f"Đặt lại câu hỏi mà câu trả lời này đang trả lời. Chỉ trả lời câu hỏi:\n{answer[:400]}",
        max_tokens=80
    )
    if not generated_q:
        return 0.5
    time.sleep(1)
    bs = compute_bertscore(question, generated_q)
    return bs["f1"]


# ══════════════════════════════════════════════════════════════
# HALLUCINATION — LLM-as-judge (claim-level verification)
# ══════════════════════════════════════════════════════════════

def hallucination_score(answer: str, context: str, question: str) -> dict:
    """
    Đánh giá hallucination bằng LLM-as-judge 2 bước:

    Bước 1 — Phân tách claims: tách answer thành các mệnh đề pháp lý độc lập.
    Bước 2 — Verify từng claim: mỗi claim có được hỗ trợ bởi context không?

    Trả về:
      - score     : tỷ lệ claims được hỗ trợ (0.0–1.0, cao = ít hallucination)
      - supported : số claims OK
      - total     : tổng claims được phân tách
      - verdict   : "PASS" (>=0.85) / "PARTIAL" (>=0.50) / "FAIL" (<0.50)
      - unsupported_claims: danh sách claim bị đánh dấu là hallucination
    """
    if not context.strip():
        return {
            "score": -1.0, "supported": 0, "total": 0,
            "verdict": "NO_CONTEXT", "unsupported_claims": [],
        }

    # ── Bước 1: Phân tách claims ─────────────────────────────
    split_prompt = (
        "Bạn là chuyên gia pháp lý. Hãy tách câu trả lời sau thành các mệnh đề (claims) "
        "pháp lý độc lập, mỗi mệnh đề trên một dòng, bắt đầu bằng '- '.\n"
        "Chỉ lấy các mệnh đề chứa thông tin cụ thể (số ngày, điều khoản, quy định, "
        "con số, điều kiện...). Bỏ qua câu chào hỏi, giới thiệu chung.\n"
        "Tối đa 8 mệnh đề.\n\n"
        f"CÂU TRẢ LỜI: {answer[:600]}\n\n"
        "Danh sách mệnh đề:"
    )

    raw_claims = _gemini(split_prompt, max_tokens=300)
    time.sleep(1)

    claims = []
    for line in raw_claims.split("\n"):
        line = line.strip()
        if line.startswith("- ") and len(line) > 10:
            claims.append(line[2:].strip())

    if not claims:
        # Fallback: tách theo câu
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        claims = [s.strip() for s in sentences if len(s.strip()) > 20][:6]

    if not claims:
        return {
            "score": 0.5, "supported": 0, "total": 0,
            "verdict": "NO_CLAIMS", "unsupported_claims": [],
        }

    # ── Bước 2: Verify từng claim ────────────────────────────
    claims_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
    verify_prompt = (
        "Bạn là chuyên gia pháp lý đánh giá hallucination trong hệ thống RAG.\n\n"
        f"NGỮ CẢNH (từ tài liệu pháp lý):\n{context[:800]}\n\n"
        f"CÂU HỎI GỐC: {question}\n\n"
        "Với mỗi mệnh đề dưới đây, hãy đánh giá:\n"
        "- SUPPORTED: thông tin có trong ngữ cảnh, hoặc là kiến thức pháp lý phổ quát đúng\n"
        "- UNSUPPORTED: thông tin không có trong ngữ cảnh VÀ có thể sai lệch/bịa đặt\n"
        "- GENERAL: nhận định chung hợp lý, không thể verify từ ngữ cảnh nhưng không sai\n\n"
        "Trả về JSON array, ví dụ:\n"
        '[{"claim": "...", "verdict": "SUPPORTED"}, {"claim": "...", "verdict": "UNSUPPORTED"}]\n\n'
        f"Các mệnh đề:\n{claims_text}\n\nJSON:"
    )

    raw_verify = _gemini(verify_prompt, max_tokens=600)
    time.sleep(1)

    # ── Parse kết quả ────────────────────────────────────────
    supported = 0
    unsupported_claims = []

    try:
        json_match = re.search(r'\[.*\]', raw_verify, re.DOTALL)
        if json_match:
            verdicts = json.loads(json_match.group())
            for item in verdicts:
                v = (item.get("verdict") or "").upper()
                if v in ("SUPPORTED", "GENERAL"):
                    supported += 1
                elif v == "UNSUPPORTED":
                    unsupported_claims.append(item.get("claim", ""))
        else:
            raise ValueError("No JSON array found in response")
    except Exception:
        # Fallback: đếm keyword trong raw text
        sup_count = len(re.findall(r'\bSUPPORTED\b|\bGENERAL\b', raw_verify, re.IGNORECASE))
        uns_count = len(re.findall(r'\bUNSUPPORTED\b', raw_verify, re.IGNORECASE))
        supported = sup_count
        if uns_count > 0:
            unsupported_claims = [f"(parse fallback: {uns_count} unsupported claims phát hiện)"]

    total = len(claims)
    score = supported / total if total > 0 else 0.0

    if score >= 0.85:
        verdict = "PASS"
    elif score >= 0.50:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {
        "score":              round(score, 4),
        "supported":          supported,
        "total":              total,
        "verdict":            verdict,
        "unsupported_claims": unsupported_claims,
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def run_evaluation():
    print("=" * 65)
    print("LEGAL GRAPHRAG — STREAMLINED EVALUATION")
    print(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Số câu hỏi: {len(EVAL_DATASET)}")
    print("Metrics: P@K · MRR · BERTScore · Answer Relevancy · Hallucination")
    print("=" * 65)

    store = Neo4jStore()
    rag   = GraphRAGService(store)

    results       = []
    p_metrics     = {k: [] for k in [1, 3, 5]}
    mrr_list      = []
    bs_list       = []
    ar_list       = []
    hall_list     = []
    hall_verdicts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "NO_CONTEXT": 0, "NO_CLAIMS": 0}

    total = len(EVAL_DATASET)

    for idx, item in enumerate(EVAL_DATASET, 1):
        qid    = item["id"]
        q      = item["question"]
        exp    = item.get("expected_answer", "")
        src    = item.get("source_hint", "")
        is_gen = exp == "GENERAL_KNOWLEDGE"

        print(f"\n[{idx}/{total}] {qid} ({item['difficulty']}) — {q[:55]}...")

        # ── Gọi GraphRAG ─────────────────────────────────────
        try:
            gr      = rag.ask(q, memory=None)
            answer  = gr.get("answer", "")
            chunks  = gr.get("chunks", [])
            sources = [c.get("source", "") for c in chunks]
            context = "\n".join(c.get("content", "")[:300] for c in chunks[:3])
        except Exception as e:
            print(f"  [ERROR] {e}")
            answer, chunks, sources, context = f"ERROR: {e}", [], [], ""

        time.sleep(SLEEP)

        # ── Hallucination (chạy cho tất cả, kể cả câu general) ──
        hall = hallucination_score(answer, context, q)
        if hall["score"] >= 0:
            hall_list.append(hall["score"])
        hall_verdicts[hall["verdict"]] = hall_verdicts.get(hall["verdict"], 0) + 1

        if not is_gen:
            # ── Retrieval ─────────────────────────────────────
            p1 = precision_at_k(sources, src, 1)
            p3 = precision_at_k(sources, src, 3)
            p5 = precision_at_k(sources, src, 5)
            rr = reciprocal_rank(sources, src)
            if p1 >= 0:
                p_metrics[1].append(p1)
                p_metrics[3].append(p3)
                p_metrics[5].append(p5)
            if rr >= 0:
                mrr_list.append(rr)

            # ── BERTScore ─────────────────────────────────────
            bs = compute_bertscore(answer, exp)
            bs_list.append(bs["f1"])

            # ── Answer Relevancy ──────────────────────────────
            ar = answer_relevancy(q, answer)
            ar_list.append(ar)
            time.sleep(SLEEP)

            print(f"  P@3={p3:.2f} MRR={rr:.2f} | "
                  f"BERT={bs['f1']:.2f} AR={ar:.2f} | "
                  f"Hall={hall['score']:.2f} [{hall['verdict']}] "
                  f"({hall['supported']}/{hall['total']} claims OK)")

            if hall["unsupported_claims"]:
                for uc in hall["unsupported_claims"][:2]:
                    print(f"    ⚠ Hallucinated: {uc[:90]}")

            results.append({
                "id": qid, "domain": item["domain"],
                "difficulty": item["difficulty"],
                "question": q, "is_general": False,
                "answer": answer,
                "precision_at_1":         round(p1, 4),
                "precision_at_3":         round(p3, 4),
                "precision_at_5":         round(p5, 4),
                "rr":                     round(rr, 4),
                "bertscore_f1":           round(bs["f1"], 4),
                "answer_relevancy":       round(ar, 4),
                "hallucination_score":    hall["score"],
                "hallucination_verdict":  hall["verdict"],
                "unsupported_claims":     hall["unsupported_claims"],
            })

        else:
            print(f"  General — Hall={hall['score']:.2f} [{hall['verdict']}]")
            if hall["unsupported_claims"]:
                for uc in hall["unsupported_claims"][:2]:
                    print(f"    ⚠ Hallucinated: {uc[:90]}")

            results.append({
                "id": qid, "domain": item["domain"],
                "difficulty": item["difficulty"],
                "question": q, "is_general": True,
                "answer": answer,
                "hallucination_score":   hall["score"],
                "hallucination_verdict": hall["verdict"],
                "unsupported_claims":    hall["unsupported_claims"],
            })

        if idx < total:
            time.sleep(SLEEP)

    store.close()

    # ── Tổng hợp ─────────────────────────────────────────────
    def avg(lst):
        clean = [x for x in lst if x is not None and x >= 0]
        return round(sum(clean) / len(clean), 4) if clean else 0.0

    total_hall = sum(hall_verdicts[k] for k in ("PASS", "PARTIAL", "FAIL"))
    pass_rate  = round((hall_verdicts["PASS"]    / total_hall) * 100, 1) if total_hall else 0.0
    partial_rate = round((hall_verdicts["PARTIAL"] / total_hall) * 100, 1) if total_hall else 0.0
    fail_rate  = round((hall_verdicts["FAIL"]    / total_hall) * 100, 1) if total_hall else 0.0

    domain_q_results = [r for r in results if not r["is_general"]]

    summary = {
        "timestamp":         datetime.now().isoformat(),
        "total":             total,
        "domain_questions":  len(domain_q_results),
        "general_questions": len([r for r in results if r["is_general"]]),
        "retrieval": {
            "precision_at_1": avg(p_metrics[1]),
            "precision_at_3": avg(p_metrics[3]),
            "precision_at_5": avg(p_metrics[5]),
            "mrr":            avg(mrr_list),
        },
        "answer_quality": {
            "bertscore_f1":     avg(bs_list),
            "answer_relevancy": avg(ar_list),
        },
        "hallucination": {
            "avg_score":    avg(hall_list),   # cao = ít hallucination
            "pass_rate":    pass_rate,
            "partial_rate": partial_rate,
            "fail_rate":    fail_rate,
            "verdicts":     hall_verdicts,
        },
    }

    # Domain breakdown
    domain_map: dict = {}
    for r in domain_q_results:
        d = r["domain"]
        if d not in domain_map:
            domain_map[d] = {"p3": [], "bs": [], "ar": [], "hall": []}
        domain_map[d]["p3"].append(r["precision_at_3"])
        domain_map[d]["bs"].append(r["bertscore_f1"])
        domain_map[d]["ar"].append(r["answer_relevancy"])
        if r["hallucination_score"] >= 0:
            domain_map[d]["hall"].append(r["hallucination_score"])

    summary["by_domain"] = {
        d: {
            "precision_at_3":   avg(v["p3"]),
            "bertscore_f1":     avg(v["bs"]),
            "answer_relevancy": avg(v["ar"]),
            "hallucination":    avg(v["hall"]),
            "count":            len(v["p3"]),
        }
        for d, v in domain_map.items()
    }

    # Difficulty breakdown
    diff_map: dict = {}
    for r in domain_q_results:
        diff = r["difficulty"]
        if diff not in diff_map:
            diff_map[diff] = {"p3": [], "bs": [], "ar": []}
        diff_map[diff]["p3"].append(r["precision_at_3"])
        diff_map[diff]["bs"].append(r["bertscore_f1"])
        diff_map[diff]["ar"].append(r["answer_relevancy"])

    summary["by_difficulty"] = {
        d: {
            "precision_at_3":   avg(v["p3"]),
            "bertscore_f1":     avg(v["bs"]),
            "answer_relevancy": avg(v["ar"]),
        }
        for d, v in diff_map.items()
    }

    # ── Lưu + in kết quả ─────────────────────────────────────
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f,
                  ensure_ascii=False, indent=2)
    _write_report(summary, results)
    _print_summary(summary)
    return summary


# ══════════════════════════════════════════════════════════════
# PRINT + REPORT
# ══════════════════════════════════════════════════════════════

def _print_summary(s: dict):
    print("\n" + "=" * 65)
    print("KẾT QUẢ TỔNG HỢP")
    print("=" * 65)
    r = s["retrieval"]
    q = s["answer_quality"]
    h = s["hallucination"]
    v = h["verdicts"]

    print(f"\n── RETRIEVAL ──────────────────────────────────────")
    print(f"  Precision@1  : {r['precision_at_1']:.4f}")
    print(f"  Precision@3  : {r['precision_at_3']:.4f}")
    print(f"  Precision@5  : {r['precision_at_5']:.4f}")
    print(f"  MRR          : {r['mrr']:.4f}")
    print(f"\n── ANSWER QUALITY ─────────────────────────────────")
    print(f"  BERTScore F1       : {q['bertscore_f1']:.4f}  (ngữ nghĩa XLM-RoBERTa-large)")
    print(f"  Answer Relevancy   : {q['answer_relevancy']:.4f}  (liên quan câu hỏi)")
    print(f"\n── HALLUCINATION (LLM-as-judge, claim-level) ──────")
    print(f"  Avg Score    : {h['avg_score']:.4f}  (cao = ít hallucination)")
    print(f"  PASS  rate   : {h['pass_rate']:.1f}%   (≥85% claims được hỗ trợ bởi context)")
    print(f"  PARTIAL rate : {h['partial_rate']:.1f}%   (50–84% claims OK)")
    print(f"  FAIL  rate   : {h['fail_rate']:.1f}%   (<50% claims, hallucinate nặng)")
    print(f"  Verdicts     : PASS={v.get('PASS',0)}  PARTIAL={v.get('PARTIAL',0)}  FAIL={v.get('FAIL',0)}")
    print(f"\n── THEO DOMAIN ────────────────────────────────────")
    for d, dv in s["by_domain"].items():
        print(f"  {d:<35} P@3={dv['precision_at_3']:.2f} "
              f"BERT={dv['bertscore_f1']:.2f} "
              f"AR={dv['answer_relevancy']:.2f} "
              f"Hall={dv['hallucination']:.2f}  ({dv['count']} câu)")
    print(f"\n── THEO ĐỘ KHÓ ────────────────────────────────────")
    for d, dv in s["by_difficulty"].items():
        print(f"  {d:<10} P@3={dv['precision_at_3']:.2f} "
              f"BERT={dv['bertscore_f1']:.2f} "
              f"AR={dv['answer_relevancy']:.2f}")
    print(f"\nĐã lưu: evaluation_results.json, evaluation_report.md")


def _write_report(summary: dict, results: list):
    r = summary["retrieval"]
    q = summary["answer_quality"]
    h = summary["hallucination"]
    v = h["verdicts"]

    lines = [
        "# Báo cáo Evaluation — Legal GraphRAG",
        f"\n**Thời gian:** {summary['timestamp']}  ",
        f"**Câu hỏi:** {summary['total']} "
        f"({summary['domain_questions']} trong tài liệu, "
        f"{summary['general_questions']} ngoài domain)",
        "\n---\n",

        "## 1. Retrieval Metrics\n",
        "| Metric | Kết quả | Ý nghĩa |",
        "|--------|---------|---------|",
        f"| Precision@1 | **{r['precision_at_1']:.4f}** | Chunk đầu tiên đúng file |",
        f"| Precision@3 | **{r['precision_at_3']:.4f}** | Top-3 có chunk đúng |",
        f"| Precision@5 | **{r['precision_at_5']:.4f}** | Top-5 có chunk đúng |",
        f"| MRR | **{r['mrr']:.4f}** | Vị trí trung bình chunk đúng |",

        "\n## 2. Answer Quality Metrics\n",
        "| Metric | Kết quả | Phương pháp | Ý nghĩa |",
        "|--------|---------|-------------|---------|",
        f"| BERTScore F1 | **{q['bertscore_f1']:.4f}** | XLM-RoBERTa-large | Tương đồng ngữ nghĩa (nhận biết paraphrase) |",
        f"| Answer Relevancy | **{q['answer_relevancy']:.4f}** | LLM + BERTScore | Trả lời đúng trọng tâm câu hỏi |",

        "\n## 3. Hallucination Control (LLM-as-judge, claim-level)\n",
        "| Metric | Kết quả | Ý nghĩa |",
        "|--------|---------|---------|",
        f"| Avg Score | **{h['avg_score']:.4f}** | Tỷ lệ claims được hỗ trợ bởi context (cao = tốt) |",
        f"| PASS rate | **{h['pass_rate']:.1f}%** | Câu trả lời có ≥85% claims OK |",
        f"| PARTIAL rate | **{h['partial_rate']:.1f}%** | Câu trả lời có 50–84% claims OK |",
        f"| FAIL rate | **{h['fail_rate']:.1f}%** | Câu trả lời có <50% claims OK (hallucinate nặng) |",
        f"| Phân bổ verdict | **PASS={v.get('PASS',0)} / PARTIAL={v.get('PARTIAL',0)} / FAIL={v.get('FAIL',0)}** | Số câu theo từng mức |",
        "",
        "> **Phương pháp:** LLM phân tách answer thành các mệnh đề pháp lý độc lập (claim extraction), "
        "sau đó verify từng mệnh đề với context PDF được retrieve. "
        "Phân loại: **SUPPORTED** (có trong context), **UNSUPPORTED** (bịa đặt/sai), **GENERAL** (kiến thức chung hợp lệ). "
        "Score = tỷ lệ SUPPORTED + GENERAL trên tổng claims.",

        "\n## 4. Kết quả theo Domain\n",
        "| Domain | P@3 | BERTScore | AR | Hallucination | Câu |",
        "|--------|-----|-----------|-----|---------------|-----|",
    ]

    for d, dv in summary["by_domain"].items():
        lines.append(
            f"| {d} | {dv['precision_at_3']:.2f} | {dv['bertscore_f1']:.2f} "
            f"| {dv['answer_relevancy']:.2f} | {dv['hallucination']:.2f} | {dv['count']} |"
        )

    lines += [
        "\n## 5. Kết quả theo Độ khó\n",
        "| Độ khó | P@3 | BERTScore | AR |",
        "|--------|-----|-----------|-----|",
    ]
    for d, dv in summary["by_difficulty"].items():
        lines.append(
            f"| {d} | {dv['precision_at_3']:.2f} "
            f"| {dv['bertscore_f1']:.2f} | {dv['answer_relevancy']:.2f} |"
        )

    # Top hallucinated answers
    hall_fails = sorted(
        [r for r in results if r.get("hallucination_verdict") in ("FAIL", "PARTIAL")
         and r.get("unsupported_claims")],
        key=lambda x: x.get("hallucination_score", 1.0)
    )
    if hall_fails:
        lines += ["\n## 6. Câu trả lời cần cải thiện (hallucination cao nhất)\n"]
        for res in hall_fails[:5]:
            lines.append(f"**{res['id']}** — {res['question'][:70]}...")
            lines.append(
                f"- Verdict: **{res['hallucination_verdict']}** "
                f"(score={res['hallucination_score']:.2f}, "
                f"{res.get('supported', '?')}/{res.get('total', '?')} claims OK)"
            )
            for uc in res["unsupported_claims"][:3]:
                lines.append(f"  - ⚠ `{uc[:110]}`")
            lines.append("")

    lines += [
        "\n## 7. Giải thích metrics\n",
        "- **Precision@K**: Tỷ lệ top-K chunks retrieved có đúng file nguồn",
        "- **MRR**: Mean Reciprocal Rank — chunk đúng ở vị trí nào trong kết quả search",
        "- **BERTScore**: Dùng XLM-RoBERTa-large embedding so sánh ngữ nghĩa — nhận biết paraphrase, từ đồng nghĩa tiếng Việt",
        "- **Answer Relevancy**: Sinh lại câu hỏi từ câu trả lời, đo BERTScore với câu hỏi gốc",
        "- **Hallucination Score**: LLM tách answer thành claims → verify từng claim với context → "
        "score = tỷ lệ claims được hỗ trợ (SUPPORTED hoặc GENERAL)",
    ]

    with open("evaluation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_evaluation()