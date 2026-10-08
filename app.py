# app.py
from __future__ import annotations

from flask import Flask, jsonify, render_template, request, Response, stream_with_context
from flask_cors import CORS
import os, json
from werkzeug.utils import secure_filename

from ingest_service import IngestService
from neo4j_store import Neo4jStore
from rag_service import GraphRAGService
from memory_service import MemoryService  # Memory service

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'pdfs')
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# =========================
# INIT SERVICES
# =========================
store = Neo4jStore()
store.setup_schema()
ingest_service = IngestService(store)
rag_service = GraphRAGService(store)

# Two-Tier Memory: short-term RAM (30) + long-term Neo4j (unlimited)
memory_service = MemoryService(short_term_limit=30, store=store)

# =========================
# UI
# =========================
@app.route("/")
def index():
    return render_template("index.html")


# =========================
# SYSTEM
# =========================
@app.route("/api/health", methods=["GET"])
def health():
    try:
        return jsonify({
            "status": "ok",
            "stats": store.stats(),
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500


@app.route("/api/ingest", methods=["POST"])
def ingest():
    try:
        result = ingest_service.ingest_all_pdfs()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500


# =========================
# CHAT (RAG + MEMORY)
# =========================
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True) or {}
        question = data.get("question", "").strip()
        if not question:
            return jsonify({
                "status": "error",
                "message": "Thiếu câu hỏi",
                "answer": "",
                "chunks": [],
                "graph_facts": [],
                "answer_graph": {"nodes": [], "edges": []},
            }), 400

        # Gọi RAG + Memory
        result = rag_service.ask(question, memory=memory_service)

        # Memory sẽ được cập nhật bên trong ask()
        # không cần add thêm ở đây nữa

        return jsonify(result)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "answer": "Có lỗi xảy ra khi xử lý câu hỏi.",
            "chunks": [],
            "graph_facts": [],
            "answer_graph": {"nodes": [], "edges": []},
        }), 500


# =========================
# GRAPH API
# =========================
@app.route("/api/graph", methods=["GET"])
def get_graph():
    try:
        limit = request.args.get("limit", default=120, type=int)
        if limit <= 0:
            limit = 120

        data = store.get_graph_data(limit=limit)

        return jsonify({
            "status": "ok",
            "graph": data,
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500


# =========================
# ENTITY DETAIL (CLICK NODE)
# =========================
@app.route("/api/entity", methods=["GET"])
def get_entity():
    try:
        name = request.args.get("name", "").strip()

        if not name:
            return jsonify({
                "status": "error",
                "message": "Thiếu tên entity",
            }), 400

        data = store.get_entity_context(
            entity_name=name,
            limit=6
        )

        return jsonify({
            "status": "ok",
            "data": data,
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500


# =========================
# LẤY CHUNK THEO ID (CLICK SOURCE)
# =========================
@app.route("/api/chunk", methods=["GET"])
def get_chunk():
    try:
        chunk_id = request.args.get("id", "").strip()

        if not chunk_id:
            return jsonify({
                "status": "error",
                "message": "Thiếu chunk_id",
            }), 400

        with store.driver.session() as session:
            record = session.run(
                """
                MATCH (c:Chunk {chunk_id: $chunk_id})
                RETURN c.content AS content,
                       c.source AS source,
                       c.page AS page
                """,
                chunk_id=chunk_id,
            ).single()

        if not record:
            return jsonify({
                "status": "error",
                "message": "Không tìm thấy chunk",
            }), 404

        return jsonify({
            "status": "ok",
            "data": dict(record),
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500


# =========================
# MEMORY STATS (debug/demo)
# =========================
@app.route("/api/memory/stats", methods=["GET"])
def memory_stats():
    try:
        return jsonify({"status": "ok", "memory": memory_service.stats()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# =========================
# STREAMING CHAT
# =========================
@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    try:
        data = request.get_json(force=True) or {}
        question = data.get("question", "").strip()
        if not question:
            def err():
                yield f"data: {json.dumps({'error': 'Thiếu câu hỏi'})}\n\n"
            return Response(stream_with_context(err()), mimetype="text/event-stream")

        def generate():
            try:
                # Bước 1: lấy chunks + graph (nhanh)
                chunks = rag_service._get_chunks_for_question(question)
                answer_graph = rag_service._build_answer_graph(question)
                context = rag_service._build_context(chunks, max_chunks=5)

                # Gửi metadata trước
                meta = {
                    "type": "meta",
                    "chunks": chunks[:5],
                    "answer_graph": answer_graph,
                    "has_pdf_data": bool(chunks),
                }
                yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

                # Bước 2: stream từng token từ LLM
                from llm_functions import (
                    assess_context_relevance, _build_system_prompt,
                    _build_pdf_prompt, _build_general_prompt,
                    RELEVANCE_SCORE_THRESHOLD, MIN_USEFUL_CONTEXT_CHARS,
                    MODEL_CANDIDATES, client
                )
                from google.genai import types

                if chunks:
                    rel = assess_context_relevance(chunks, question)
                    has_data = rel["has_data"]
                    confidence = rel["confidence"]
                else:
                    has_data = False
                    confidence = "low"

                system_prompt = _build_system_prompt(memory_service, question=question)

                if has_data and context.strip():
                    user_prompt = _build_pdf_prompt(context, [], question, confidence)
                    temperature = 0.05
                else:
                    user_prompt = _build_general_prompt(question)
                    temperature = 0.25

                full_answer = ""
                for model in MODEL_CANDIDATES:
                    try:
                        cfg = types.GenerateContentConfig(
                            temperature=temperature,
                            max_output_tokens=3000,
                            system_instruction=system_prompt,
                        )
                        for chunk_resp in client.models.generate_content_stream(
                            model=model, contents=user_prompt, config=cfg
                        ):
                            token = (chunk_resp.text or "")
                            if token:
                                full_answer += token
                                yield f"data: {json.dumps({'type': 'token', 'text': token}, ensure_ascii=False)}\n\n"
                        break
                    except Exception as e:
                        print(f"[STREAM] Model {model} error: {e}")
                        continue

                # Lưu memory sau khi stream xong
                if full_answer:
                    memory_service.add(question, full_answer)

                yield f"data: {json.dumps({'type': 'done', 'has_pdf_data': has_data}, ensure_ascii=False)}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# UPLOAD PDF
# =========================
@app.route("/api/upload", methods=["POST"])
def upload_pdf():
    try:
        if 'files' not in request.files:
            return jsonify({"status": "error", "message": "Không có file"}), 400

        files = request.files.getlist('files')
        uploaded, skipped, errors = [], [], []

        for file in files:
            if not file or file.filename == '':
                continue
            if not allowed_file(file.filename):
                skipped.append(file.filename)
                continue
            filename = secure_filename(file.filename)
            dest = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(dest)
            uploaded.append(filename)

        return jsonify({
            "status": "ok",
            "uploaded": uploaded,
            "skipped": skipped,
            "message": f"Đã lưu {len(uploaded)} file. Bấm 'Nạp toàn bộ PDF' để ingest.",
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/pdfs", methods=["GET"])
def list_pdfs():
    try:
        pdf_dir = app.config['UPLOAD_FOLDER']
        files = []
        for f in sorted(os.listdir(pdf_dir)):
            if f.lower().endswith('.pdf'):
                path = os.path.join(pdf_dir, f)
                files.append({
                    "name": f,
                    "size_kb": round(os.path.getsize(path) / 1024, 1),
                })
        return jsonify({"status": "ok", "files": files})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/pdfs/<filename>", methods=["DELETE"])
def delete_pdf(filename):
    try:
        filename = secure_filename(filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(path):
            return jsonify({"status": "error", "message": "File không tồn tại"}), 404
        os.remove(path)
        return jsonify({"status": "ok", "message": f"Đã xoá {filename}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)