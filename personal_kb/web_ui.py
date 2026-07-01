"""web_ui —— Flask Web 界面。

提供：
    create_app(...) -> Flask
        创建 Flask 应用实例，注册路由：
          GET  /              - 聊天界面主页
          POST /api/chat      - 发送消息并流式获取回答
          POST /api/update    - 触发向量数据库更新
          GET  /api/status    - 获取知识库统计信息

运行：
    python -m personal_kb.web_ui
    或通过 run_kb.py 启动。
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from .rag_engine import KnowledgeBaseRAG
from .vectorstore import build_vectorstore

COURSE_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = COURSE_ROOT / "open_models" / "bge-m3"
KNOWLEDGE_DIR = COURSE_ROOT / "knowledge"
DB_DIR = COURSE_ROOT / "kb_data"

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
LLM_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
TEMPERATURE = float(os.getenv("QA_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("QA_MAX_TOKENS", "1024"))
TOP_K = int(os.getenv("RAG_TOP_K", "5"))

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

rag: KnowledgeBaseRAG | None = None
_build_lock = threading.Lock()
_build_status: dict = {"running": False, "result": None}


def _get_rag() -> KnowledgeBaseRAG:
    global rag
    if rag is None:
        rag = KnowledgeBaseRAG(
            db_dir=DB_DIR,
            model_dir=MODEL_DIR,
            api_key=API_KEY,
            base_url=BASE_URL,
            llm_model=LLM_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_k=TOP_K,
        )
        rag.load()
    return rag


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(TEMPLATES_DIR))

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/chat", methods=["POST"])
    def api_chat():
        data = request.get_json(force=True)
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"error": "问题不能为空"}), 400

        history = data.get("history", [])

        r = _get_rag()
        retrieved = r.search(question)

        def generate():
            buffer: list[str] = []
            for piece in r.ask_stream(question, history=history):
                buffer.append(piece)
                yield f"data: {json.dumps({'token': piece, 'type': 'token'})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'sources': retrieved})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.route("/api/update", methods=["POST"])
    def api_update():
        global _build_status

        if _build_status["running"]:
            return jsonify({"status": "running", "message": "数据库更新正在进行中，请稍后再试。"})

        def run_build():
            global _build_status
            with _build_lock:
                _build_status["running"] = True
                _build_status["result"] = None
                try:
                    result = build_vectorstore(
                        knowledge_dir=KNOWLEDGE_DIR,
                        db_dir=DB_DIR,
                        model_dir=MODEL_DIR,
                    )
                    _build_status["result"] = result
                except Exception as e:
                    _build_status["result"] = {"status": "error", "message": str(e)}
                finally:
                    _build_status["running"] = False
                    global rag
                    rag = None

        _build_status["running"] = True
        _build_status["result"] = None
        thread = threading.Thread(target=run_build, daemon=True)
        thread.start()

        return jsonify({"status": "started", "message": "数据库更新已启动。"})

    @app.route("/api/update/status", methods=["GET"])
    def api_update_status():
        return jsonify({
            "running": _build_status["running"],
            "result": _build_status["result"],
        })

    @app.route("/api/status", methods=["GET"])
    def api_status():
        r = _get_rag()
        stats = r.get_stats()
        stats["loaded"] = r.is_loaded
        stats["model_available"] = MODEL_DIR.is_dir()
        stats["knowledge_dir_exists"] = KNOWLEDGE_DIR.is_dir()
        return jsonify(stats)

    return app


if __name__ == "__main__":
    app = create_app()
    print(f"知识库目录: {KNOWLEDGE_DIR}")
    print(f"向量库目录: {DB_DIR}")
    print(f"模型目录:   {MODEL_DIR}")
    app.run(host="0.0.0.0", port=7860, debug=True)