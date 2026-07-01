"""run_kb.py —— 个人知识库系统入口。

用法：
    # 1. 启动 Web 服务
    python run_kb.py

    # 2. 仅构建/更新向量数据库（不启动 Web）
    python run_kb.py --build

    # 3. 命令行问答模式
    python run_kb.py --ask "什么是候补申请？"

前置条件：
    1. 将 PDF / Markdown 文件放入 knowledge/ 目录
    2. 确保 open_models/bge-m3/ 存在 BGE-m3 模型
    3. 设置环境变量 DEEPSEEK_API_KEY（或在 code/.env 中写入）
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

COURSE_ROOT = Path(__file__).resolve().parent
if not (COURSE_ROOT / "code").is_dir() and not (COURSE_ROOT / "knowledge").is_dir():
    COURSE_ROOT = Path.cwd().resolve()

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(COURSE_ROOT))

from personal_kb.vectorstore import build_vectorstore, get_kb_stats
from personal_kb.rag_engine import KnowledgeBaseRAG

MODEL_DIR = COURSE_ROOT / "open_models" / "bge-m3"
KNOWLEDGE_DIR = COURSE_ROOT / "knowledge"
DB_DIR = COURSE_ROOT / "kb_data"

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
LLM_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
TEMPERATURE = float(os.getenv("QA_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("QA_MAX_TOKENS", "1024"))
TOP_K = int(os.getenv("RAG_TOP_K", "5"))


def _load_env():
    env_file = COURSE_ROOT / "code" / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def cmd_build():
    """构建/更新向量数据库。"""
    print("=" * 60)
    print("个人知识库 — 构建向量数据库")
    print("=" * 60)
    print(f"  知识源目录: {KNOWLEDGE_DIR}")
    print(f"  输出目录:   {DB_DIR}")
    print(f"  模型目录:   {MODEL_DIR}")
    print()

    result = build_vectorstore(
        knowledge_dir=KNOWLEDGE_DIR,
        db_dir=DB_DIR,
        model_dir=MODEL_DIR,
    )
    print(f"状态: {result.get('status')}")
    print(f"文件总数: {result.get('total_files', 0)}")
    print(f"新增/变更: {result.get('new_files', 0)}")
    print(f"总切片数: {result.get('total_chunks', 0)}")
    print(f"向量维度: {result.get('vector_dim', 'N/A')}")
    print(f"耗时: {result.get('build_time_ms', 0):.0f} ms")

    if result.get("status") == "error":
        print(f"错误信息: {result.get('message')}")


def cmd_ask(question: str):
    """命令行单次问答。"""
    if not API_KEY:
        print("[错误] 未设置 DEEPSEEK_API_KEY")
        sys.exit(1)

    if not MODEL_DIR.is_dir():
        print(f"[错误] 未找到 BGE-m3 模型: {MODEL_DIR}")
        sys.exit(1)

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

    if not rag.load():
        print("[提示] 知识库尚未构建，正在自动构建...")
        result = build_vectorstore(
            knowledge_dir=KNOWLEDGE_DIR,
            db_dir=DB_DIR,
            model_dir=MODEL_DIR,
        )
        print(f"  构建结果: {result.get('status')}, {result.get('total_chunks', 0)} 条切片")
        if not rag.load():
            print("[错误] 知识库构建后仍无法加载")
            sys.exit(1)

    retrieved = rag.search(question)
    print(f"\n检索到 {len(retrieved)} 条相关文档:")
    for i, doc in enumerate(retrieved, 1):
        print(f"  [{i}] {doc['title']} (相关度: {doc['score']:.3f}) — {doc['source']}")

    print(f"\n回答:\n")
    for piece in rag.ask_stream(question):
        print(piece, end="")
    print()


def cmd_status():
    """显示知识库状态。"""
    stats = get_kb_stats(DB_DIR)
    print("知识库状态:")
    print(f"  向量库存在: {stats.get('has_vectorstore', False)}")
    print(f"  元数据存在: {stats.get('has_metadata', False)}")
    print(f"  文件数:     {stats.get('file_count', 0)}")
    print(f"  切片数:     {stats.get('chunk_count', 0)}")
    print(f"  版本:       {stats.get('version', 'N/A')}")
    print(f"  知识源目录: {KNOWLEDGE_DIR} ('exists' if KNOWLEDGE_DIR.is_dir() else 'DOES NOT EXIST')")


def cmd_web():
    """启动 Web 服务。"""
    from personal_kb.web_ui import create_app

    print("=" * 60)
    print("个人知识库 — 启动 Web 服务")
    print("=" * 60)
    print(f"  访问地址: http://localhost:7860")
    print(f"  知识源目录: {KNOWLEDGE_DIR}")
    print(f"  向量库目录: {DB_DIR}")
    print()
    print("提示: 在浏览器中打开后，先点击「更新数据库」按钮构建向量库")
    print()

    app = create_app()
    app.run(host="0.0.0.0", port=7860, debug=False)


def main():
    _load_env()

    parser = argparse.ArgumentParser(
        description="个人知识库系统 — RAG 问答 + Web 界面",
    )
    parser.add_argument("--build", action="store_true", help="构建/更新向量数据库")
    parser.add_argument("--ask", type=str, default="", help="命令行问答")
    parser.add_argument("--status", action="store_true", help="显示知识库状态")
    parser.add_argument("--web", action="store_true", help="启动 Web 服务（默认行为）")

    args = parser.parse_args()

    if args.build:
        cmd_build()
    elif args.ask:
        cmd_ask(args.ask)
    elif args.status:
        cmd_status()
    else:
        cmd_web()


if __name__ == "__main__":
    main()