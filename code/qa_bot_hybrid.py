"""qa_bot_hybrid.py —— 向量 + BM25 混合检索问答机器人（RRF 融合）。

运行方式：
    cd rag-tool-agent-course
    python code/qa_bot_hybrid.py

前置条件：
    1. 先运行 make_hybrid_database.py 构建混合数据库
    2. 设置 DEEPSEEK_API_KEY（或在 code/.env 中写入）

退出方式：输入 quit / exit / q
"""

from __future__ import annotations

import json
import math
import os
import pickle
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import faiss
import jieba
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ── 路径配置 ──

COURSE_ROOT = Path(__file__).resolve().parent.parent
if not (COURSE_ROOT / "code").is_dir():
    COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

CODE_ROOT = COURSE_ROOT / "code"
MODEL_DIR = COURSE_ROOT / "open_models" / "bge-m3"
KB_DIR = CODE_ROOT / "hybrid_kb"

FAISS_PATH = KB_DIR / "faiss.index"
JSONL_PATH = KB_DIR / "metadata.jsonl"
BM25_PATH = KB_DIR / "bm25_index.pkl"

# ── 读取 .env ──

_ENV_FILE = CODE_ROOT / ".env"
if _ENV_FILE.exists():
    with open(_ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
CHAT_URL = BASE_URL + "/v1/chat/completions"
LLM_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
MAX_TOKENS = int(os.getenv("QA_MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("QA_TEMPERATURE", "0.3"))
USER_ID = os.getenv("QA_USER_ID", os.getenv("USERNAME", os.getenv("USER", "user"))).strip()
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))

SYSTEM_PROMPT = (
    "你是 通用客服助手，回答简洁专业、条理清晰。"
    "请根据提供的参考资料回答问题。如果参考资料不足以回答，请诚实说明。"
    "涉及业务规则时注明以官方页面为准。"
)

console = Console()


# ── 前置检查 ──

def _check_prerequisites():
    if not API_KEY:
        console.print(Panel.fit(
            "[bold red]未设置 DEEPSEEK_API_KEY[/bold red]\n"
            "方式一: export DEEPSEEK_API_KEY=your_api_key_here\n"
            "方式二: 在 code/.env 中写入 DEEPSEEK_API_KEY=your_api_key_here",
            title="错误", border_style="red",
        ))
        raise SystemExit(1)

    if not MODEL_DIR.is_dir():
        console.print(Panel.fit(
            f"[bold red]未找到 BGE-m3 模型[/bold red]\n路径: {MODEL_DIR}",
            title="错误", border_style="red",
        ))
        raise SystemExit(1)

    if not (FAISS_PATH.exists() and JSONL_PATH.exists() and BM25_PATH.exists()):
        console.print(Panel.fit(
            "[bold yellow]未找到混合知识库[/bold yellow]\n"
            f"请先运行: python code/make_hybrid_database.py\n"
            f"期望路径: {KB_DIR.relative_to(COURSE_ROOT)}/",
            title="提示", border_style="yellow",
        ))
        raise SystemExit(1)


_check_prerequisites()


# ── 加载知识库 ──

console.print("[dim]加载知识库...[/dim]", end="")
_start = time.perf_counter()

# FAISS 向量索引
index = faiss.read_index(str(FAISS_PATH))

# 元数据
metadata = []
for line in JSONL_PATH.read_text(encoding="utf-8").splitlines():
    if line.strip():
        metadata.append(json.loads(line))

# BM25 索引
with BM25_PATH.open("rb") as f:
    bm25_data = pickle.load(f)
    inverted_index = bm25_data["inverted_index"]
    idf_cache = bm25_data["idf_cache"]
    N = bm25_data["N"]
    avgdl = bm25_data["avgdl"]
    k1 = bm25_data["k1"]
    b = bm25_data["b"]

# BGE-m3 模型
model = SentenceTransformer(str(MODEL_DIR), device="cpu")

_load_ms = (time.perf_counter() - _start) * 1000
console.print(f" 完成 ({_load_ms:.0f}ms, {index.ntotal} 条记录)")


# ── 分词 ──

def tokenize(text: str) -> list[str]:
    """使用 jieba 进行中文分词，过滤短词。"""
    return [w.strip() for w in jieba.cut(text) if len(w.strip()) >= 2]


# ── BM25 检索 ──

def bm25_search(query: str, top_k: int = TOP_K) -> list[tuple[float, dict]]:
    """BM25 关键词检索，返回 top_k 条结果。"""
    tokens = tokenize(query)
    scores = []
    for chunk_id, chunk in enumerate(metadata):
        content = chunk["content"]
        doc_tokens = tokenize(content)
        doc_len = len(doc_tokens)
        score = 0.0
        for term in tokens:
            if term not in idf_cache:
                continue
            tf = doc_tokens.count(term)
            if tf == 0:
                continue
            idf_val = idf_cache[term]
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avgdl)
            score += idf_val * numerator / denominator
        scores.append((score, chunk))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]


# ── 向量检索 ──

def vector_search(query: str, top_k: int = TOP_K) -> list[tuple[float, dict]]:
    """BGE-m3 向量检索，返回 top_k 条最相似文档。"""
    query_vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    query_vec = np.asarray(query_vec, dtype="float32")
    scores, row_ids = index.search(query_vec, top_k)
    results = []
    for score, row_id in zip(scores[0], row_ids[0]):
        if row_id < 0:
            continue
        results.append((float(score), metadata[int(row_id)]))
    return results


# ── RRF 融合 ──

def rrf_fusion(bm25_results: list[tuple[float, dict]], 
               vector_results: list[tuple[float, dict]], 
               k: int = RRF_K) -> list[tuple[float, dict]]:
    """RRF (Reciprocal Rank Fusion) 融合两路检索结果。"""
    scores = Counter()
    doc_map = {}
    # BM25 结果
    for rank, (score, doc) in enumerate(bm25_results, 1):
        doc_id = id(doc)
        scores[doc_id] += 1.0 / (k + rank)
        doc_map[doc_id] = doc
    # 向量结果
    for rank, (score, doc) in enumerate(vector_results, 1):
        doc_id = id(doc)
        scores[doc_id] += 1.0 / (k + rank)
        doc_map[doc_id] = doc
    # 按 RRF 分数排序
    fused = []
    for doc_id, rrf_score in scores.most_common():
        fused.append((rrf_score, doc_map[doc_id]))
    return fused


# ── 混合检索 ──

def hybrid_search(query: str, top_k: int = TOP_K) -> list[tuple[float, dict]]:
    """混合检索：BM25 + 向量 + RRF 融合。"""
    bm25_results = bm25_search(query, top_k=top_k * 2)
    vector_results = vector_search(query, top_k=top_k * 2)
    fused = rrf_fusion(bm25_results, vector_results, k=RRF_K)
    return fused[:top_k]


# ── Prompt 构建 ──

def build_messages(question: str, retrieved: list[tuple[float, dict]]) -> list[dict]:
    """将检索到的文档拼入 prompt。"""
    context_parts = []
    for i, (score, doc) in enumerate(retrieved, 1):
        context_parts.append(
            f"[{i}] {doc['title']} (来源: {doc['source']}, RRF分数: {score:.4f})\n"
            f"{doc['content']}"
        )
    context = "\n\n".join(context_parts)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}\n\n请基于以上参考资料回答。如果资料不足以回答，请诚实说明。"},
    ]


# ── API 调用 ──

def call_llm(messages: list[dict]) -> str:
    """调用 DeepSeek API（流式）。"""
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "top_p": 0.9,
        "stream": True,
    }
    if USER_ID:
        payload["user"] = USER_ID
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    resp = requests.post(CHAT_URL, headers=headers, json=payload, timeout=300, stream=True)
    resp.raise_for_status()

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            if chunk.get("choices"):
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("content"):
                    yield delta["content"]
        except json.JSONDecodeError:
            pass


# ── 显示检索结果表格 ──

def show_retrieved(retrieved: list[tuple[float, dict]], search_ms: float):
    tbl = Table(title=f"混合检索结果 ({search_ms:.0f}ms)", show_header=True, header_style="dim")
    tbl.add_column("#", width=3)
    tbl.add_column("RRF分数", width=10)
    tbl.add_column("标题", width=20)
    tbl.add_column("来源", width=32)
    for i, (score, doc) in enumerate(retrieved, 1):
        tbl.add_row(str(i), f"{score:.4f}", doc["title"][:18], doc["source"][:50])
    console.print(tbl)


# ── TUI 主循环 ──

def main():
    global LLM_MODEL, TEMPERATURE

    console.print(Panel.fit(
        f"[bold]示例业务系统 混合检索问答机器人[/bold]\n"
        f"后端: DeepSeek ({LLM_MODEL})  |  知识库: {index.ntotal} 条  |  top_k={TOP_K}\n"
        f"检索方式: BM25 + 向量 + RRF 融合  |  temperature={TEMPERATURE}",
        border_style="cyan",
    ))
    console.print("[dim]quit/exit/q 退出  |  clear 清空历史  |  /model 切换模型  |  /temp 调温度  |  /sources 查看上次检索来源[/dim]\n")

    history = []
    last_retrieved = []

    while True:
        try:
            user_input = input("🧑 你：").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]再见！[/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]再见！[/dim]")
            break

        if user_input.lower() == "clear":
            history = []
            last_retrieved = []
            console.print("[green]✓ 对话历史已清空[/green]\n")
            continue

        # / 命令
        if user_input.startswith("/model"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                new_model = parts[1].strip()
                if new_model in ("deepseek-chat", "deepseek-reasoner"):
                    LLM_MODEL = new_model
                    console.print(f"[green]✓ 已切换模型: {LLM_MODEL}[/green]\n")
                else:
                    console.print(f"[red]未知模型: {new_model}，可选: deepseek-chat / deepseek-reasoner[/red]\n")
            else:
                console.print(f"当前模型: {LLM_MODEL}\n")
            continue

        if user_input.startswith("/temp"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                try:
                    t = float(parts[1].strip())
                    if 0 <= t <= 2:
                        TEMPERATURE = t
                        console.print(f"[green]✓ temperature = {TEMPERATURE}[/green]\n")
                    else:
                        console.print("[red]temperature 范围 0~2[/red]\n")
                except ValueError:
                    console.print("[red]temperature 必须是数字[/red]\n")
            else:
                console.print(f"当前 temperature: {TEMPERATURE}\n")
            continue

        if user_input.lower() == "/sources":
            if not last_retrieved:
                console.print("[dim]尚无检索记录[/dim]\n")
            else:
                tbl = Table(title="上次检索来源", show_header=True, header_style="dim")
                tbl.add_column("#", width=3)
                tbl.add_column("RRF分数", width=10)
                tbl.add_column("标题", width=20)
                tbl.add_column("文件位置", width=55)
                for i, (score, doc) in enumerate(last_retrieved, 1):
                    tbl.add_row(str(i), f"{score:.4f}", doc["title"][:18], doc["source"])
                console.print(tbl)
                console.print()
            continue

        # ── 混合检索 ──
        _t0 = time.perf_counter()
        last_retrieved = hybrid_search(user_input)
        search_ms = (time.perf_counter() - _t0) * 1000

        show_retrieved(last_retrieved, search_ms)

        # ── 构建 Prompt + 调用 LLM ──
        prompt_msgs = build_messages(user_input, last_retrieved)
        full_messages = history[-6:] + prompt_msgs

        try:
            full_answer = ""
            console.print("🤖 客服：", end="")
            for piece in call_llm(full_messages):
                full_answer += piece
                console.print(piece, end="")
            console.print()
            console.print()

            if full_answer:
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": full_answer})

        except requests.Timeout:
            console.print("\n[red]请求超时，请重试[/red]\n")
        except requests.HTTPError as e:
            console.print(f"\n[red]API 错误: {e.response.status_code} {e.response.text[:200]}[/red]\n")
        except Exception as e:
            console.print(f"\n[red]异常: {type(e).__name__}: {e}[/red]\n")


if __name__ == "__main__":
    main()