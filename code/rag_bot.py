"""RAG 知识库聊天机器人 —— 检索增强 + DeepSeek API + rich TUI。

前置条件：
  1. 先运行 rag_build.py 构建知识库（code/rag_kb/）
  2. 设置 DEEPSEEK_API_KEY（或在 code/.env 中写入）

运行方式：
  python code/rag_bot.py

退出方式：输入 quit / exit / q
"""

import json
import os
import sys
import time
from pathlib import Path

# 读取 .env
_ENV_FILE = Path(__file__).resolve().parent / ".env"
if _ENV_FILE.exists():
    with open(_ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

COURSE_ROOT = Path(__file__).resolve().parent.parent
if not (COURSE_ROOT / "code").is_dir():
    COURSE_ROOT = Path.cwd().resolve()

CODE_ROOT = COURSE_ROOT / "code"
MODEL_DIR = COURSE_ROOT / "open_models" / "bge-m3"
KB_DIR = CODE_ROOT / "rag_kb"

FAISS_PATH = KB_DIR / "faiss.index"
JSONL_PATH = KB_DIR / "metadata.jsonl"

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
CHAT_URL = BASE_URL + "/v1/chat/completions"
LLM_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
MAX_TOKENS = int(os.getenv("QA_MAX_TOKENS", "800"))
TEMPERATURE = float(os.getenv("QA_TEMPERATURE", "0.3"))
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
USER_ID = os.getenv("DEEPSEEK_USER_ID", os.getenv("USERNAME", os.getenv("USER", "user"))).strip()

SYSTEM_PROMPT = (
    "你是 通用客服助手，回答简洁专业、条理清晰。"
    "请根据提供的参考资料回答问题。如果参考资料不足以回答，请诚实说明并建议用户咨询官方渠道。"
    "涉及业务规则时注明以官方页面为准。"
)

console = Console()

# ── 检查前置条件 ──

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

if not (FAISS_PATH.exists() and JSONL_PATH.exists()):
    console.print(Panel.fit(
        "[bold yellow]未找到知识库[/bold yellow]\n"
        f"请先运行: python code/rag_build.py\n"
        f"期望路径: {KB_DIR.relative_to(COURSE_ROOT)}/",
        title="提示", border_style="yellow",
    ))
    raise SystemExit(1)


# ── 加载知识库 ──

console.print("[dim]加载知识库...[/dim]", end="")
_start = time.perf_counter()
index = faiss.read_index(str(FAISS_PATH))
metadata = []
for line in JSONL_PATH.read_text(encoding="utf-8").splitlines():
    if line.strip():
        metadata.append(json.loads(line))
model = SentenceTransformer(str(MODEL_DIR), device="cpu")
console.print(f" 完成 ({(time.perf_counter() - _start) * 1000:.0f}ms, {index.ntotal} 条记录)")


def search(query: str, top_k: int = TOP_K) -> list[dict]:
    """向量检索，返回 top_k 条最相似文档。"""
    vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    vec = np.asarray(vec, dtype="float32")
    scores, row_ids = index.search(vec, top_k)
    results = []
    for score, row_id in zip(scores[0], row_ids[0]):
        if row_id < 0:
            continue
        row = dict(metadata[int(row_id)])
        row["score"] = float(score)
        results.append(row)
    return results


def build_prompt(question: str, retrieved: list[dict]) -> list[dict]:
    """将检索到的文档拼入 prompt。"""
    context_parts = []
    for i, doc in enumerate(retrieved, 1):
        context_parts.append(f"[{i}] {doc['title']} (来源: {doc['source']}, 相关度: {doc['score']:.3f})\n{doc['content']}")
    context = "\n\n".join(context_parts)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}\n\n请基于以上参考资料回答。如果资料不足以回答，请诚实说明。"},
    ]


def call_llm(messages: list[dict]) -> str:
    """调用 DeepSeek API（流式）。"""
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "top_p": 0.9,
        "stream": True,
        "user_id": USER_ID,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    resp = requests.post(CHAT_URL, headers=headers, json=payload, timeout=300, stream=True)
    resp.raise_for_status()

    full = ""
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                piece = delta["content"]
                full += piece
                yield piece
        except json.JSONDecodeError:
            pass
    return full


# ── TUI 主循环 ──

def main():
    console.print(Panel.fit(
        f"[bold]RAG 知识库问答[/bold]\n"
        f"后端: DeepSeek ({LLM_MODEL})  |  向量库: {index.ntotal} 条  |  top_k={TOP_K}\n"
        f"temperature={TEMPERATURE}  max_tokens={MAX_TOKENS}",
        border_style="cyan",
    ))
    console.print("[dim]quit/exit/q 退出  |  clear 清空历史  |  /sources 查看引用来源[/dim]\n")

    messages = []  # 对话历史（不含检索上下文，每次提问重新检索）

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
            messages = []
            console.print("[green]✓ 对话历史已清空[/green]\n")
            continue

        if user_input.lower() == "/sources":
            if not messages:
                console.print("[dim]暂无对话[/dim]\n")
            else:
                console.print(f"[dim]已检索 {len(messages)} 轮对话[/dim]\n")
            continue

        # ── 检索 ──
        _t0 = time.perf_counter()
        retrieved = search(user_input)
        search_ms = (time.perf_counter() - _t0) * 1000

        # ── 显示检索结果 ──
        tbl = Table(title=f"检索结果 ({search_ms:.0f}ms)", show_header=True, header_style="dim")
        tbl.add_column("#", width=3)
        tbl.add_column("相关度", width=8)
        tbl.add_column("标题", width=18)
        tbl.add_column("内容摘要", width=55)
        for i, doc in enumerate(retrieved, 1):
            preview = doc["content"].replace("\n", " ")[:70]
            tbl.add_row(str(i), f"{doc['score']:.3f}", doc["title"][:16], preview)
        console.print(tbl)

        # ── 构建 prompt + 调用 LLM ──
        prompt_msgs = build_prompt(user_input, retrieved)
        # 拼接历史（只保留最近 3 轮，控制 token）
        full_messages = messages[-6:] + prompt_msgs

        try:
            console.print("🤖 客服：", end="")
            full_answer = ""
            for piece in call_llm(full_messages):
                full_answer += piece
                console.print(piece, end="")
            console.print()
            console.print()
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": full_answer})
        except requests.Timeout:
            console.print("\n[red]请求超时，请重试[/red]\n")
        except requests.HTTPError as e:
            console.print(f"\n[red]API 错误: {e.response.status_code}[/red]\n")
        except Exception as e:
            console.print(f"\n[red]异常: {type(e).__name__}: {e}[/red]\n")


if __name__ == "__main__":
    main()