"""01 DeepSeek 多轮对话：纯对话模式。

这个脚本演示普通多轮对话：
1. 第一轮：用户问一个问题。
2. 模型回答，同时可能返回 reasoning_content。
3. 第二轮：用户追问。
4. 程序只把 assistant 的最终回答 content 放回历史，不把 reasoning_content 放回去。

DeepSeek 官方说明：
- API 是无状态的，每次请求都要显式传入历史 messages。
- 没有发生 tool call 的普通多轮对话，历史 reasoning_content 不需要传回。

运行方式：
    cd rag-tool-agent-course
    export DEEPSEEK_API_KEY=your_api_key_here
    python code/12_专题_多轮对话机制/code/01_DeepSeek多轮对话_纯对话模式.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("01 DeepSeek 多轮对话：纯对话模式")
print("=" * 72)

API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
URL = BASE_URL + "/chat/completions"

if not API_KEY:
    raise SystemExit(
        "未设置 DEEPSEEK_API_KEY。\n"
        "macOS/Linux: export DEEPSEEK_API_KEY=your_api_key_here\n"
        "Windows PowerShell: $env:DEEPSEEK_API_KEY=\"sk-xxx\""
    )

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

messages = [
    {
        "role": "system",
        "content": "你是一个培训课堂里的 AI 助手，回答要简洁、清楚，适合初学者理解。",
    }
]

questions = [
    "请用三句话解释什么是多轮对话。",
    "那为什么第二轮提问时，模型还能知道我刚才问的是多轮对话？",
]

for round_index, question in enumerate(questions, start=1):
    print(f"\n{'=' * 24} 第 {round_index} 轮 {'=' * 24}")
    print("用户：", question)

    messages.append({"role": "user", "content": question})

    payload = {
        "model": MODEL,
        "messages": messages,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
        "max_tokens": 900,
        "stream": False,
    }

    response = requests.post(URL, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        print("请求失败：")
        print(response.text[:2000])
        raise SystemExit(1)

    data = response.json()
    message = data["choices"][0]["message"]

    reasoning_content = message.get("reasoning_content") or ""
    content = message.get("content") or ""

    print("\n模型 reasoning_content（只展示，不放回下一轮）：")
    print(reasoning_content[:1000] if reasoning_content else "本轮没有返回 reasoning_content。")

    print("\n模型最终回答 content：")
    print(content)

    # 普通多轮对话只需要把最终回答放回历史。
    # 如果把 reasoning_content 也放回去，DeepSeek 普通对话会忽略它；
    # 课堂上建议先保持简单：历史里只保留用户问题和最终回答。
    messages.append({"role": "assistant", "content": content})

    print("\n当前传给下一轮的 messages：")
    print(json.dumps(messages, ensure_ascii=False, indent=2))

print("\n结论")
print("普通多轮对话的关键是自己维护 messages。")
print("没有 tool call 的情况下，历史里不需要保留 reasoning_content。")
