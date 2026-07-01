"""03 百炼 Qwen 多轮对话：thinking 是否需要传回。

本脚本验证两种写法：
1. 普通多轮：只传 user 和 assistant.content。
2. 延续思考：把 assistant.reasoning_content 放回 messages，并设置 preserve_thinking=True。

百炼官方文档说明：
- 通义千问 API 是无状态的，多轮对话需要显式传入历史 messages。
- 默认不会读取历史消息里的 reasoning_content。
- 如果希望模型参考上一轮思考过程，需要设置 preserve_thinking=True。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("03 百炼 Qwen 多轮对话：thinking 传回规则")
print("=" * 72)

API_BASE = os.getenv("BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("BAILIAN_MODEL", "qwen-plus")
API_KEY = os.getenv("BAILIAN_API_KEY", "").strip()
URL = API_BASE.rstrip("/") + "/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def call_qwen(messages: list[dict], preserve_thinking: bool = False) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "enable_thinking": True,
        "thinking_budget": 512,
        "max_tokens": 900,
        "stream": False,
    }
    if preserve_thinking:
        payload["preserve_thinking"] = True

    response = requests.post(URL, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        print("请求失败：")
        print(response.text[:2000])
        raise SystemExit(1)
    return response.json()


def print_message(data: dict) -> tuple[str, str]:
    message = data["choices"][0]["message"]
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""

    print("\nreasoning_content：")
    print(reasoning[:1200] if reasoning else "本轮没有返回 reasoning_content。")

    print("\ncontent：")
    print(content)

    return reasoning, content


print("\n" + "=" * 60)
print("一、普通多轮：不传回 reasoning_content")
print("=" * 60)

messages_a = [
    {"role": "system", "content": "你是一个课程讲解助手，回答要清楚、短句。"},
    {"role": "user", "content": "请解释 RAG 为什么要先检索资料再回答。"},
]

print("\n第 1 轮用户：", messages_a[-1]["content"])
reasoning_a, content_a = print_message(call_qwen(messages_a))

# 普通多轮只放最终回答 content。
messages_a.append({"role": "assistant", "content": content_a})
messages_a.append({"role": "user", "content": "那如果知识库资料过期，会有什么问题？"})

print("\n第 2 轮用户：", messages_a[-1]["content"])
print("本次请求没有传回上一轮 reasoning_content。")
print_message(call_qwen(messages_a))

print("\n" + "=" * 60)
print("二、延续思考：传回 reasoning_content，并设置 preserve_thinking=True")
print("=" * 60)

messages_b = [
    {"role": "system", "content": "你是一个技术选型助手，回答要列出取舍。"},
    {"role": "user", "content": "请比较关键词检索和向量检索在知识库问答中的差异。"},
]

print("\n第 1 轮用户：", messages_b[-1]["content"])
reasoning_b, content_b = print_message(call_qwen(messages_b))

# 需要模型参考上一轮思考过程时，assistant 里保留 reasoning_content。
messages_b.append({
    "role": "assistant",
    "content": content_b,
    "reasoning_content": reasoning_b,
})
messages_b.append({"role": "user", "content": "如果只能先做一个，应该优先做哪个？"})

print("\n第 2 轮用户：", messages_b[-1]["content"])
print("本次请求传回上一轮 reasoning_content，并设置 preserve_thinking=True。")
print_message(call_qwen(messages_b, preserve_thinking=True))

print("\n结论")
print("百炼 Qwen 普通多轮：传回历史 user 和 assistant.content 即可。")
print("如果确实要让模型参考上一轮思考过程：传 reasoning_content，并设置 preserve_thinking=True。")
