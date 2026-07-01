"""01_DeepSeek直接选择工具。

这个例子只演示一件事：
DeepSeek 能不能根据用户问题，直接选择一个工具，并填写工具参数。

注意：
模型只会返回 tool_calls。
真正执行工具的是 Python 程序，不是模型。

运行方式：
    cd rag-tool-agent-course
    export DEEPSEEK_API_KEY=your_api_key_here
    python code/04_工具调用与Agent/code/lesson_01_RAG加工具调用/01_DeepSeek直接选择工具.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("01_DeepSeek直接选择工具")
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

# 给模型看的工具清单。
# description 写得越清楚，模型越容易选对。
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "查询订单服务政策、候补申请、退款变更、学生优惠等规则资料。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要查询的政策问题。",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        # 我想知道T7次服务流程还有没有订单，从北京到上海大概多少钱。
        "type": "function",
        "function": {
            "name": "query_ticket",
            "description": "查询指定订单编号的示例库存状态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "train_no": {
                        "type": "string",
                        "description": "订单编号号，例如 G107。",
                    }
                },
                "required": ["train_no"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc_refund_fee",
            "description": "根据价格和距离服务开始小时数，估算退款手续费。",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_price": {
                        "type": "number",
                        "description": "价格，单位元。",
                    },
                    "hours_before_departure": {
                        "type": "number",
                        "description": "距离服务开始还有多少小时。",
                    },
                },
                "required": ["ticket_price", "hours_before_departure"],
            },
        },
    },
]

question = "G107 今天还有标准服务吗？我购买了订单，价格 553 元，服务开始前 12 小时退款，手续费大概怎么算？"

messages = [

    {
        "role": "system",
        "content": (
            "你是工具选择器。"
            "如果用户问题需要外部数据或规则查询，请选择一个最合适的工具。"
            "本轮只需要选择工具，不需要解释。"
        ),
    },

    {"role": "user", "content": question},
]

payload = {
    "model": MODEL,
    "messages": messages,
    "tools": tools,
    "tool_choice": "auto",
    "thinking": {"type": "disabled"},
    "temperature": 0.1,
    "max_tokens": 500,
    "stream": False,
}

print("\n用户问题：")
print(question)

print("\n发送给 DeepSeek 的 tools：")
for item in tools:
    fn = item["function"]
    print(f"- {fn['name']}: {fn['description']}")

response = requests.post(
    URL,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json=payload,
    timeout=90,
)

print("\nHTTP 状态码：", response.status_code)

if response.status_code != 200:
    print("请求失败：")
    print(response.text[:2000])
    raise SystemExit(1)

data = response.json()
print("\n完整返回结果：")
print(json.dumps(data, ensure_ascii=False, indent=2))
message = data["choices"][0]["message"]

print("\n模型原始 message：")
print(json.dumps(message, ensure_ascii=False, indent=2))

tool_calls = message.get("tool_calls") or []

print("\n工具选择结果：")
if not tool_calls:
    print("模型没有选择工具，直接输出：")
    print(message.get("content", ""))
else:
    for call in tool_calls:
        fn = call["function"]
        name = fn["name"]
        arguments = json.loads(fn.get("arguments") or "{}")
        print("选择工具：", name)
        print("工具参数：", json.dumps(arguments, ensure_ascii=False, indent=2))

print("\n课堂观察点")
print("1. DeepSeek 可以直接返回 tool_calls，说明它完成了工具选择。")
print("2. 这一步只是在选择工具和参数，还没有执行工具。")
print("3. 后续脚本 03_DeepSeek真实工具调用.py 会继续执行工具，并把结果传回模型。")
