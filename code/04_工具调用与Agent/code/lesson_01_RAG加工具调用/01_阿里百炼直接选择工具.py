"""01_阿里百炼直接选择工具。

这个例子只演示一件事：
阿里百炼的 OpenAI 兼容接口能不能根据用户问题，直接选择工具并填写参数。

注意：
模型只返回 tool_calls。
真正执行工具的是 Python 程序，不是模型。

运行方式：
    cd rag-tool-agent-course
    python code/04_工具调用与Agent/code/lesson_01_RAG加工具调用/01_阿里百炼直接选择工具.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("01_阿里百炼直接选择工具")
print("=" * 72)

API_BASE = os.getenv("BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("BAILIAN_MODEL", "qwen-plus")
API_KEY = os.getenv("BAILIAN_API_KEY", "").strip()
URL = API_BASE.rstrip("/") + "/chat/completions"

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

# 可以把这个问题换成下面几类，观察模型是否会换工具：
# 1. 候补申请一定能成功吗？
# 2. G107 今天还有标准服务吗？
# 3. 价格 553 元，服务开始前 12 小时退款，手续费大概怎么算？
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
    "parallel_tool_calls": False,
    "enable_thinking": False,
    "temperature": 0.1,
    "max_tokens": 500,
    "stream": False,
}

print("\n用户问题：")
print(question)

print("\n发送给阿里百炼的 tools：")
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
print("1. 阿里百炼兼容 OpenAI 的 tools / tool_choice 写法。")
print("2. tool_choice='auto' 表示由模型自主选择是否调用工具。")
print("3. 这一步只是在选择工具和参数，还没有执行工具。")
print("4. 如需完整执行工具，可参考 DeepSeek 的 03_DeepSeek真实工具调用.py，再把接口换成百炼。")
