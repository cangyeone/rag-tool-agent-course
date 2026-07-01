"""03_DeepSeek真实工具调用。

这是 DeepSeek API 真实工具调用版本。
流程尽量保持简单：
1. 把工具 schema 发给 DeepSeek
2. DeepSeek 判断要调用哪个工具，并给出参数
3. Python 本地执行工具
4. 把工具结果再发给 DeepSeek
5. DeepSeek 生成最终回答

运行方式：
    cd rag-tool-agent-course
    export DEEPSEEK_API_KEY=your_api_key_here
    python code/04_工具调用与智能体/code/lesson_01_RAG加工具调用/03_DeepSeek真实工具调用.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("03_DeepSeek真实工具调用")
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


def call_deepseek(messages: list[dict], tools: list[dict] | None = None) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "thinking": {"type": "disabled"},
        "temperature": 0.2,
        "max_tokens": 800,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    response = requests.post(
        URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=90,
    )

    if response.status_code != 200:
        print("请求失败：")
        print(response.text[:2000])
        raise SystemExit(1)

    return response.json()


def search_policy(query: str) -> dict:
    """真实系统里这里会查知识库；课堂里用固定资料模拟。"""
    return {
        "title": "候补申请规则",
        "content": "候补申请不能保证一定成功，兑现结果取决于退款、变更、新增席位和排队顺序。",
        "source": "课堂业务样例",
        "query": query,
    }


def query_order_status(order_id: str) -> dict:
    """真实系统里这里会查业务接口；课堂里用固定数据模拟。"""
    if order_id in ["T701", "T90"]:
        return {
        "order_id": order_id,
        "date": "2026-06-22",
        "remaining": 0,
        "status": "无库存",
        "suggestion": "可尝试候补申请，或查看同方向其他服务编号。",
    }
    else:
        return f"查询失败，订单号{order_id}没有在数据库中。"


tools = [
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "查询订单服务政策、候补申请、退款变更等规则资料。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要检索的政策问题，例如：候补申请是否保证成功",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_order_status",
            "description": "查询指定服务编号当前示例库存状态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "服务编号号，例如 ORD-1001",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
]

tool_map = {
    "search_policy": search_policy,
    "query_order_status": query_order_status,
}

question = "ORD-1001 没票了，候补申请一定能成功吗？请给我一个稳妥答复。"

messages = [
    {
        "role": "system",
        "content": (
            "你是客服辅助助手。需要实时状态或政策依据时，应先调用工具。"
        ),
    },
    {"role": "user", "content": question},
]

print("\n一、用户问题")
print(question)

print("\n二、第一次请求：让模型选择工具")
first_data = call_deepseek(messages, tools=tools)
assistant_message = first_data["choices"][0]["message"]
print(json.dumps(assistant_message, ensure_ascii=False, indent=2))

tool_calls = assistant_message.get("tool_calls") or []
if not tool_calls:
    print("\n模型没有选择工具，直接回答：")
    print(assistant_message.get("content", ""))
    raise SystemExit(0)

messages.append(assistant_message)

print("\n三、本地执行工具")
for tool_call in tool_calls:
    function = tool_call["function"]
    tool_name = function["name"]
    args = json.loads(function.get("arguments") or "{}")

    print(f"\n调用工具：{tool_name}")
    print("参数：", json.dumps(args, ensure_ascii=False))

    if tool_name not in tool_map:
        tool_result = {"error": f"未知工具：{tool_name}"}
    else:
        tool_result = tool_map[tool_name](**args)

    print("工具结果：")
    print(json.dumps(tool_result, ensure_ascii=False, indent=2))

    messages.append({
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": json.dumps(tool_result, ensure_ascii=False),
    })

print("\n四、第二次请求：让模型根据工具结果生成最终回答")
second_data = call_deepseek(messages, tools=tools)
final_message = second_data["choices"][0]["message"]

print("\n最终回答：")
print(final_message.get("content", ""))

print("\n课堂观察点")
print("1. 模型只负责决定调用哪个工具和填写参数。")
print("2. 工具函数由 Python 本地执行，模型自己不会真的查系统。")
print("3. 工具结果要作为 role=tool 的消息传回模型，模型才能基于结果回答。")