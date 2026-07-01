"""02 DeepSeek 多轮对话：tool calls + thinking 模式。

这个脚本演示工具调用场景下的多轮对话：
1. 用户提出需要查工具的问题。
2. DeepSeek 选择工具，并返回 tool_calls。
3. Python 执行工具。
4. 工具结果用 role=tool 传回模型。
5. 模型基于工具结果回答。
6. 用户继续追问，程序继续带着完整历史对话。

重点：
发生 tool call 的 assistant 消息里，如果有 reasoning_content，需要保留并传回。
否则后续请求可能因为缺少上下文而出错。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("02 DeepSeek 多轮对话：tool calls + thinking 模式")
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


def search_policy(query: str) -> dict:
    return {
        "tool": "search_policy",
        "query": query,
        "title": "候补申请说明",
        "content": "候补申请是排队下单机制，是否兑现取决于退款、变更、库存释放和排队顺序，不能保证一定成功。",
        "source": "课堂示例政策库",
    }


def query_ticket(train_no: str) -> dict:
    return {
        "tool": "query_ticket",
        "train_no": train_no,
        "date": "2026-06-22",
        "seat_type": "标准服务",
        "remaining": 0,
        "status": "无库存",
        "suggestion": "可尝试候补申请，同时查看临近订单编号。",
    }


tools = [
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "查询订单服务政策、候补申请、退款变更等规则。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要检索的政策问题"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_ticket",
            "description": "查询某个订单编号的示例库存状态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "train_no": {"type": "string", "description": "订单编号号，例如 G107"}
                },
                "required": ["train_no"],
            },
        },
    },
]

tool_map = {
    "search_policy": search_policy,
    "query_ticket": query_ticket,
}

messages = [
    {
        "role": "system",
        "content": (
            "你是客服辅助助手。遇到政策、订单编号状态、规则依据时，要优先调用工具。"
            "最终回答要说明依据来自工具结果。"
        ),
    }
]


def call_deepseek() -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
        "max_tokens": 1200,
        "stream": False,
    }
    response = requests.post(URL, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        print("请求失败：")
        print(response.text[:2000])
        raise SystemExit(1)
    return response.json()


def run_user_turn(question: str) -> None:
    print(f"\n{'=' * 24} 用户新问题 {'=' * 24}")
    print("用户：", question)
    messages.append({"role": "user", "content": question})

    sub_turn = 1
    while True:
        print(f"\n--- 模型子步骤 {sub_turn}：请求模型 ---")
        data = call_deepseek()
        assistant_message = data["choices"][0]["message"]

        print("assistant message：")
        print(json.dumps(assistant_message, ensure_ascii=False, indent=2))

        # 这里要保留完整 assistant_message。
        # 对 DeepSeek thinking + tool calls 来说，里面的 reasoning_content、tool_calls 都是后续请求需要的上下文。
        messages.append(assistant_message)
        print("\n Assitent输出:\n", json.dumps(assistant_message, ensure_ascii=False, indent=2))
        tool_calls = assistant_message.get("tool_calls") or []
        if not tool_calls:
            print("\n模型最终回答：")
            print(assistant_message.get("content", ""))
            #print("\n没有工具调用了，结束本轮对话。")
            print("\n完整输入:\n", json.dumps(messages, ensure_ascii=False, indent=2))
            break

        print("\n--- 本地执行工具 ---")
        for tool_call in tool_calls:
            function = tool_call["function"]
            tool_name = function["name"]
            args = json.loads(function.get("arguments") or "{}")

            if tool_name not in tool_map:
                tool_result = {"error": f"未知工具：{tool_name}"}
            else:
                tool_result = tool_map[tool_name](**args)

            print(f"\n调用工具：{tool_name}")
            print("参数：", json.dumps(args, ensure_ascii=False))
            print("结果：", json.dumps(tool_result, ensure_ascii=False, indent=2))

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(tool_result, ensure_ascii=False),
            })

        sub_turn += 1


run_user_turn("G107 今天标准服务没票了，候补申请一定能成功吗？请给一个稳妥答复。")
#run_user_turn("如果用户很着急，客服回答时应该怎么补充建议？")

print("\n结论")
print("DeepSeek 普通多轮可以只传最终回答。")
print("DeepSeek tool calls + thinking 场景，要保留带 tool_calls 的完整 assistant 消息。")
print("工具结果用 role=tool 放回 messages，模型才能继续完成回答。")
