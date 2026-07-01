"""06 百炼 Qwen 多轮对话：tool calls + thinking。

这个脚本演示百炼平台 Qwen 在线模型的工具调用流程。

执行链路：
1. 用户提出问题。
2. Qwen 根据 tools 说明选择工具，并返回 tool_calls。
3. Python 执行本地工具函数。
4. 工具结果以 role=tool 放回 messages。
5. 再请求 Qwen，让模型基于工具结果生成最终回答。
6. 用户继续追问，程序继续带着历史 messages。

注意：
- 百炼 Function Calling 的基本结构与 OpenAI Chat Completions 接近。
- 第一次请求可以设置 tool_choice="auto"，让模型自己选工具。
- 提交工具结果、让模型总结时，不再传 tool_choice，避免模型继续只返回工具调用信息。
- 如果 history 中保留 reasoning_content，并希望模型参考上一轮思考，可设置 preserve_thinking=True。

运行方式：
    cd rag-tool-agent-course
    python code/12_专题_多轮对话机制/code/06_百炼Qwen多轮对话_tool_calls_thinking.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("06 百炼 Qwen 多轮对话：tool calls + thinking")
print("=" * 72)

API_BASE = os.getenv("BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("BAILIAN_MODEL", "qwen-plus")
API_KEY = os.getenv("BAILIAN_API_KEY", "").strip()
URL = API_BASE.rstrip("/") + "/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def search_policy(query: str) -> dict:
    """模拟政策知识库查询。真实项目里这里会连接 RAG、数据库或业务接口。"""
    return {
        "tool": "search_policy",
        "query": query,
        "title": "候补申请规则",
        "content": "候补申请是排队兑现机制，不保证一定成功。兑现结果与库存释放、退款变更、候补申请顺序和截止时间有关。",
        "source": "课堂示例政策库",
    }


def query_ticket(train_no: str, date: str = "2026-06-22") -> dict:
    """模拟订单编号库存查询。真实项目里这里会调用订单服务或运营系统接口。"""
    return {
        "tool": "query_ticket",
        "train_no": train_no,
        "date": date,
        "from": "北京南",
        "to": "上海虹桥",
        "seat_type": "标准服务",
        "remaining": 0,
        "status": "无库存",
        "suggestion": "可提交候补申请，并同步查看临近订单编号或不同服务类型。",
    }


def reply_template(scene: str) -> dict:
    """模拟话术模板工具。真实项目里这里可以接客服知识库或模板系统。"""
    templates = {
        "no_ticket": "当前订单编号暂无库存，可以提交候补申请订单。候补申请不保证一定兑现，建议同时关注其他订单编号。",
        "policy": "相关规则以 示例业务系统 页面和服务网点公告为准，如遇特殊情况建议联系人工客服确认。",
        "default": "请根据查询结果向用户说明现状，并给出可执行的下一步建议。",
    }
    return {
        "tool": "reply_template",
        "scene": scene,
        "template": templates.get(scene, templates["default"]),
    }


tools = [
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "查询候补申请、退款变更、订单服务规则等政策资料。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要查询的政策问题，例如：候补申请是否保证成功",
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
            "description": "查询指定订单编号在指定日期的示例库存状态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "train_no": {
                        "type": "string",
                        "description": "订单编号号，例如 G107",
                    },
                    "date": {
                        "type": "string",
                        "description": "使用服务日期，格式 YYYY-MM-DD；不知道时使用 2026-06-22",
                    },
                },
                "required": ["train_no"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reply_template",
            "description": "根据业务场景返回客服答复模板。",
            "parameters": {
                "type": "object",
                "properties": {
                    "scene": {
                        "type": "string",
                        "enum": ["no_ticket", "policy", "default"],
                        "description": "答复场景：no_ticket 表示无票，policy 表示规则解释，default 表示通用答复",
                    }
                },
                "required": ["scene"],
            },
        },
    },
]

tool_map = {
    "search_policy": search_policy,
    "query_ticket": query_ticket,
    "reply_template": reply_template,
}

messages = [
    {
        "role": "system",
        "content": (
            "你是客服辅助助手。遇到订单编号状态、订单服务规则、答复模板时，应先调用工具。"
            "不要编造工具结果。最终回答要说明依据来自查询结果，并提醒以官方页面为准。"
        ),
    }
]


def call_qwen(use_tool_choice: bool, preserve_thinking: bool = False) -> dict:
    """调用百炼 OpenAI 兼容 Chat Completions 接口。"""
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": tools,
        "enable_thinking": True,
        "thinking_budget": 1024,
        "max_tokens": 1200,
        "temperature": 0.1,
        "stream": False,
    }

    if use_tool_choice:
        payload["tool_choice"] = "auto"

    if preserve_thinking:
        payload["preserve_thinking"] = True

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
    need_tool_choice = True

    while True:
        print(f"\n--- 模型子步骤 {sub_turn}：请求 Qwen ---")

        # 第一次让模型自己选工具：tool_choice="auto"。
        # 执行完工具后，让模型总结工具结果：不再传 tool_choice。
        data = call_qwen(
            use_tool_choice=need_tool_choice,
            preserve_thinking=sub_turn > 1,
        )

        assistant_message = data["choices"][0]["message"]
        print("assistant message：")
        print(json.dumps(assistant_message, ensure_ascii=False, indent=2))

        # 保留完整 assistant message。
        # 如果里面有 reasoning_content，下一步 preserve_thinking=True 时模型可以继续参考。
        # 如果里面有 tool_calls，后面的 role=tool 结果也要和它对应。
        messages.append(assistant_message)

        tool_calls = assistant_message.get("tool_calls") or []
        if not tool_calls:
            print("\n模型最终回答：")
            print(assistant_message.get("content", ""))
            print("\n当前完整 messages：")
            print(json.dumps(messages, ensure_ascii=False, indent=2))
            break

        print("\n--- Python 本地执行工具 ---")
        for tool_call in tool_calls:
            function = tool_call["function"]
            tool_name = function["name"]
            args_text = function.get("arguments") or "{}"

            try:
                args = json.loads(args_text)
            except json.JSONDecodeError:
                args = {}
                tool_result = {
                    "error": "工具参数不是合法 JSON",
                    "raw_arguments": args_text,
                }
            else:
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

        # 已经有工具结果了，下一次请求主要是总结工具结果。
        # 按百炼 Function Calling 文档，不再携带 tool_choice。
        need_tool_choice = False
        sub_turn += 1


run_user_turn("G107 在 2026-06-22 标准服务没票了，候补申请一定能成功吗？请给一个客服答复。")
run_user_turn("如果用户很着急，下一句应该怎么补充建议？")

print("\n结论")
print("1. 百炼 Qwen 的工具调用也是：模型选工具，程序执行工具，工具结果回传模型。")
print("2. 第一次请求可以使用 tool_choice='auto'，让模型自己判断工具。")
print("3. 总结工具结果时不要继续传 tool_choice，避免模型继续返回工具调用。")
print("4. 如果保留 reasoning_content 并希望继续参考，可设置 preserve_thinking=True。")
