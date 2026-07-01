"""
07 DeepSeek thinking、JSON 和流式输出。

这一节继续使用同一个 DeepSeek Chat Completions 接口，重点观察：
1. thinking enabled / disabled 的区别
2. reasoning_content 和 content 的区别
3. JSON 输出如何约束
4. stream=True 时如何逐段读取
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("07 DeepSeek thinking、JSON 和流式输出")
print("=" * 72)

API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
URL = BASE_URL + "/chat/completions"
USER_ID = (
    os.getenv("DEEPSEEK_USER_ID")
    or os.getenv("CLASSROOM_USER_ID")
    or os.getenv("USERNAME")
    or os.getenv("USER")
    or "classroom_user"
).strip()

print("接口地址：", URL)
print("模型名称：", MODEL)
print("user_id： ", USER_ID)
print("密钥状态：", "已设置（不打印明文）" if API_KEY else "未设置")
if not API_KEY:
    print("  未设置 DEEPSEEK_API_KEY，脚本进入讲解模式，不发送真实请求。")
    print("  Windows: $env:DEEPSEEK_API_KEY=\"sk-xxx\"")
    print("  macOS/Linux: export DEEPSEEK_API_KEY=your_api_key_here")
    print("  多人共用时建议设置不同的 DEEPSEEK_USER_ID。")


def show_json(obj: object) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def post_chat(title: str, payload: dict) -> dict | None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print("请求体：")
    show_json(payload)

    if not API_KEY:
        print("\n未设置 DEEPSEEK_API_KEY，本段不发送真实请求。")
        return None

    payload_with_user = dict(payload)
    payload_with_user.setdefault("user_id", USER_ID)

    start = time.time()
    response = requests.post(
        URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload_with_user,
        timeout=90,
    )
    elapsed = time.time() - start

    print("\nHTTP 状态码：", response.status_code, f"耗时：{elapsed:.2f}s")

    if response.status_code != 200:
        print("请求失败：")
        print(response.text[:1600])
        return None

    data = response.json()
    print("完整的响应输出：", json.dumps(data, ensure_ascii=False, indent=2))
    message = data["choices"][0]["message"]

    reasoning = message.get("reasoning_content")
    content = message.get("content", "")

    if reasoning:
        print("\nreasoning_content：")
        print(reasoning[:1000])

    print("\ncontent：")
    print(content)

    print("\nfinish_reason：", data["choices"][0].get("finish_reason"))
    print("usage：")
    show_json(data.get("usage", {}))

    return data


# ------------------------------------------------------------
# 1. 普通模式：thinking disabled
# ------------------------------------------------------------

payload_disabled = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": 
"""
根据这个回答:
候补申请能不能保证成功？请结合以下信息进行回答《行业退款政策》....,<其他知识>....，不要编造，根据提供的信息进行回答。

和工具列表:
名称xxxx,说明
参数xxx
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

返回工具的json结构示意
{
      "id": "call_900c6d9d3e144c1ba1b93e6e",
      "type": "function",
      "function": {
        "name": "query_ticket",
        "arguments": "{\"train_no\": \"G107\"}"
      }
只返回JOSN文件。
""",
        }
    ],
    "thinking": {"type": "disabled"},
    "temperature": 0.3,
    "top_p": 0.9,
    "max_tokens": 300,
    "stream": False,
}

post_chat("一、普通模式：thinking disabled", payload_disabled)


# ------------------------------------------------------------
# 2. 思考模式：thinking enabled
# ------------------------------------------------------------

# 注意：
# thinking enabled 时，不建议设置 temperature、top_p、presence_penalty、frequency_penalty。
# 它们不会产生普通模式中的采样控制效果。
# 返回中可能包含 reasoning_content，业务系统通常只展示 content。

payload_enabled = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": (
                "请判断这个问题是否适合直接回答："
                "用户说自己错过了服务流程，要求全额退款。"
                "请先判断风险，再给出客服辅助答复。"
            ),
        }
    ],
    "thinking": {"type": "enabled", "reasoning_effort": "high"},
    "max_tokens": 1000,
    "stream": False,
}

post_chat("二、思考模式：thinking enabled", payload_enabled)


# ------------------------------------------------------------
# 3. JSON 输出：response_format
# ------------------------------------------------------------

# JSON 输出适合做：
# - 问题分类
# - 工单字段抽取
# - 风险标签识别
# - 后续程序继续处理
#
# 使用 response_format 时，消息中要明确要求输出 json。

payload_json = {
    "model": MODEL,
    "messages": [
        {
            "role": "system",
            "content": "你是问题分类器。必须输出合法 json，不要输出 json 之外的文字。",
        },
        {
            "role": "user",
            "content": (
                "请把问题分类为 json："
                "我买的 G107 没赶上，还能退款吗？"
                "字段包括 intent、risk_level、need_policy_check、reply。"
            ),
        },
    ],
    "thinking": {"type": "disabled"},
    "response_format": {"type": "json_object"},
    "temperature": 0.2,
    "max_tokens": 500,
    "stream": False,
}

json_data = post_chat("三、结构化输出：JSON", payload_json)

if json_data:
    content = json_data["choices"][0]["message"].get("content", "")
    print("\n尝试把 content 解析成 Python 字典：")
    try:
        parsed = json.loads(content)
        show_json(parsed)
    except json.JSONDecodeError as exc:
        print("解析失败：", exc)


# ------------------------------------------------------------
# 4. 流式输出：stream=True
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("四、流式输出：stream=True")
print("=" * 60)

payload_stream = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": "请用四个短句说明：为什么客服辅助系统适合使用流式输出？",
        }
    ],
    "thinking": {"type": "disabled"},
    "temperature": 0.3,
    "max_tokens": 300,
    "stream": True,
    "user_id": USER_ID,
}

print("请求体：")
show_json(payload_stream)

if not API_KEY:
    print("\n未设置 DEEPSEEK_API_KEY，本段不发送真实请求。")
else:
    response = requests.post(
        URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload_stream,
        timeout=90,
        stream=True,
    )

    print("\nHTTP 状态码：", response.status_code)

    if response.status_code != 200:
        print("请求失败：")
        print(response.text[:1600])
    else:
        print("\n逐段输出：")
        final_text = ""

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue

            if raw_line == "data: [DONE]":
                break

            if not raw_line.startswith("data: "):
                continue

            chunk_text = raw_line[len("data: "):]
            chunk = json.loads(chunk_text)
            delta = chunk["choices"][0].get("delta", {})

            # 普通流式输出主要读取 delta.content。
            # 如果 thinking enabled，也可能出现 delta.reasoning_content。
            text_piece = delta.get("content") or ""
            reasoning_piece = delta.get("reasoning_content") or ""

            if reasoning_piece:
                print(reasoning_piece, end="", flush=True)

            if text_piece:
                print(text_piece, end="", flush=True)
                final_text += text_piece

        print("\n\n完整 content：")
        print(final_text)

print("\n本节要点：")
print("1. thinking disabled 适合稳定业务问答。")
print("2. thinking enabled 适合复杂判断，返回中可能有 reasoning_content。")
print("3. response_format 可以要求模型输出合法 JSON。")
print("4. stream=True 可以把答案一段一段显示出来。")