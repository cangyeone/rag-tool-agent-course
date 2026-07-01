"""11 DeepSeek API 调用完整示例。

学习目标：掌握新版 DeepSeek API 的调用方式：
1. 使用最新模型名 deepseek-v4-flash / deepseek-v4-pro
2. 区分非思考模式与思考模式
3. 解析 content 与 reasoning_content
4. 使用流式输出和 JSON 输出
5. 处理常见 HTTP 错误

运行方式：
    python code/02_模型接口与指令设计/code/lesson_01_本地模型与服务化过渡/04_DeepSeek_API调用.py

前置条件：
    macOS / Linux:
        export DEEPSEEK_API_KEY=your_api_key_here
        export DEEPSEEK_USER_ID=student_001

    Windows PowerShell:
        $env:DEEPSEEK_API_KEY="your_api_key_here"
        $env:DEEPSEEK_USER_ID="student_001"

多人共用同一个 API Key 时，建议每个人设置不同的 DEEPSEEK_USER_ID。
这个值会写入请求体的 user_id 字段，方便后台日志、用量统计和问题排查。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("11 DeepSeek API 调用完整示例（新版兼容）")
print("=" * 72)

# ------------------------------------------------------------
# 一、基础配置
# ------------------------------------------------------------
# DeepSeek 的 OpenAI 兼容 base_url 是 https://api.deepseek.com。
# 最新文档里的 Chat Completion 路径是 /chat/completions。
# 有些旧代码使用 /v1/chat/completions，当前可能仍兼容，但课堂代码按新文档写。

API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
CHAT_URL = BASE_URL + "/chat/completions"

DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
USER_ID = (
    os.getenv("DEEPSEEK_USER_ID")
    or os.getenv("CLASSROOM_USER_ID")
    or os.getenv("USERNAME")
    or os.getenv("USER")
    or "classroom_user"
).strip()

print("\n一、基础配置")
print(f"  BASE_URL : {BASE_URL}")
print(f"  CHAT_URL : {CHAT_URL}")
print(f"  MODEL    : {DEFAULT_MODEL}")
print(f"  USER_ID  : {USER_ID}")
print(f"  API Key  : {'已设置（不会打印明文）' if API_KEY else '未设置'}")

if not API_KEY:
    print("\n  未设置 DEEPSEEK_API_KEY，脚本会进入讲解模式，不发送真实网络请求。")
    print("  Windows 示例：$env:DEEPSEEK_API_KEY=\"sk-xxx\"")
    print("  macOS/Linux 示例：export DEEPSEEK_API_KEY=your_api_key_here")
    print("  多人共用 Key 示例：export DEEPSEEK_USER_ID=student_001")

HAS_KEY = bool(API_KEY)

# ------------------------------------------------------------
# 二、模型名变化
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("二、模型名与思考模式")
print("=" * 60)

model_rows = [
    ("deepseek-v4-flash", "推荐默认", "速度快、成本低；支持非思考与思考模式"),
    ("deepseek-v4-pro", "更强模型", "复杂任务质量更高；支持非思考与思考模式"),
    ("deepseek-v4-flash", "兼容旧名", "将弃用；等价于 deepseek-v4-flash 的非思考模式"),
    ("deepseek-v4-pro", "兼容旧名", "将弃用；等价于 deepseek-v4-flash 的思考模式"),
]

for name, tag, desc in model_rows:
    print(f"  {name:<22} {tag:<10} {desc}")

print("\n  新版建议：")
print("  - 普通问答：model=deepseek-v4-flash + thinking={'type':'disabled'}")
print("  - 复杂推理：model=deepseek-v4-flash/pro + thinking={'type':'enabled','reasoning_effort':'high' 或 'max'}")

# ------------------------------------------------------------
# 三、通用工具函数
# ------------------------------------------------------------

def safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def add_user_id(payload: dict[str, Any]) -> dict[str, Any]:
    """给请求体补充 user_id 字段，用来区分同一个 API Key 下的不同使用者。"""
    payload_with_user = dict(payload)
    payload_with_user.setdefault("user_id", USER_ID)
    return payload_with_user


def print_error_response(resp: Any) -> None:
    """打印 HTTP 错误，避免课堂上只看到一长串 traceback。"""
    print(f"  请求失败：HTTP {resp.status_code}")
    try:
        data = resp.json()
        print(safe_json(data))
    except Exception:
        print(resp.text[:1000])

    hints = {
        400: "请求体格式不正确。检查 messages、model、thinking、response_format 等字段。",
        401: "API Key 错误或过期。检查 DEEPSEEK_API_KEY。",
        402: "账户余额不足。需要到 DeepSeek 控制台充值。",
        422: "参数不合法。通常是模型名、thinking、max_tokens 或 response_format 写错。",
        429: "请求过快。降低并发或稍后重试。",
        500: "服务端异常。稍后重试。",
        503: "服务繁忙。稍后重试。",
    }
    if resp.status_code in hints:
        print("  处理建议：" + hints[resp.status_code])


def post_chat(payload: dict[str, Any], title: str) -> dict[str, Any] | None:
    """发送一次非流式 Chat Completion 请求。"""
    payload = add_user_id(payload)

    print(f"\n  → {title}")
    print("  请求体：")
    print(safe_json(payload))

    if not HAS_KEY:
        print("\n  讲解模式：未发送请求。")
        return None

    try:
        import requests
    except ImportError:
        print("  缺少 requests，请先安装：pip install requests")
        return None

    start = time.time()
    try:
        resp = requests.post(
            CHAT_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90,
        )
    except Exception as exc:
        print(f"  请求异常：{type(exc).__name__}: {exc}")
        return None

    elapsed = time.time() - start
    print(f"  HTTP {resp.status_code}，耗时 {elapsed:.2f}s")

    if resp.status_code != 200:
        print_error_response(resp)
        return None

    data = resp.json()
    choice = data["choices"][0]
    message = choice.get("message", {})

    reasoning = message.get("reasoning_content")
    content = message.get("content", "")

    if reasoning:
        print("\n  reasoning_content（思考内容，业务展示时通常不直接给用户）：")
        print("  " + reasoning[:800].replace("\n", "\n  "))

    print("\n  content（最终回答）：")
    print("  " + content.replace("\n", "\n  "))

    print("\n  元数据：")
    print(f"  finish_reason = {choice.get('finish_reason')}")
    print(f"  model         = {data.get('model')}")
    if "usage" in data:
        usage = data["usage"]
        print(f"  prompt_tokens     = {usage.get('prompt_tokens')}")
        print(f"  completion_tokens = {usage.get('completion_tokens')}")
        print(f"  total_tokens      = {usage.get('total_tokens')}")

    return data

# ------------------------------------------------------------
# 四、非思考模式：普通问答
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("三、非思考模式：普通问答")
print("=" * 60)

messages = [
    {
        "role": "system",
        "content": "你是 示例业务系统 客服辅助助手。回答要简洁、稳妥；涉及规则时提醒以官方页面为准。",
    },
    {
        "role": "user",
        "content": "ORD-1001 次服务流程标准服务没票了，我还能怎么办？请给出可操作建议。",
    },
]

payload_non_thinking = {
    "model": DEFAULT_MODEL,
    "messages": messages,
    "thinking": {"type": "disabled"},
    "temperature": 0.3,
    "top_p": 0.9,
    "max_tokens": 500,
    "stream": False,
}

post_chat(payload_non_thinking, "非思考模式 deepseek-v4-flash 普通问答")

# ------------------------------------------------------------
# 五、思考模式：reasoning_content 与 content 分开
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("四、思考模式：reasoning_content 与 content")
print("=" * 60)

print("  思考模式下，不建议再设置 temperature / top_p / presence_penalty / frequency_penalty。")
print("  返回结果中：")
print("  - reasoning_content：思考过程，适合调试和课堂观察")
print("  - content：最终回答，适合给业务用户展示")

payload_thinking = {
    "model": DEFAULT_MODEL,
    "messages": [
        {
            "role": "user",
            "content": "问题1",
        }
    ],
    "thinking": {"type": "enabled", "reasoning_effort": "high"},
    "max_tokens": 1200,
    "stream": False,
}

post_chat(payload_thinking, "思考模式：读取 reasoning_content 和最终 content")

# ------------------------------------------------------------
# 六、JSON 输出
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("五、JSON 输出：response_format")
print("=" * 60)

print("  JSON 输出要点：")
print("  1. response_format={'type':'json_object'}")
print("  2. system 或 user 中要明确出现 json 字样")
print("  3. max_tokens 要给够，避免 JSON 被截断")

payload_json = {
    "model": DEFAULT_MODEL,
    "messages": [
        {
            "role": "system",
            "content": "你是客服问题分类器。必须输出合法 json，不要输出 json 之外的文字。",
        },
        {
            "role": "user",
            "content": "请把这个问题分类为 json：ORD-1001 标准服务没票了，候补申请能保证成功吗？字段包括 intent、risk_level、answer。",
        },
    ],
    "thinking": {"type": "disabled"},
    "response_format": {"type": "json_object"},
    "temperature": 0.1,
    "max_tokens": 600,
}

json_data = post_chat(payload_json, "JSON 输出模式")
if json_data:
    content = json_data["choices"][0]["message"].get("content", "")
    try:
        parsed = json.loads(content)
        print("\n  Python json.loads 解析成功：")
        print(safe_json(parsed))
    except Exception as exc:
        print(f"\n  JSON 解析失败：{exc}")

# ------------------------------------------------------------
# 七、流式输出
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("六、流式输出：stream=True")
print("=" * 60)

stream_payload = add_user_id({
    "model": DEFAULT_MODEL,
    "messages": [
        {"role": "user", "content": "用三句话解释 示例业务系统 候补申请功能。"},
    ],
    "thinking": {"type": "disabled"},
    "temperature": 0.3,
    "max_tokens": 300,
    "stream": True,
})

print("  请求体：")
print(safe_json(stream_payload))

if not HAS_KEY:
    print("\n  讲解模式：未发送流式请求。")
else:
    try:
        import requests

        resp = requests.post(
            CHAT_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json=stream_payload,
            timeout=90,
            stream=True,
        )
        print(f"\n  HTTP {resp.status_code}")
        if resp.status_code != 200:
            print_error_response(resp)
        else:
            print("\n  实时输出：")
            full_text = ""
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                # 思考模式流式输出时可能有 reasoning_content；这里也兼容解析。
                text = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or ""
                if reasoning:
                    print(reasoning, end="", flush=True)
                if text:
                    print(text, end="", flush=True)
                    full_text += text
            print("\n\n  完整回答：")
            print(full_text)
    except ImportError:
        print("  缺少 requests，请先安装：pip install requests")
    except Exception as exc:
        print(f"  流式请求异常：{type(exc).__name__}: {exc}")

# ------------------------------------------------------------
# 八、课堂总结
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("七、课堂要点")
print("=" * 60)
print("  1. 最新推荐模型名是 deepseek-v4-flash / deepseek-v4-pro。")
print("  2. deepseek-v4-flash / deepseek-v4-pro 是兼容旧名，将于 2026-07-24 弃用。")
print("  3. thinking 参数控制思考模式：enabled / disabled。")
print("  4. 思考模式的推理内容在 reasoning_content，最终回答在 content。")
print("  5. JSON 输出使用 response_format={'type':'json_object'}，提示词里也要写 json。")
print("  6. stream=True 时逐块读取 delta.content，并兼容 delta.reasoning_content。")
print("  7. 多人共用 API Key 时，可以用 user_id 字段区分不同学员或不同终端。")
print("  8. 生产代码要打印 HTTP 状态码和错误体，方便定位 401、402、422、429 等问题。")