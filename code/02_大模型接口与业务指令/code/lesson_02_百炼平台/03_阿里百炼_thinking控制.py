"""
03 阿里百炼 thinking 控制与 JSON 输出。

这一节重点演示 Qwen 混合思考模型的真实调用方式：
1. enable_thinking=False：普通回答，直接返回最终内容
2. enable_thinking=True + stream=False：非实时返回完整 reasoning_content 和 content
3. enable_thinking=True + stream=True：流式返回 reasoning_content 和 content
4. thinking_budget：选择不同思考层级
5. response_format：关闭 thinking 后输出 JSON，方便程序继续处理

注意：
    enable_thinking 不是标准 OpenAI 参数。
    使用 OpenAI Python SDK 时，需要放到 extra_body 里。
    本脚本直接使用 requests 发送 HTTP JSON，所以把 enable_thinking 放在请求体顶层。
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

print("03 阿里百炼 thinking 控制与 JSON 输出")
print("=" * 72)

API_BASE = os.getenv("BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("BAILIAN_MODEL", "qwen-plus")
API_KEY = os.getenv("BAILIAN_API_KEY", "").strip()
URL = API_BASE.rstrip("/") + "/chat/completions"

# 是否打印接口返回的原始内容。
# True 适合课堂观察：可以看到 data:、[DONE]、JSON 转义字符等接口层面的符号。
# False 适合平时运行：只看整理后的思考过程和最终回答。
SHOW_RAW_RESPONSE = True

# 是否一次性运行所有 thinking 层级。
# 默认 False，只运行 SELECTED_THINKING_LEVEL 指定的一档，避免课堂等待太久。
RUN_ALL_THINKING_LEVELS = False

# 思考层级选择：
# quick    ：快速判断，适合课堂演示和简单问题
# standard ：标准分析，兼顾速度和推理展开程度
# deep     ：深度分析，适合复杂判断，但耗时和 token 都更多
SELECTED_THINKING_LEVEL = "standard"

THINKING_LEVELS = {
    "quick": {
        "label": "快速档",
        "thinking_budget": 512,
        "max_tokens": 700,
        "description": "适合简单判断，速度快，思考过程较短。",
    },
    "standard": {
        "label": "标准档",
        "thinking_budget": 2048,
        "max_tokens": 1200,
        "description": "适合课堂主要演示，能看到较完整的分析过程。",
    },
    "deep": {
        "label": "深度档",
        "thinking_budget": 4096,
        "max_tokens": 1800,
        "description": "适合复杂任务，输出更长，等待时间和 token 消耗也更高。",
    },
}


def pretty(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def post_once(title: str, payload: dict) -> dict | None:
    """非流式请求：适合普通问答、非实时 thinking、JSON 输出。"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print("请求体：")
    print(pretty(payload))

    start = time.time()
    response = requests.post(
        URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    elapsed = time.time() - start

    print("\nHTTP 状态码：", response.status_code, f"耗时：{elapsed:.2f}s")

    if response.status_code != 200:
        print("请求失败：")
        print(response.text[:2000])
        return None

    data = response.json()
    message = data["choices"][0]["message"]

    if SHOW_RAW_RESPONSE:
        print("\n完整原始 JSON 响应：")
        print(pretty(data))

    reasoning = message.get("reasoning_content")
    if reasoning:
        print("\nreasoning_content：")
        print(reasoning)

    content = message.get("content", "")
    print("\ncontent：")
    print(content)

    if SHOW_RAW_RESPONSE:
        print("\n字段 repr：")
        print("reasoning_content repr：")
        print(repr(reasoning or ""))
        print("content repr：")
        print(repr(content))

    print("\nusage：")
    print(pretty(data.get("usage", {})))
    return data


def post_stream_with_thinking(title: str, payload: dict) -> None:
    """流式请求：适合观察 Qwen 的 thinking 过程。"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print("请求体：")
    print(pretty(payload))

    start = time.time()
    response = requests.post(
        URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
        stream=True,
    )

    print("\nHTTP 状态码：", response.status_code)

    if response.status_code != 200:
        print("请求失败：")
        print(response.text[:2000])
        return

    reasoning_text = ""
    answer_text = ""
    is_answering = False
    usage = None
    raw_lines = []

    print("\n==================== 思考过程 reasoning_content ====================\n")

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue

        raw_lines.append(raw_line)

        if raw_line == "data: [DONE]":
            break

        if not raw_line.startswith("data: "):
            continue

        chunk_text = raw_line[len("data: "):]

        try:
            chunk = json.loads(chunk_text)
        except json.JSONDecodeError:
            print("\n无法解析的流式片段：", chunk_text[:300])
            continue

        # stream_options.include_usage=True 时，最后一个 chunk 可能只有 usage，没有 choices。
        if chunk.get("usage") is not None:
            usage = chunk.get("usage")

        choices = chunk.get("choices") or []
        if not choices:
            continue

        delta = choices[0].get("delta") or {}
        reasoning_piece = delta.get("reasoning_content") or ""
        content_piece = delta.get("content") or ""

        if reasoning_piece:
            print(reasoning_piece, end="", flush=True)
            reasoning_text += reasoning_piece

        if content_piece:
            if not is_answering:
                print("\n\n==================== 最终回答 content ====================\n")
                is_answering = True
            print(content_piece, end="", flush=True)
            answer_text += content_piece

    elapsed = time.time() - start

    print("\n\n==================== 汇总 ====================")
    print(f"耗时：{elapsed:.2f}s")
    print(f"思考过程字符数：{len(reasoning_text)}")
    print(f"最终回答字符数：{len(answer_text)}")
    if usage:
        print("usage：")
        print(pretty(usage))

    if SHOW_RAW_RESPONSE:
        print("\n==================== 完整原始流式响应 ====================")
        print("下面是接口逐行返回的原始 SSE 内容。")
        print("repr(...) 会把换行、引号、反斜杠等特殊符号显式展示出来。")
        for index, raw_line in enumerate(raw_lines, start=1):
            print(f"{index:03d}: {raw_line!r}")

        print("\n==================== 拼接后的完整字段 ====================")
        print("reasoning_content repr：")
        print(repr(reasoning_text))
        print("\ncontent repr：")
        print(repr(answer_text))


print("接口地址：", URL)
print("模型名称：", MODEL)
print("密钥状态：已填写（不打印明文）")

print("\n可选 thinking 层级：")
for key, config in THINKING_LEVELS.items():
    selected = "  ← 当前选择" if key == SELECTED_THINKING_LEVEL else ""
    print(
        f"- {key:<8} {config['label']:<6} "
        f"thinking_budget={config['thinking_budget']:<5} "
        f"max_tokens={config['max_tokens']:<5} "
        f"{config['description']}{selected}"
    )

question = (
    "某个客服助手收到问题：用户说自己错过了车，要求全额退款。"
    "请判断这个问题为什么不能直接承诺，并给出一段稳妥答复。"
)

# 一、关闭 thinking：普通非流式回答。
# 适合大多数客服辅助、摘要、分类、格式改写任务。
payload_no_thinking = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": question},
    ],
    "enable_thinking": False,
    "temperature": 0.3,
    "max_tokens": 500,
    "stream": False,
}

post_once("一、关闭 thinking：普通回答", payload_no_thinking)

# 二、选择 thinking 层级。
# thinking_budget 控制思考过程最多消耗多少 token，可按课堂时间调小或调大。
if SELECTED_THINKING_LEVEL not in THINKING_LEVELS:
    raise SystemExit(f"未知 thinking 层级：{SELECTED_THINKING_LEVEL}")

levels_to_run = (
    list(THINKING_LEVELS.items())
    if RUN_ALL_THINKING_LEVELS
    else [(SELECTED_THINKING_LEVEL, THINKING_LEVELS[SELECTED_THINKING_LEVEL])]
)

# 三、开启 thinking：非实时完整返回。
# 这个版本不需要 stream，也可以在完整 JSON 响应中看到 reasoning_content。
# 适合讲清楚“模型响应里到底有哪些字段”。
for level_name, level_config in levels_to_run:
    payload_thinking_once = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": question},
        ],
        "enable_thinking": True,
        "thinking_budget": level_config["thinking_budget"],
        "max_tokens": level_config["max_tokens"],
        "stream": False,
    }

    post_once(
        f"二、开启 thinking：非实时完整返回 / {level_config['label']}（{level_name}）",
        payload_thinking_once,
    )

# 四、开启 thinking：流式输出。
# 这个版本用于观察 delta.reasoning_content 和 delta.content 如何一段一段返回。
for level_name, level_config in levels_to_run:
    payload_thinking_stream = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": question},
        ],
        "enable_thinking": True,
        "thinking_budget": level_config["thinking_budget"],
        "max_tokens": level_config["max_tokens"],
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }

    post_stream_with_thinking(
        f"三、开启 thinking：流式返回 / {level_config['label']}（{level_name}）",
        payload_thinking_stream,
    )

# 五、JSON 输出：建议关闭 thinking。
# 结构化输出更关注格式稳定，通常不需要展示思考过程。
payload_json = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": (
                "请输出合法 JSON，不要输出 JSON 之外的文字。"
                "分析问题：用户说 G107 没赶上，问能不能全额退款。"
                "字段包括 intent、risk_level、need_policy_check、reply。"
            ),
        }
    ],
    "enable_thinking": False,
    "response_format": {"type": "json_object"},
    "temperature": 0.2,
    "max_tokens": 600,
    "stream": False,
}

json_data = post_once("四、关闭 thinking 后输出 JSON", payload_json)

if json_data:
    content = json_data["choices"][0]["message"].get("content", "")
    print("\n解析 JSON：")
    try:
        parsed = json.loads(content)
        print(pretty(parsed))
    except json.JSONDecodeError as exc:
        print("解析失败：", exc)

print("\n本节要点：")
print("1. 百炼 Qwen 的思考开关是 enable_thinking。")
print("2. 直接 HTTP 请求时，enable_thinking 放在请求体顶层。")
print("3. thinking_budget 可以控制思考层级，数值越大，思考空间越充足。")
print("4. 非流式请求也可以在完整响应中读取 reasoning_content。")
print("5. 流式请求可以观察 delta.reasoning_content 的逐段返回。")
print("6. SHOW_RAW_RESPONSE=True 时，可以看到完整原始响应和特殊符号。")
print("7. JSON 输出建议关闭 thinking，让格式更稳定。")