"""07 自动辩论系统：多轮辩论 + 裁判工具打分。

功能：
1. 定义辩题（system prompt）
2. 正方辩手先发言
3. 反方辩手听取正方发言后回复
4. 反复辩论共 3 轮
5. 裁判根据双方发言打分（逻辑性、语言组织、说服力）
6. 使用 function calling 工具 calculate_final_score 计算总分，给出优胜者
7. 每一步都 print 输出
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

API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
API_KEY = os.getenv("BAILIAN_API_KEY", "").strip()
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

if not API_KEY:
    raise SystemExit(
        "未设置 DEEPSEEK_API_KEY。\n"
        "macOS/Linux: export DEEPSEEK_API_KEY=your_api_key_here\n"
        "Windows PowerShell: $env:DEEPSEEK_API_KEY=\"sk-xxx\""
    )


def show_json(obj: object) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


# ------------------------------------------------------------
# 工具定义：calculate_final_score
# ------------------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_final_score",
            "description": (
                "根据正方和反方的逻辑性得分、语言组织得分、说服力得分，"
                "分别计算双方总分并给出优胜者。"
                "公式：总分 = 逻辑性 + 语言组织 + 说服力。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "affirmative_logic": {
                        "type": "number",
                        "description": "正方逻辑性得分（1-10）",
                    },
                    "affirmative_language": {
                        "type": "number",
                        "description": "正方语言组织得分（1-10）",
                    },
                    "affirmative_persuasiveness": {
                        "type": "number",
                        "description": "正方说服力得分（1-10）",
                    },
                    "negative_logic": {
                        "type": "number",
                        "description": "反方逻辑性得分（1-10）",
                    },
                    "negative_language": {
                        "type": "number",
                        "description": "反方语言组织得分（1-10）",
                    },
                    "negative_persuasiveness": {
                        "type": "number",
                        "description": "反方说服力得分（1-10）",
                    },
                },
                "required": [
                    "affirmative_logic",
                    "affirmative_language",
                    "affirmative_persuasiveness",
                    "negative_logic",
                    "negative_language",
                    "negative_persuasiveness",
                ],
            },
        },
    },
]


def execute_calculate_final_score(arguments: dict) -> dict:
    """本地执行打分工具：直接相加计算总分，给出优胜者。"""
    a_logic = arguments.get("affirmative_logic", 0)
    a_lang = arguments.get("affirmative_language", 0)
    a_pers = arguments.get("affirmative_persuasiveness", 0)
    n_logic = arguments.get("negative_logic", 0)
    n_lang = arguments.get("negative_language", 0)
    n_pers = arguments.get("negative_persuasiveness", 0)

    total_affirmative = a_logic + a_lang + a_pers
    total_negative = n_logic + n_lang + n_pers

    if total_affirmative > total_negative:
        winner = "正方"
    elif total_negative > total_affirmative:
        winner = "反方"
    else:
        winner = "平局"

    return {
        "scores": {
            "正方": {
                "逻辑性": a_logic,
                "语言组织": a_lang,
                "说服力": a_pers,
                "总分": total_affirmative,
            },
            "反方": {
                "逻辑性": n_logic,
                "语言组织": n_lang,
                "说服力": n_pers,
                "总分": total_negative,
            },
        },
        "winner": winner,
    }


# ------------------------------------------------------------
# 通用 API 调用
# ------------------------------------------------------------

def call_api(messages: list[dict], tools: list[dict] | None = None,
             tool_choice: str | None = None, max_tokens: int = 800) -> dict:
    """调用 DeepSeek Chat Completions 接口。"""
    payload: dict = {
        "model": MODEL,
        "messages": messages,
        "thinking": {"type": "disabled"},
        "temperature": 0.5,
        "max_tokens": max_tokens,
        "stream": False,
        "user_id": USER_ID,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice

    start = time.time()
    response = requests.post(
        URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=90,
    )
    elapsed = time.time() - start

    if response.status_code != 200:
        print(f"\n请求失败 HTTP {response.status_code}，耗时 {elapsed:.2f}s")
        print(response.text[:2000])
        raise SystemExit(1)

    return response.json()


# ------------------------------------------------------------
# 一、辩题与角色定义
# ------------------------------------------------------------

TOPIC = os.getenv("DEBATE_TOPIC", "人工智能的发展对人类是利大于弊还是弊大于利")
TOPIC = "人工智能代替编程是好还是坏？"
ROUNDS = 10

print("=" * 72)
print("07 自动辩论系统：多轮辩论 + 裁判工具打分")
print("=" * 72)
print("模型：", MODEL)
print("辩题：", TOPIC)
print("辩论轮数：", ROUNDS)
print()

AFFIRMATIVE_SYSTEM = {
    "role": "system",
    "content": (
        f"你是辩论赛的正方辩手。辩题：「{TOPIC}」。"
        "你坚定支持正方立场（利大于弊）。"
        "请用有力的论据、清晰的逻辑和生动的语言来论证你的观点。"
        "每次发言控制在 300 字以内，直接给出论证，不要寒暄。"
    ),
}

NEGATIVE_SYSTEM = {
    "role": "system",
    "content": (
        f"你是辩论赛的反方辩手。辩题：「{TOPIC}」。"
        "你坚定支持反方立场（弊大于利）。"
        "请针对正方的发言进行有力反驳，用清晰的逻辑和生动的语言论证你的观点。"
        "每次发言控制在 300 字以内，直接反驳，不要寒暄。"
    ),
}

COMPRESSOR_SYSTEM = {
    "role": "system",
    "content": (
        "你是辩论记录压缩员。请将下方的辩论记录压缩成简洁的摘要，"
        "保留双方的核心论点、关键反驳和逻辑脉络。压缩的不要太多，保留主要的信息点。"
        "必须输出 JSON，字段为 compressed_history。"
        "compressed_history 是一个字符串，用简洁的要点形式总结，控制在 500 字以内。"
    ),
}

JUDGE_SYSTEM = {
    "role": "system",
    "content": (
        "你是辩论赛的裁判。请你仔细阅读下方「辩论记录」中双方的每一轮发言，"
        "从逻辑性（1-10分）、语言组织（1-10分）、说服力（1-10分）三个维度，"
        "分别给正方和反方打分。"
        "请务必使用 calculate_final_score 工具提交分数。"
        "在你调用工具之前，请先简要评述双方的整体表现。"
    ),
}


# ------------------------------------------------------------
# 二、辩论阶段
# ------------------------------------------------------------

raw_history: list[dict] = []         # 完整原始记录，供裁判使用
compressed_history: str = ""          # 压缩后的摘要，供辩手使用

for round_no in range(1, ROUNDS + 1):
    print("=" * 60)
    print(f"第 {round_no} 轮")
    print("=" * 60)

    # ---- 正方发言 ----
    if not compressed_history:
        context = "（辩论开始，请正方先做开篇陈词。）"
    else:
        context = f"此前辩论摘要：\n{compressed_history}"

    affirmative_messages = [
        AFFIRMATIVE_SYSTEM,
        {
            "role": "user",
            "content": (
                f"{context}\n\n"
                f"请正方在第 {round_no} 轮发言。"
            ),
        },
    ]

    print("\n--- 正方发言 ---")
    aff_data = call_api(affirmative_messages)
    aff_content = aff_data["choices"][0]["message"]["content"]
    print(aff_content)
    print()
    print(f"(token 消耗: {json.dumps(aff_data.get('usage', {}), ensure_ascii=False)})")

    raw_history.append({"speaker": "正方", "content": aff_content})

    # ---- 反方发言 ----
    context = f"此前辩论摘要：\n{compressed_history}\n\n正方本轮发言：\n{aff_content}" if compressed_history else f"正方开篇陈词：\n{aff_content}"

    negative_messages = [
        NEGATIVE_SYSTEM,
        {
            "role": "user",
            "content": (
                f"{context}\n\n"
                f"请反方在第 {round_no} 轮发言，针对正方的最新观点进行反驳。"
            ),
        },
    ]

    print("\n--- 反方发言 ---")
    neg_data = call_api(negative_messages)
    neg_content = neg_data["choices"][0]["message"]["content"]
    print(neg_content)
    print()
    print(f"(token 消耗: {json.dumps(neg_data.get('usage', {}), ensure_ascii=False)})")

    raw_history.append({"speaker": "反方", "content": neg_content})

    # ---- 每轮结束后：压缩本轮历史 ----
    print("\n--- 历史压缩 ---")
    round_raw_text = "\n".join(
        f"[{h['speaker']}] {h['content']}" for h in raw_history
    )

    compressor_messages = [
        COMPRESSOR_SYSTEM,
        {
            "role": "user",
            "content": (
                f"请压缩以下辩论记录，保留双方核心论点与关键反驳：\n\n{round_raw_text}"
            ),
        },
    ]

    compressor_data = call_api(
        compressor_messages,
        max_tokens=800,
    )
    compressor_content = compressor_data["choices"][0]["message"]["content"]

    try:
        compressed = json.loads(compressor_content)
        compressed_history = compressed.get("compressed_history", compressor_content)
    except json.JSONDecodeError:
        compressed_history = compressor_content

    print(compressed_history)
    print()
    print(f"(token 消耗: {json.dumps(compressor_data.get('usage', {}), ensure_ascii=False)})")


# ------------------------------------------------------------
# 三、裁判打分阶段
# ------------------------------------------------------------

print()
print("=" * 60)
print("裁判打分")
print("=" * 60)

history_text = "\n".join(
    f"[{h['speaker']}] {h['content']}" for h in raw_history
)

judge_messages = [
    JUDGE_SYSTEM,
    {
        "role": "user",
        "content": f"辩题：「{TOPIC}」\n\n辩论记录：\n{history_text}\n\n请对双方进行打分并调用工具。",
    },
]

print("\n--- 裁判评述与打分 ---")
judge_data = call_api(judge_messages, tools=tools, tool_choice="auto")
judge_message = judge_data["choices"][0]["message"]
judge_content = judge_message.get("content") or ""
tool_calls = judge_message.get("tool_calls") or []

if judge_content:
    print(judge_content)

print()
print(f"(token 消耗: {json.dumps(judge_data.get('usage', {}), ensure_ascii=False)})")

if not tool_calls:
    print("\n裁判未调用打分工具，直接输出结论。")
else:
    print("\n--- 工具调用结果 ---")
    for call in tool_calls:
        fn = call["function"]
        fn_name = fn["name"]
        fn_args = json.loads(fn.get("arguments") or "{}")
        print(f"\n调用工具：{fn_name}")
        print("传入参数：")
        show_json(fn_args)

        if fn_name == "calculate_final_score":
            result = execute_calculate_final_score(fn_args)
            print("\n执行结果：")
            show_json(result)

            # 打印最终结果
            print()
            print("=" * 60)
            print("最终结果")
            print("=" * 60)
            scores = result["scores"]
            print(f"\n正方总分：{scores['正方']['总分']} 分")
            print(f"  逻辑性：{scores['正方']['逻辑性']} 分")
            print(f"  语言组织：{scores['正方']['语言组织']} 分")
            print(f"  说服力：{scores['正方']['说服力']} 分")
            print(f"\n反方总分：{scores['反方']['总分']} 分")
            print(f"  逻辑性：{scores['反方']['逻辑性']} 分")
            print(f"  语言组织：{scores['反方']['语言组织']} 分")
            print(f"  说服力：{scores['反方']['说服力']} 分")
            print(f"\n优胜者：{result['winner']}")
            print()
        else:
            print(f"未知工具：{fn_name}")


print("\n课堂观察点：")
print("1. 正方和反方使用不同的 system prompt 控制立场。")
print("2. 辩手使用压缩摘要作为上下文，避免历史过长超出 token 限制。")
print("3. 每轮结束后，压缩 Agent 将完整记录压缩为要点摘要。")
print("4. 裁判使用原始完整记录打分，保证评分基于全部发言。")
print("5. 裁判通过 function calling 调用 calculate_final_score 工具。")
print("6. 工具在本地执行（Python 加法），总分 = 逻辑性 + 语言组织 + 说服力。")
print("7. 每一步发言、压缩和打分结果都 print 出来，便于观察整个流程。")