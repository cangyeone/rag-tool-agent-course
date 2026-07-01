"""08 论文润色：双审稿人 + 裁判停止条件。

基于辩论系统改造而成：
1. 将正反方辩手替换为两位审稿人（温和型 + 批判型）
2. 两位审稿人同时审阅论文，输出结构化的评审意见
3. 裁判（主编）判断双方意见是否都仅剩 minor revision
4. 若不是，则修改者根据意见修改论文，继续循环
5. 当双方都只有 minor 或没有意见时，迭代停止
6. 每一步都 print 输出
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import requests

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

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

if not API_KEY:
    raise SystemExit(
        "未设置 DEEPSEEK_API_KEY。\n"
        "macOS/Linux: export DEEPSEEK_API_KEY=your_api_key_here\n"
        "Windows PowerShell: $env:DEEPSEEK_API_KEY=\"sk-xxx\""
    )


def show_json(obj: object) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


# ------------------------------------------------------------
# 工具定义：submit_review_decision
# ------------------------------------------------------------

review_tools = [
    {
        "type": "function",
        "function": {
            "name": "submit_review_decision",
            "description": (
                "主编提交本轮评审决策。"
                "gentle_has_major: 温和审稿人是否提出了 major 级别的意见。"
                "critical_has_major: 批判审稿人是否提出了 major 级别的意见。"
                "stop_reason: 如果建议停止（双方都无 major），说明原因；否则说明还需要修改什么。"
                "should_stop: 是否建议停止迭代，true 表示可以停止。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gentle_has_major": {
                        "type": "boolean",
                        "description": "温和审稿人是否有 major 级别的意见",
                    },
                    "critical_has_major": {
                        "type": "boolean",
                        "description": "批判审稿人是否有 major 级别的意见",
                    },
                    "gentle_summary": {
                        "type": "string",
                        "description": "温和审稿人意见的简要总结",
                    },
                    "critical_summary": {
                        "type": "string",
                        "description": "批判审稿人意见的简要总结",
                    },
                    "stop_reason": {
                        "type": "string",
                        "description": "建议停止或继续的原因",
                    },
                    "should_stop": {
                        "type": "boolean",
                        "description": "是否建议停止迭代",
                    },
                },
                "required": [
                    "gentle_has_major",
                    "critical_has_major",
                    "should_stop",
                    "stop_reason",
                ],
            },
        },
    },
]


def execute_review_decision(arguments: dict) -> dict:
    """本地执行评审决策工具。"""
    gentle_has_major = arguments.get("gentle_has_major", False)
    critical_has_major = arguments.get("critical_has_major", False)
    should_stop = arguments.get("should_stop", False)
    stop_reason = arguments.get("stop_reason", "")

    if should_stop:
        decision = "停止迭代 — 论文已达到发表水平（双方均无 major 意见）。"
    elif not gentle_has_major and not critical_has_major:
        decision = "停止迭代 — 双方均无 major 意见。"
    else:
        decision = "继续修改 — 仍有 major 意见需要处理。"

    return {
        "gentle_has_major": gentle_has_major,
        "critical_has_major": critical_has_major,
        "should_stop": should_stop,
        "decision": decision,
        "stop_reason": stop_reason,
    }


# ------------------------------------------------------------
# 通用 API 调用
# ------------------------------------------------------------

def call_api(messages: list[dict], tools: list[dict] | None = None,
             tool_choice: str | None = None, max_tokens: int = 1500,
             response_format: dict | None = None, temperature: float = 0.5) -> dict:
    """调用 DeepSeek Chat Completions 接口。"""
    payload: dict = {
        "model": MODEL,
        "messages": messages,
        "thinking": {"type": "disabled"},
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "user_id": USER_ID,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice
    if response_format:
        payload["response_format"] = response_format

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


def call_json_agent(agent_name: str, system_prompt: str, user_data: dict,
                     max_tokens: int = 1500, temperature: float = 0.3) -> dict:
    """调用一个要求输出 JSON 的 Agent。"""
    print(f"\n--- 调用 {agent_name} ---")
    data = call_api(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_data, ensure_ascii=False)},
        ],
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    content = data["choices"][0]["message"].get("content", "")
    print(f"(token 消耗: {json.dumps(data.get('usage', {}), ensure_ascii=False)})")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("模型返回不是合法 JSON，原文如下：")
        print(content)
        raise


# ------------------------------------------------------------
# 辅助打印函数
# ------------------------------------------------------------

def print_review_issues(title: str, review: dict):
    """打印审稿人的逐条意见。"""
    issues = review.get("issues", [])
    print(f"\n{title}")
    print("-" * 60)
    if not issues:
        print("无意见。")
        return
    for i, issue in enumerate(issues, 1):
        sev = issue.get("severity", "未标注")
        print(f"{i}. [{sev}] {issue.get('issue', '')}")
        if issue.get("evidence"):
            print(f"   证据：{issue.get('evidence')}")
        if issue.get("suggestion"):
            print(f"   建议：{issue.get('suggestion')}")


def print_change_log(change_log: list):
    """打印修改记录。"""
    print("\n修改记录")
    print("-" * 60)
    if not change_log:
        print("无修改记录。")
        return
    for i, item in enumerate(change_log, 1):
        if isinstance(item, dict):
            print(f"{i}. {item.get('change', item.get('修改', str(item)))}")
        else:
            print(f"{i}. {item}")


# ------------------------------------------------------------
# 一、准备论文
# ------------------------------------------------------------

PAPER_TITLE = os.getenv("PAPER_TITLE", "大语言模型在软件工程中的应用与挑战")
PAPER_TEXT = os.getenv("PAPER_TEXT", "").strip()
MAX_ROUNDS = int(os.getenv("PAPER_REVIEW_MAX_ROUNDS", "5"))

if not PAPER_TEXT:
    draft_system = (
        "你是学术论文作者。请根据题目撰写一篇论文初稿。"
        "必须输出 JSON，字段为 title、paper。"
        "paper 控制在 800 字以内，包含摘要、引言、方法、结论等基本结构。"
        "语言学术化但不过于晦涩。"
    )
    draft = call_json_agent(
        "初稿生成",
        draft_system,
        {
            "title": PAPER_TITLE,
            "写作要求": [
                "结构完整：摘要、引言、相关工作、方法、实验、结论",
                "内容要有具体的技术描述，不能太空泛",
                "800 字以内",
            ],
        },
        max_tokens=2000,
        temperature=0.5,
    )
    PAPER_TITLE = draft.get("title", PAPER_TITLE)
    PAPER_TEXT = draft.get("paper", "").strip()

if not PAPER_TEXT:
    raise SystemExit("没有可用论文。请设置 PAPER_TEXT，或让脚本根据 PAPER_TITLE 生成初稿。")

print("=" * 72)
print("08 论文润色：双审稿人 + 裁判停止条件")
print("=" * 72)
print("模型：", MODEL)
print("论文题目：", PAPER_TITLE)
print("最大轮数：", MAX_ROUNDS)
print("\n初始论文：")
print(PAPER_TEXT)


# ------------------------------------------------------------
# 二、角色定义
# ------------------------------------------------------------

GENTLE_REVIEWER_SYSTEM = {
    "role": "system",
    "content": (
        "你是一位温和而富有建设性的学术审稿人。你的审稿风格是："
        "先肯定论文的优点和亮点，再以鼓励的口吻指出可以改进的地方。"
        "你不会使用严厉的措辞，而是以「建议」的方式提出修改意见。"
        "必须输出 JSON，字段为 reviewer（固定为 '温和审稿人'）、overall_comment、issues。"
        "issues 是数组，每项包含 issue、evidence、suggestion、severity。"
        "severity 只能是 'major'（重要修改）或 'minor'（轻微修改）。"
        "如果论文在某个维度表现很好，请明确指出来。"
    ),
}

CRITICAL_REVIEWER_SYSTEM = {
    "role": "system",
    "content": (
        "你是一位严谨而批判性的学术审稿人。你的审稿风格是："
        "直指问题核心，不容忍逻辑漏洞、数据缺失和论证不充分。"
        "你会质疑论文中的每一个主张，要求提供充分的证据支持。"
        "必须输出 JSON，字段为 reviewer（固定为 '批判审稿人'）、overall_comment、issues。"
        "issues 是数组，每项包含 issue、evidence、suggestion、severity。"
        "severity 只能是 'major'（重要修改）或 'minor'（轻微修改）。"
        "不要因为客气而回避真正的问题。"
    ),
}

JUDGE_SYSTEM = {
    "role": "system",
    "content": (
        "你是期刊主编（裁判）。你的任务是："
        "1. 阅读两位审稿人的评审意见"
        "2. 判断每位审稿人是否提出了 major 级别的修改意见"
        "3. 如果双方都只有 minor 或没有意见，建议停止迭代（should_stop=true）"
        "4. 如果任何一方仍有 major 意见，建议继续修改（should_stop=false）"
        "请务必调用 submit_review_decision 工具来提交你的决策。"
        "在调用工具之前，先简要评价本轮评审情况。"
    ),
}

REVISER_SYSTEM = {
    "role": "system",
    "content": (
        "你是论文修改者。请根据两位审稿人的意见修改论文。"
        "只修改审稿人指出的问题，不要对论文做无关的改动。"
        "保留原论文的整体结构和核心观点。"
        "必须输出 JSON，字段为 revised_paper、change_log。"
        "change_log 是数组，说明每处主要修改及其原因。"
    ),
}


# ------------------------------------------------------------
# 三、循环评审与修改
# ------------------------------------------------------------

paper = PAPER_TEXT
review_history: list[dict] = []
revision_rounds: list[dict] = []

for round_no in range(1, MAX_ROUNDS + 1):
    print()
    print("=" * 60)
    print(f"第 {round_no} 轮评审")
    print("=" * 60)

    # ---- 温和审稿人 ----
    gentle_review = call_json_agent(
        "温和审稿人",
        GENTLE_REVIEWER_SYSTEM["content"],
        {
            "title": PAPER_TITLE,
            "paper": paper,
            "history": review_history,
            "要求": "请对当前版本的论文进行评审，指出优点和可改进之处。",
        },
        max_tokens=1500,
    )
    print("\n温和审稿人原始输出：")
    show_json(gentle_review)
    print_review_issues("温和审稿人意见", gentle_review)

    # ---- 批判审稿人 ----
    critical_review = call_json_agent(
        "批判审稿人",
        CRITICAL_REVIEWER_SYSTEM["content"],
        {
            "title": PAPER_TITLE,
            "paper": paper,
            "history": review_history,
            "要求": "请对当前版本的论文进行严格评审，指出问题和不足之处。",
        },
        max_tokens=1500,
    )
    print("\n批判审稿人原始输出：")
    show_json(critical_review)
    print_review_issues("批判审稿人意见", critical_review)

    review_history.append({
        "round": round_no,
        "gentle_review": gentle_review,
        "critical_review": critical_review,
    })

    # ---- 主编（裁判）决策 ----
    print()
    print("=" * 60)
    print("主编决策")
    print("=" * 60)

    judge_messages = [
        JUDGE_SYSTEM,
        {
            "role": "user",
            "content": (
                f"论文题目：「{PAPER_TITLE}」\n\n"
                f"当前论文：\n{paper}\n\n"
                f"温和审稿人意见：\n{json.dumps(gentle_review, ensure_ascii=False, indent=2)}\n\n"
                f"批判审稿人意见：\n{json.dumps(critical_review, ensure_ascii=False, indent=2)}\n\n"
                f"请判断本轮是否应该停止迭代（双方均无 major 意见），并调用工具提交决策。"
            ),
        },
    ]

    print("\n--- 主编评述与决策 ---")
    judge_data = call_api(judge_messages, tools=review_tools, tool_choice="auto", max_tokens=1000)
    judge_message = judge_data["choices"][0]["message"]
    judge_content = judge_message.get("content") or ""

    if judge_content:
        print(judge_content)

    print(f"\n(token 消耗: {json.dumps(judge_data.get('usage', {}), ensure_ascii=False)})")

    tool_calls = judge_message.get("tool_calls") or []

    if not tool_calls:
        print("\n主编未调用决策工具，直接输出结论。")
        break

    decision = {}
    should_stop = False

    for call in tool_calls:
        fn = call["function"]
        fn_name = fn["name"]
        fn_args = json.loads(fn.get("arguments") or "{}")

        print(f"\n调用工具：{fn_name}")
        print("传入参数：")
        show_json(fn_args)

        if fn_name == "submit_review_decision":
            decision = execute_review_decision(fn_args)
            print("\n执行结果：")
            show_json(decision)
            should_stop = decision.get("should_stop", False)

    if should_stop:
        print()
        print("=" * 60)
        print(decision.get("decision", "停止迭代"))
        print("原因：", decision.get("stop_reason", ""))
        print("=" * 60)
        break

    # ---- 修改论文 ----
    print()
    print("=" * 60)
    print("修改论文")
    print("=" * 60)

    # 提取所有 major + minor 问题
    all_gentle_issues = gentle_review.get("issues", [])
    all_critical_issues = critical_review.get("issues", [])

    revision = call_json_agent(
        "论文修改者",
        REVISER_SYSTEM["content"],
        {
            "title": PAPER_TITLE,
            "paper": paper,
            "gentle_review": gentle_review,
            "critical_review": critical_review,
            "要求": "请综合两位审稿人的意见修改论文，优先处理 major 级别的问题。",
        },
        max_tokens=2500,
        temperature=0.3,
    )

    print("\n修改者原始输出：")
    show_json(revision)

    revised_paper = revision.get("revised_paper", "").strip()
    if not revised_paper:
        print("\n修改者未返回 revised_paper，循环停止。")
        break

    paper = revised_paper
    revision_rounds.append({
        "round": round_no,
        "change_log": revision.get("change_log", []),
    })

    print_change_log(revision.get("change_log", []))

    print("\n本轮修改后的论文：")
    print(paper)

else:
    print(f"\n达到最大轮数 {MAX_ROUNDS}，循环停止。")


# ------------------------------------------------------------
# 四、最终输出
# ------------------------------------------------------------

print()
print("=" * 72)
print("最终论文")
print("=" * 72)
print(f"题目：{PAPER_TITLE}")
print()
print(paper)

print()
print("=" * 72)
print("评审与修改摘要")
print("=" * 72)
for item in review_history:
    r = item["round"]
    g_count = len(item["gentle_review"].get("issues", []))
    c_count = len(item["critical_review"].get("issues", []))
    print(f"第 {r} 轮 — 温和审稿人 {g_count} 条意见，批判审稿人 {c_count} 条意见")

print(f"\n共经历 {len(revision_rounds)} 轮修改。")

# 保存结果
output_dir = COURSE_ROOT / "code/11_专题_极简辩论机器/code/output"
output_dir.mkdir(parents=True, exist_ok=True)

safe_title = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", PAPER_TITLE)[:40]
result_path = output_dir / f"paper_review_{safe_title}.json"
paper_path = output_dir / f"paper_review_{safe_title}.md"

result = {
    "title": PAPER_TITLE,
    "final_paper": paper,
    "review_history": review_history,
    "revision_rounds": revision_rounds,
}

result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
paper_path.write_text(f"# {PAPER_TITLE}\n\n{paper}\n", encoding="utf-8")

print("\n保存结果：")
print("JSON：", result_path.relative_to(COURSE_ROOT))
print("Markdown：", paper_path.relative_to(COURSE_ROOT))

print("\n课堂观察点：")
print("1. 两个审稿人（温和型 vs 批判型）从不同风格评审同一篇论文。")
print("2. 主编（裁判）通过 function calling 工具 submit_review_decision 做停止判断。")
print("3. 停止条件：双方审稿人都没有 major 级别意见时停止。")
print("4. 修改者综合两位审稿人的意见修改论文，优先处理 major 问题。")
print("5. 循环非固定轮数，由裁判动态决定何时停止。")
print("6. 每一步的评审、决策、修改结果都 print 出来。")