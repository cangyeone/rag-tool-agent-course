"""06 文章润色：辩论式评审循环。

课堂目标：
1. 把“辩论机器”改造成“文章评审 + 修改”机器。
2. 两个评审 Agent 从不同角度找问题。
3. 判断 Agent 只判断有没有新增问题。
4. 修改 Agent 根据新增问题润色文章。
5. 当判断 Agent 认为没有新增问题时，循环停止。

这个例子适合讲：
- Role Playing：结构评审、表达评审、修改者、判断者
- Multi-Agent：多个 Agent 协作处理同一篇文章
- Memory：记住已经发现和处理过的问题
- JSON：每个 Agent 都输出结构化结果，程序才能继续循环
- Stop Condition：没有新增问题就停止
"""

import os
import json
import re
import requests
from pathlib import Path

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")


# ------------------------------------------------------------
# 一、读取 API Key
# ------------------------------------------------------------

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    env_file = COURSE_ROOT / "code/11_专题_极简辩论机器/code/.env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("DEEPSEEK_API_KEY="):
                api_key = line.strip().split("=", 1)[1]

if not api_key:
    raise SystemExit("请设置 DEEPSEEK_API_KEY，或复制仓库根目录 .env.example 为 .env 后填写。")

base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
url = base_url + "/chat/completions"
model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
user_id = os.getenv("DEEPSEEK_USER_ID", os.getenv("USER", "classroom_user"))
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def call_json_agent(agent_name, system_prompt, user_data, max_tokens=1200):
    """调用一个要求输出 JSON 的 Agent。"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_data, ensure_ascii=False)},
        ],
        "thinking": {"type": "disabled"},
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "user_id": user_id,
    }

    print(f"\n调用 {agent_name}")
    response = requests.post(url, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("模型返回不是合法 JSON，原文如下：")
        print(content)
        raise


def print_review_issues(title, review):
    """把评审 Agent 的问题逐条打印出来，课堂上不用盯着 JSON 看。"""
    issues = review.get("issues", [])
    print(f"\n{title}")
    print("-" * 72)
    if not issues:
        print("没有提出问题。")
        return

    for i, issue in enumerate(issues, 1):
        print(f"{i}. 问题：{issue.get('issue', '')}")
        print(f"   严重程度：{issue.get('severity', '未标注')}")
        print(f"   证据：{issue.get('evidence', '')}")
        print(f"   建议：{issue.get('suggestion', '')}")


def print_judge_result(judge):
    """把判断 Agent 的结论、采纳和忽略的问题都打印出来。"""
    print("\n判断 Agent 结论")
    print("-" * 72)
    print("是否有新增问题：", judge.get("has_new_issues"))
    print("停止原因：", judge.get("stop_reason", "未给出"))

    new_issues = judge.get("new_issues", [])
    print("\n采纳的新增问题：")
    if not new_issues:
        print("无")
    for i, issue in enumerate(new_issues, 1):
        print(f"{i}. [{issue.get('severity', '未标注')}] {issue.get('issue', '')}")
        print(f"   来源：{issue.get('source', '未标注')}")
        print(f"   建议：{issue.get('suggestion', '')}")

    ignored_issues = judge.get("ignored_issues", [])
    print("\n忽略的问题：")
    if not ignored_issues:
        print("无")
    for i, issue in enumerate(ignored_issues, 1):
        if isinstance(issue, dict):
            print(f"{i}. {issue.get('issue', '')}")
            print(f"   忽略原因：{issue.get('reason', issue.get('ignored_reason', '未说明'))}")
        else:
            print(f"{i}. {issue}")


def print_change_log(change_log):
    """把修改 Agent 的修改记录逐条打印出来。"""
    print("\n修改 Agent 的改动记录")
    print("-" * 72)
    if not change_log:
        print("没有返回 change_log。")
        return

    for i, item in enumerate(change_log, 1):
        if isinstance(item, dict):
            print(f"{i}. {item.get('change', item.get('修改', item))}")
            if item.get("reason"):
                print(f"   原因：{item.get('reason')}")
            if item.get("before"):
                print(f"   修改前：{item.get('before')}")
            if item.get("after"):
                print(f"   修改后：{item.get('after')}")
        else:
            print(f"{i}. {item}")


# ------------------------------------------------------------
# 二、准备文章
# ------------------------------------------------------------
# 可以通过环境变量传入题目和文章。
# 如果只传题目，脚本会先让模型生成一版初稿，再进入评审循环。

title = os.getenv("ARTICLE_TITLE", "构建企业知识助手为什么需要 RAG")
article = os.getenv("ARTICLE_TEXT", "").strip()

if not article:
    draft_system = (
        "你是技术文章作者。请根据题目写一篇短文初稿。"
        "必须输出 JSON，字段为 title、article。article 控制在 600 字以内。"
    )
    draft = call_json_agent(
        "初稿生成 Agent",
        draft_system,
        {
            "title": title,
            "写作要求": [
                "面向技术培训读者",
                "语言自然，不要口号化",
                "要说明 RAG、知识库、检索、引用来源的关系",
            ],
        },
        max_tokens=1200,
    )
    title = draft.get("title", title)
    article = draft.get("article", "").strip()

if not article:
    raise SystemExit("没有可用文章。请设置 ARTICLE_TEXT，或让脚本根据 ARTICLE_TITLE 生成初稿。")


print("=" * 72)
print("06 文章润色：辩论式评审循环")
print("=" * 72)
print("模型：", model)
print("user_id：", user_id)
print("题目：", title)
print("\n初始文章：")
print(article)


# ------------------------------------------------------------
# 三、四个 Agent 的角色定义
# ------------------------------------------------------------

structure_reviewer_prompt = (
    "你是文章结构评审 Agent。只关注结构、逻辑、信息顺序、论证是否完整。"
    "请找出当前文章中仍然值得修改的问题。"
    "不要重复 known_issues 中已经记录的问题。"
    "必须输出 JSON，字段为 reviewer、issues。"
    "issues 是数组，每项包含 issue、evidence、suggestion、severity。"
    "如果没有问题，issues 输出空数组。"
)

expression_reviewer_prompt = (
    "你是文章表达评审 Agent。只关注语言是否自然、是否空泛、是否啰嗦、读者是否容易理解。"
    "请找出当前文章中仍然值得修改的问题。"
    "不要重复 known_issues 中已经记录的问题。"
    "必须输出 JSON，字段为 reviewer、issues。"
    "issues 是数组，每项包含 issue、evidence、suggestion、severity。"
    "如果没有问题，issues 输出空数组。"
)

judge_prompt = (
    "你是判断 Agent。你的任务不是润色文章，而是判断两位评审提出的问题里有没有新增问题。"
    "新增问题指：known_issues 里没有出现过，并且确实值得修改的问题。"
    "重复表达、轻微措辞偏好、已经修过的问题，不算新增问题。"
    "直到达到出版质量为准"
    "必须输出 JSON，字段为 has_new_issues、new_issues、ignored_issues、stop_reason。"
    "has_new_issues 是 boolean。new_issues 是数组，每项包含 issue、source、suggestion、severity。"
)

revision_prompt = (
    "你是文章润色 Agent。请根据 new_issues 修改文章。"
    "要求保留原文章主题，不要扩写成论文，不要加入没有依据的新事实。"
    "修改后语言要自然、具体、适合技术培训材料。"
    "必须输出 JSON，字段为 revised_article、change_log。"
    "change_log 是数组，说明每处主要修改。"
)


# ------------------------------------------------------------
# 四、循环评审与修改
# ------------------------------------------------------------

known_issues = []
revision_history = []
max_rounds = int(os.getenv("ARTICLE_POLISH_MAX_ROUNDS", "4"))

for round_no in range(1, max_rounds + 1):
    print("\n" + "=" * 72)
    print(f"第 {round_no} 轮：评审当前文章")
    print("=" * 72)

    shared_input = {
        "title": title,
        "article": article,
        "known_issues": known_issues,
        "要求": "只找当前仍然需要修改的问题，不要为了挑问题而挑问题。",
    }

    structure_review = call_json_agent(
        "结构评审 Agent",
        structure_reviewer_prompt,
        shared_input,
        max_tokens=1200,
    )

    expression_review = call_json_agent(
        "表达评审 Agent",
        expression_reviewer_prompt,
        shared_input,
        max_tokens=1200,
    )

    print("\n结构评审结果：")
    print(json.dumps(structure_review, ensure_ascii=False, indent=2))
    print("\n表达评审结果：")
    print(json.dumps(expression_review, ensure_ascii=False, indent=2))
    print_review_issues("结构评审 Agent 的逐条意见", structure_review)
    print_review_issues("表达评审 Agent 的逐条意见", expression_review)

    judge_input = {
        "title": title,
        "article": article,
        "known_issues": known_issues,
        "structure_review": structure_review,
        "expression_review": expression_review,
        "判断要求": "只把真正新增、值得修改的问题放入 new_issues。",
    }

    judge = call_json_agent("判断 Agent", judge_prompt, judge_input, max_tokens=1200)

    print("\n判断 Agent 输出：")
    print(json.dumps(judge, ensure_ascii=False, indent=2))
    print_judge_result(judge)

    new_issues = judge.get("new_issues", [])
    has_new_issues = bool(judge.get("has_new_issues")) and bool(new_issues)

    if not has_new_issues:
        print("\n没有新增问题，润色循环停止。")
        print("停止原因：", judge.get("stop_reason", "判断 Agent 未给出原因"))
        break

    print("\n本轮新增问题：")
    for i, issue in enumerate(new_issues, 1):
        print(f"{i}. [{issue.get('severity', '未标注')}] {issue.get('issue')}")
        print("   建议：", issue.get("suggestion"))

    revision = call_json_agent(
        "修改 Agent",
        revision_prompt,
        {
            "title": title,
            "article": article,
            "new_issues": new_issues,
            "known_issues": known_issues,
        },
        max_tokens=1800,
    )

    revised_article = revision.get("revised_article", "").strip()
    if not revised_article:
        print("\n修改 Agent 没有返回 revised_article，循环停止。")
        break

    article = revised_article
    known_issues.extend(new_issues)
    revision_history.append(
        {
            "round": round_no,
            "new_issues": new_issues,
            "change_log": revision.get("change_log", []),
        }
    )

    print("\n修改 Agent 原始输出：")
    print(json.dumps(revision, ensure_ascii=False, indent=2))
    print_change_log(revision.get("change_log", []))

    print("\n本轮修改后的文章：")
    print(article)

else:
    print("\n达到最大轮数，停止循环。")


# ------------------------------------------------------------
# 五、保存结果
# ------------------------------------------------------------

output_dir = COURSE_ROOT / "code/11_专题_极简辩论机器/code/output"
output_dir.mkdir(parents=True, exist_ok=True)

safe_title = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", title)[:40]
result_path = output_dir / f"article_polish_{safe_title}.json"
article_path = output_dir / f"article_polish_{safe_title}.md"

result = {
    "title": title,
    "final_article": article,
    "known_issues": known_issues,
    "revision_history": revision_history,
}

result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
article_path.write_text(f"# {title}\n\n{article}\n", encoding="utf-8")

print("\n" + "=" * 72)
print("最终文章")
print("=" * 72)
print(article)

print("\n累计处理过的问题")
print("=" * 72)
if not known_issues:
    print("本次循环没有采纳任何新增问题。")
else:
    for i, issue in enumerate(known_issues, 1):
        print(f"{i}. [{issue.get('severity', '未标注')}] {issue.get('issue', '')}")
        print(f"   建议：{issue.get('suggestion', '')}")

print("\n每轮修改摘要")
print("=" * 72)
if not revision_history:
    print("没有发生文章改写。")
else:
    for item in revision_history:
        print(f"第 {item['round']} 轮：")
        print("  新增问题数：", len(item.get("new_issues", [])))
        print("  修改记录数：", len(item.get("change_log", [])))

print("\n保存结果：")
print("JSON：", result_path.relative_to(COURSE_ROOT))
print("Markdown：", article_path.relative_to(COURSE_ROOT))

print("\n课堂观察点：")
print("1. 两个评审 Agent 的关注点不同，发现的问题也不同。")
print("2. 判断 Agent 控制停止条件，避免无限修改。")
print("3. 修改 Agent 只处理新增问题，known_issues 就是这个流程的 memory。")