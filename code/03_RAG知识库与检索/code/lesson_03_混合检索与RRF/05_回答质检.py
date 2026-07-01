"""05_回答质检。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
学习目标：掌握回答质量的多项评分方法，包括事实准确性、
         相关性、完整性、安全性，实现 Pass/Fail 质检机制。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import math
import jieba
import time
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

# ── 从课程目录加载真实文档 ──
def _load_real_docs():
    real_docs = []
    sample_files = [
        ("code/01_大模型基础/README.md", 350),
        ("code/02_模型接口与指令设计/README.md", 350),
        ("code/03_RAG知识库与检索/README.md", 350),
        ("README.md", 350),
    ]
    for rel_path, max_chars in sample_files:
        file_path = COURSE_ROOT / rel_path
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")[:max_chars]
            title = rel_path.split("/")[1] if rel_path.startswith("code/") else "课程总览"
            real_docs.append({"title": title, "source": rel_path, "content": content})
    return real_docs

docs = _load_real_docs()


print("=" * 72)
print("05_回答质检 —— 多项评分与 Pass/Fail 机制")
print("=" * 72)

# ═══════════════════════════════════════════
# 一、质检框架总览
# ═══════════════════════════════════════════
print("\n一、质检五维度框架")

quality_dims = [
    ("事实准确性", "Factuality", "回答是否与检索到的文档内容一致，不编造信息"),
    ("相关性", "Relevance", "回答是否直接回应了用户的问题"),
    ("完整性", "Completeness", "是否覆盖了问题的所有子问题"),
    ("安全性", "Safety", "是否包含敏感/误导/有害内容"),
    ("引用可追溯", "Citation", "是否标注了信息来源"),
]

print(f"\n  {'维度':<14}{'英文':<16}{'检查内容'}")
print("  " + "-" * 56)
for cn, en, desc in quality_dims:
    print(f"  {cn:<14}{en:<16}{desc}")

# ═══════════════════════════════════════════
# 二、准备测试回答（好、中、差三档）
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("二、准备测试回答（好/中/差三档）")
print("━" * 60)

# 源文档内容
source_doc = docs[0]

test_answers = [
    {
        "id": "A-好",
        "question": "候补申请一定能成功吗？",
        "answer": (
            f"根据《{source_doc['title']}》，"
            f"{source_doc['content']}"
            f"建议您以 官方页面显示的兑现状态为准。"
        ),
        "sources": [source_doc["title"]],
        "expected_good": True,
    },
    {
        "id": "B-中（缺少引用）",
        "question": "候补申请一定能成功吗？",
        "answer": "候补申请不一定能成功，兑现结果取决于退款变更和新增库存。",
        "sources": [],
        "expected_good": False,
    },
    {
        "id": "C-差（编造信息）",
        "question": "候补申请一定能成功吗？",
        "answer": "候补申请100%保证成功，只要您排队等就行，不用看示例业务系统。",
        "sources": [],
        "expected_good": False,
    },
    {
        "id": "D-差（答非所问）",
        "question": "候补申请一定能成功吗？",
        "answer": "示例服务点是示例线路业务服务的重要枢纽站，每天有多趟服务流程经过。",
        "sources": [docs[2]["title"]],
        "expected_good": False,
    },
]

for ans in test_answers:
    print(f"\n  [{ans['id']}] 问题：{ans['question']}")
    print(f"  回答：{ans['answer']}")
    print(f"  引用来源：{ans['sources']}")

# ═══════════════════════════════════════════
# 三、逐维度评分函数
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("三、逐维度评分函数实现")
print("━" * 60)

def score_factuality(answer, source_docs):
    """事实准确性：回答内容是否能在源文档中找到支撑。
    返回 (分数 0~100, 证据列表, 问题列表)
    """
    score = 100
    issues = []
    evidence = []

    for doc in source_docs:
        # 简化检测：回答中的关键句子是否出现在源文档中
        content = doc["content"]
        # 取回答的关键短句逐一检查
        sentences = answer.replace("。", "。\n").split("\n")
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 5:
                continue
            # 检查这句话的核心词是否在文档中
            overlap = sum(1 for ch in sent if ch in content)
            ratio = overlap / len(sent) if sent else 0
            if ratio > 0.3:
                evidence.append((sent[:40], doc["title"], ratio))
            elif "根据" not in sent and "建议" not in sent:
                issues.append(sent[:40])

    if issues:
        score -= min(80, len(issues) * 20)
    return max(0, score), evidence, issues

def score_relevance(answer, question):
    """相关性：回答是否与问题相关。"""
    q_keywords = set(w for w in jieba.cut(question) if len(w.strip()) >= 2)
    a_keywords = set(w for w in jieba.cut(answer) if len(w.strip()) >= 2)
    if not q_keywords:
        return 100, [], []

    overlap = q_keywords & a_keywords
    ratio = len(overlap) / len(q_keywords)
    score = min(100, int(ratio * 100))
    missing = q_keywords - a_keywords
    issues = [f"未回应关键词：{list(missing)[:3]}"] if missing and ratio < 0.3 else []
    return score, list(overlap), issues

def score_completeness(answer, question):
    """完整性：是否覆盖了问题的关键方面。"""
    score = 100
    # 检查是否包含：结论、依据、建议
    checks = []
    if any(w in answer for w in ["根据", "依据", "按照", "规定"]):
        checks.append("有依据引用")
    else:
        checks.append("缺少依据引用")
        score -= 30

    if any(w in answer for w in ["建议", "可以", "推荐", "请"]):
        checks.append("有操作建议")
    else:
        checks.append("缺少操作建议")
        score -= 20

    if any(w in answer for w in ["示例业务系统", "官方", "页面"]):
        checks.append("有官方引导")
    else:
        checks.append("缺少官方引导")
        score -= 30

    # 检查回答长度
    if len(answer) < 20:
        checks.append("回答过短")
        score -= 20

    return max(0, score), checks, []

def score_safety(answer):
    """安全性：是否包含风险内容。"""
    score = 100
    issues = []

    dangerous_patterns = [
        ("保证", "使用了绝对化保证用词"),
        ("100%", "使用了100%绝对表述"),
        ("一定", "过于绝对的承诺（结合上下文）"),
        ("不用看", "鼓励忽视官方信息"),
    ]

    for pattern, reason in dangerous_patterns:
        if pattern in answer:
            score -= 25
            issues.append(f"危险模式'{pattern}'：{reason}")

    return max(0, score), issues

def score_citation(answer, sources):
    """引用可追溯：是否标注了信息来源。"""
    if not sources:
        return 0, ["未标注任何来源"]
    cited = sum(1 for s in sources if s in answer)
    if cited == 0:
        return 30, [f"来源 {sources} 未在回答中明确引用"]
    ratio = cited / len(sources)
    score = min(100, int(ratio * 100))
    return score, [f"已引用 {cited}/{len(sources)} 个来源"]

# ═══════════════════════════════════════════
# 四、执行多项质检
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("四、执行质检（逐条评分）")
print("━" * 60)

def quality_check(answer, question, sources, source_docs_list):
    """综合质检函数。"""
    results = {}
    all_issues = []

    # 事实准确性
    fact_score, fact_evidence, fact_issues = score_factuality(
        answer, source_docs_list)
    results["事实准确性"] = fact_score
    all_issues.extend(fact_issues)

    # 相关性
    rel_score, rel_overlap, rel_issues = score_relevance(answer, question)
    results["相关性"] = rel_score
    all_issues.extend(rel_issues)

    # 完整性
    comp_score, comp_checks, _ = score_completeness(answer, question)
    results["完整性"] = comp_score

    # 安全性
    safe_score, safe_issues = score_safety(answer)
    results["安全性"] = safe_score
    all_issues.extend(safe_issues)

    # 引用可追溯
    cite_score, cite_notes = score_citation(answer, sources)
    results["引用可追溯"] = cite_score
    all_issues.extend(cite_notes)

    # 综合得分
    weights = {"事实准确性": 0.35, "相关性": 0.25, "完整性": 0.15,
               "安全性": 0.15, "引用可追溯": 0.10}
    weighted_score = sum(results[k] * weights[k] for k in results)

    # Pass/Fail 判定
    # 安全性一票否决（< 50 直接 fail）
    # 其他维度 < 60 也 fail
    if results["安全性"] < 50:
        verdict = "FAIL（安全一票否决）"
    elif weighted_score >= 70 and all(v >= 50 for v in results.values()):
        verdict = "PASS"
    else:
        verdict = "FAIL（综合分或单项分不达标）"

    return verdict, weighted_score, results, all_issues

# 执行所有测试回答的质检
summary = []
for ta in test_answers:
    # 找到相关源文档
    relevant_docs = [d for d in docs if d["title"] in ta["sources"]]
    if not relevant_docs:
        relevant_docs = docs  # 兜底：用所有文档检查

    verdict, score, results, issues = quality_check(
        ta["answer"], ta["question"], ta["sources"], relevant_docs)

    summary.append((ta["id"], verdict, score))

    print(f"\n{'─' * 56}")
    print(f"  [{ta['id']}] {ta['question']}")
    print(f"  回答：{ta['answer'][:60]}...")
    print(f"\n  {'维度':<14}{'得分':<8}{'判定'}")
    print("  " + "-" * 34)
    for dim, s in results.items():
        status = "✓" if s >= 60 else "✗"
        bar = "█" * (s // 10) + "░" * (10 - s // 10)
        print(f"  {dim:<14}{s:<8}{status} {bar}")
    print(f"  {'─' * 34}")
    print(f"  加权综合分：{score:.1f}")
    print(f"  判定：{verdict}")
    if issues:
        print(f"  问题：")
        for iss in issues[:5]:
            print(f"    - {iss}")

# ═══════════════════════════════════════════
# 五、质检汇总报告
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("五、质检汇总报告")
print("━" * 60)

print(f"\n  {'编号':<12}{'判定':<24}{'综合分':<10}{'期望'}")
print("  " + "-" * 48)
# 建立 ID -> expected_good 的映射
expected_map = {ta["id"]: ta["expected_good"] for ta in test_answers}
passed = 0
for sid, verdict, score in summary:
    expected_val = expected_map.get(sid, False)
    match = "✓" if (verdict == "PASS") == expected_val else "✗"
    if verdict == "PASS":
        passed += 1
    print(f"  {sid:<12}{verdict:<24}{score:<10.1f}{match}")

print(f"\n  通过率：{passed}/{len(test_answers)}（{passed/len(test_answers)*100:.0f}%）")

print("\n" + "=" * 72)
print("学习要点：")
print("  1. 回答质检 ≠ 只看对不对，需要五维度综合评分")
print("  2. 安全性采用一票否决（< 50 直接 FAIL）")
print("  3. 事实准确性权重最高（35%），核心是「不编造」")
print("  4. 质检结果可用于线上拦截 + 线下复盘")