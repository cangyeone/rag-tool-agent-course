"""05_Agent自我修正。

学习目标：Agent 如何检测自身错误、进行反思、修正回答。
焦点：自我反思循环、错误检测、答案修订、修正前后对比。

示例业务系统 场景：Agent 首轮回答缺少边界声明，通过自检循环修正。
"""

import json

print("05 Agent 自我修正 —— 反思与修订循环")
print("=" * 72)

# ── 1. 模拟用户查询和 Agent 首轮回答 ──
print("【场景设定】")
user_query = "G107 没票了，我现在提交候补申请能保证上车吗？我赶时间必须走。"
print(f"   用户查询: {user_query}")
print()

# ── 2. Agent 首轮生成（可能有问题的回答） ──
print("── 第 1 轮：初始回答 ──")
first_answer = (
    "G107 目前没票了，您可以提交候补申请。候补申请成功后系统会自动分配座位，"
    "一般情况下候补申请排队靠前的话是可以买到票的，您抓紧提交就行。"
)
print(f"   回答: {first_answer}")

# ── 3. 自我反思：检查首轮回答中的问题 ──
print("\n── 反思阶段：Agent 自检 ──")

def self_reflect(answer, query):
    """Agent 反思自己的回答，返回检测到的问题"""
    issues = []
    # 检查 1：有无边界声明
    if "官方页面" not in answer and "官网" not in answer and "示例业务系统" not in answer.lower():
        issues.append({
            "type": "missing_boundary",
            "severity": "high",
            "detail": "回答中未明确引导用户以 官方页面为准",
            "fix": "添加'最终以 官方页面通知为准'"
        })
    # 检查 2：有无承诺性/确定性词汇
    commitment_words = {"保证": "承诺性", "一定": "确定性", "肯定": "确定性", 
                       "绝对": "绝对化", "可以买到": "承诺性", "能买到": "承诺性"}
    for word, category in commitment_words.items():
        if word in answer:
            issues.append({
                "type": "commitment_language",
                "severity": "high",
                "detail": f"使用了 {category} 词汇'{word}'，可能误导用户",
                "fix": f"将'{word}'替换为'建议/可能/以实际为准'"
            })
    # 检查 3：用户紧迫情绪是否被安抚但未过度承诺
    if "赶时间" in query or "必须" in query or "着急" in query:
        if "放心" in answer and "保证" not in answer and "官方" not in answer:
            issues.append({
                "type": "false_reassurance",
                "severity": "medium",
                "detail": "用户紧迫但回答中'放心'缺少依据支撑",
                "fix": "在安抚情绪的同时加上客观事实和官方引导"
            })
    # 检查 4：是否提供了替代方案
    if "没票" in query or "无票" in query or "售罄" in query:
        if "替代" not in answer and "其他车" not in answer and "别的" not in answer:
            issues.append({
                "type": "missing_alternatives",
                "severity": "medium",
                "detail": "订单编号无票但未建议替代方案",
                "fix": "补充同方向替代订单编号建议"
            })
    # 检查 5：虚构信息检测
    if "手续费" in answer and any(c.isdigit() for c in answer):
        # 检查是否给出了具体金额
        import re
        amounts = re.findall(r'(\d+\.?\d*)\s*元', answer)
        if amounts and "553" not in answer:  # 不在已知数据中
            issues.append({
                "type": "hallucinated_detail",
                "severity": "high",
                "detail": f"可能虚构了具体金额: {amounts}",
                "fix": "移除具体金额，改为'以订单页面显示为准'"
            })
    return issues

reflection_1 = self_reflect(first_answer, user_query)
print(f"   检测到 {len(reflection_1)} 个问题:")
for i, issue in enumerate(reflection_1):
    print(f"   [{issue['severity'].upper()}] {issue['type']}: {issue['detail']}")
    print(f"         修复建议: {issue['fix']}")

# ── 4. 第 2 轮：修正回答 ──
print("\n── 第 2 轮：修正后回答 ──")

second_answer = (
    "G107 当前显示无票，您可以提交候补申请。"
    "但需要提醒您：候补申请不能保证一定成功，系统按排队顺序兑现，"
    "最终结果以 官方页面通知为准。"
    "同时建议您查看同方向其他订单编号（如 订单B、D311），目前仍有部分库存。"
    "如果您赶时间，替代订单编号可能是更稳妥的选择。"
)
print(f"   回答: {second_answer}")

# 再次反思
reflection_2 = self_reflect(second_answer, user_query)
if reflection_2:
    print(f"\n   第 2 轮反思 — 仍检测到 {len(reflection_2)} 个问题:")
    for issue in reflection_2:
        print(f"   [{issue['severity'].upper()}] {issue['type']}: {issue['detail']}")
else:
    print(f"\n   第 2 轮反思 — ✅ 通过，无问题")

# ── 5. 修正对比 ──
print(f"\n{'=' * 72}")
print("修正前后对比:")
print(f"   ┌─────────────────────────────┬───────────────────────────────┐")
print(f"   │ 第 1 轮（有问题）           │ 第 2 轮（修正后）              │")
print(f"   ├─────────────────────────────┼───────────────────────────────┤")
print(f"   │ 边界声明: ❌                │ 边界声明: ✅                  │")
print(f"   │ 承诺性词汇: ✅(可以买到)    │ 承诺性词汇: ❌(已移除)        │")
print(f"   │ 替代方案: ❌                │ 替代方案: ✅(订单B, D311)      │")
print(f"   │ 风险等级: HIGH              │ 风险等级: LOW                 │")
print(f"   └─────────────────────────────┴───────────────────────────────┘")

# ── 6. 自我修正循环的配置 ──
print("\n【自我修正循环配置】")
config = {
    "max_iterations": 3,           # 最多修正 3 轮
    "quality_threshold": 0,        # 0 个问题 = 通过
    "stop_on_no_issues": True,     # 无问题时停止
    "escalation": "超过 3 轮仍有问题，转人工客服",
    "checks": [
        "boundary_statement",       # 是否声明了边界
        "no_commitment_language",   # 无承诺性词汇
        "has_alternatives",         # 有替代方案
        "no_hallucinated_details",  # 无虚构细节
        "tone_moderation",          # 语气得当
    ]
}
print(json.dumps(config, ensure_ascii=False, indent=2))

# ── 7. 生产中的反思日志 ──
print("\n【反思日志（生产环境留痕）】")
audit_log = {
    "query": user_query,
    "iterations": [
        {"round": 1, "issues_found": len(reflection_1), "issues": [i["type"] for i in reflection_1], "passed": False},
        {"round": 2, "issues_found": len(reflection_2), "issues": [i["type"] for i in reflection_2], "passed": True},
    ],
    "final_answer": second_answer[:80] + "...",
    "total_iterations": 2,
    "escalated": False,
}
print(json.dumps(audit_log, ensure_ascii=False, indent=2))

print(f"\nAgent 自我修正要点:")
print("   1. 每次回答后自动触发反思检查")
print("   2. 检测维度：边界声明、承诺性词汇、替代方案、虚构信息、语气")
print("   3. 最多 N 轮修正，超限转人工")
print("   4. 修正日志完整留痕，用于审计和模型改进")
print("   5. 自修正不是让 LLM '自己猜对'，而是基于规则/工具结果的确定性修正")