"""04 多工具 Agent。

多工具 Agent = 意图识别 → 工具计划 → 并行/串行执行 → 结果整合 → 质检。
本脚本拆成可观察的 6 步流水线，每步输出格式清晰可读。
"""

import json
import time
from collections import OrderedDict

print("04 多工具 Agent —— 6 步可观察流水线")
print("=" * 72)

# ── 第 0 步：模拟工具注册表 ──
tools_registry = OrderedDict([
    ("search_station", {"desc": "根据站名查编码和拼音", "params": ["station_name"]}),
    ("query_train", {"desc": "根据订单编号查停靠站、库存状态", "params": ["train_no", "date"]}),
    ("search_policy", {"desc": "查退变更、候补申请等业务规则", "params": ["keyword", "top_k"]}),
    ("check_alternatives", {"desc": "查同方向替代订单编号或中转方案", "params": ["from_station", "to_station", "date"]}),
    ("risk_assess", {"desc": "对敏感回答做风险打分（幻觉/承诺/越权）", "params": ["answer_draft"]}),
])
print("\n【0. 工具注册表】共 {} 个工具已就绪".format(len(tools_registry)))
for name, meta in tools_registry.items():
    print(f"   ├─ {name}: {meta['desc']}")

# ── 第 1 步：意图识别 ──
user_question = "G107 北京南到上海虹桥标准服务没票，能候补申请吗？还有没有别的车能走？"
print(f"\n【1. 意图识别】\n   用户输入: {user_question}")

def classify_intent(text):
    scores = {}
    if any(w in text for w in ["没票", "无票", "售罄", "满了"]):
        scores["库存查询"] = 0.9
    if any(w in text for w in ["候补申请", "排队", "等待"]):
        scores["候补申请咨询"] = 0.85
    if any(w in text for w in ["别的车", "其他车", "替代", "还有"]):
        scores["替代订单编号"] = 0.8
    if any(w in text for w in ["退款", "变更", "退款"]):
        scores["退变更"] = 0.7
    if not scores:
        return {"primary": "通用咨询", "confidence": 0.5, "all": {}}
    primary = max(scores, key=scores.get)
    return {"primary": primary, "confidence": scores[primary], "all": scores}

intent_result = classify_intent(user_question)
print(f"   主意图: {intent_result['primary']} (置信度 {intent_result['confidence']:.0%})")
print(f"   候选意图: {json.dumps(intent_result['all'], ensure_ascii=False)}")

# ── 第 2 步：工具计划 ──
print(f"\n【2. 工具计划】")
tool_plan = []
if intent_result["primary"] == "库存查询":
    tool_plan.append({"id": "P1", "tool": "query_train", "params": {"train_no": "G107", "date": "2026-06-19"}, "depends_on": [], "why": "查当前订单编号库存"})
if intent_result["primary"] == "候补申请咨询":
    tool_plan.append({"id": "P2", "tool": "search_policy", "params": {"keyword": "候补申请规则", "top_k": 3}, "depends_on": [], "why": "先查候补申请规则，避免承诺"})
if intent_result.get("all", {}).get("替代订单编号", 0) > 0.5:
    tool_plan.append({"id": "P3", "tool": "check_alternatives", "params": {"from_station": "北京南", "to_station": "上海虹桥", "date": "2026-06-19"}, "depends_on": [], "why": "帮用户查替代方案"})
# 输出依赖：P2 的结果可能影响 P1 的解释
tool_plan.append({"id": "P4", "tool": "risk_assess", "params": {"answer_draft": "待生成"}, "depends_on": ["P1", "P2", "P3"], "why": "生成回答后质检风险"})

print(f"   共规划 {len(tool_plan)} 个工具调用")
for item in tool_plan:
    deps = f" 依赖: {item['depends_on']}" if item['depends_on'] else " 无依赖"
    print(f"   ├─ [{item['id']}] {item['tool']}({json.dumps(item['params'], ensure_ascii=False)}){deps}")
    print(f"   │   原因: {item['why']}")

# ── 第 3 步：执行（并行/串行） ──
print(f"\n【3. 工具执行】")

# 模拟工具执行函数
def exec_tool(tool_name, params):
    time.sleep(0.01)  # 模拟延迟
    if tool_name == "query_train":
        return {"train_no": params["train_no"], "date": params.get("date", ""), "seats": {"标准服务": 0, "高级服务": 3, "商务座": 2}, "status": "标准服务已售罄"}
    elif tool_name == "search_policy":
        return {"keyword": params["keyword"], "hits": ["候补申请不能保证成功，兑现结果以官方页面通知为准。", "候补申请订单可在服务开始前 2 小时提交，每人最多 4 个候补申请需求。", "候补申请不收取额外费用，仅支付预付款。"], "source": "示例业务系统 业务规则库"}
    elif tool_name == "check_alternatives":
        return {"from": params["from_station"], "to": params["to_station"], "date": params.get("date", ""), "alternatives": [{"train_no": "订单B", "seats": {"标准服务": 12}}, {"train_no": "G121", "seats": {"标准服务": 5}}, {"train_no": "D311", "seats": {"标准服务": 20}}]}
    elif tool_name == "risk_assess":
        answer = params.get("answer_draft", "")
        issues = []
        if "保证" in answer or "一定" in answer or "肯定" in answer:
            issues.append("含承诺性词汇")
        if "手续费 X 元" in answer or "退款 X 元" in answer:
            issues.append("虚构具体金额")
        if "官方页面" not in answer and "官方" not in answer:
            issues.append("缺少官方页面引导")
        return {"risk_level": "HIGH" if issues else "LOW", "issues": issues}
    return {"error": "unknown tool"}

# 按依赖关系执行：无依赖的并行，有依赖的串行
exec_results = {}
# 第一波：无依赖的工具并行执行
wave1 = [p for p in tool_plan if not p["depends_on"]]
print(f"   ⚡ 第 1 波（并行）: {[p['id'] for p in wave1]}")
for p in wave1:
    start = time.time()
    exec_results[p["id"]] = exec_tool(p["tool"], p["params"])
    elapsed = (time.time() - start) * 1000
    print(f"   ├─ [{p['id']}] {p['tool']} → {elapsed:.1f}ms")
    print(f"   │   结果: {json.dumps(exec_results[p['id']], ensure_ascii=False)}")

# 第二波：依赖第一波结果的
wave2 = [p for p in tool_plan if p["depends_on"] and all(d in exec_results for d in p["depends_on"])]
if wave2:
    print(f"   ⚡ 第 2 波（依赖前序）: {[p['id'] for p in wave2]}")
    for p in wave2:
        exec_results[p["id"]] = exec_tool(p["tool"], p["params"])
        print(f"   ├─ [{p['id']}] {p['tool']} → 完成")
        print(f"   │   结果: {json.dumps(exec_results[p['id']], ensure_ascii=False)}")

# ── 第 4 步：结果整合 ──
print(f"\n【4. 结果整合】")
# 从工具结果中提取关键信息
train_info = exec_results.get("P1", {})
policy_info = exec_results.get("P2", {})
alt_info = exec_results.get("P3", {})
risk_info = exec_results.get("P4", {})

# 结构化整合
synthesis = {
    "订单编号状态": f"{train_info.get('train_no', 'G107')} 标准服务已售罄，高级服务余 {train_info.get('seats', {}).get('高级服务', '?')} 张",
    "候补申请规则": policy_info.get("hits", ["无"])[0] if policy_info.get("hits") else "无相关规则",
    "替代方案": [f"{a['train_no']}(标准服务余{a['seats'].get('标准服务', '?')}张)" for a in alt_info.get("alternatives", [])],
    "风险评估": f"{risk_info.get('risk_level', 'N/A')} — {', '.join(risk_info.get('issues', ['无问题']))}"
}
print(json.dumps(synthesis, ensure_ascii=False, indent=2))

# ── 第 5 步：生成回答 ──
print(f"\n【5. 生成回答】")
final_answer = (
    f"G107 标准服务当前已售罄，高级服务还有少量库存。"
    f"如果坚持标准服务，可以尝试候补申请——候补申请不能保证成功，最终以 官方页面通知为准。"
    f"同时为您查到同方向替代订单编号：{'、'.join(synthesis['替代方案'])}，可登录 示例业务系统 查看实时库存。"
)
print(f"   {final_answer}")

# ── 第 6 步：质检 ──
print(f"\n【6. 质量检查】")
qa_report = {
    "有无边界声明": "官方页面" in final_answer,
    "有无承诺性词汇": not any(w in final_answer for w in ["保证", "一定", "肯定"]),
    "是否覆盖替代方案": len(synthesis["替代方案"]) > 0,
    "是否引用规则依据": len(policy_info.get("hits", [])) > 0,
    "工具调用完整度": f"{len(exec_results)}/{len(tool_plan)}",
    "风险评估级别": risk_info.get("risk_level", "N/A"),
}
all_pass = all(v if isinstance(v, bool) else True for v in qa_report.values())
qa_report["总体结论"] = "✅ 通过" if all_pass else "⚠️ 需人工复核"
print(json.dumps(qa_report, ensure_ascii=False, indent=2))

# ── Agent Harness 汇总 ──
print(f"\n{'=' * 72}")
print("Agent Harness 执行摘要:")
print(f"   输入长度: {len(user_question)} 字符")
print(f"   工具调用: {len(tool_plan)} 次 (并行 {len(wave1)} + 串行 {len(wave2)})")
print(f"   总耗时: ~{sum(10 for _ in tool_plan)}ms (模拟)")
print(f"   质检结果: {qa_report['总体结论']}")