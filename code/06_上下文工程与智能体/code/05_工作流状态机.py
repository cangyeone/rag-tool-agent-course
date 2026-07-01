"""05 工作流状态机。

工作流和 Agent 的区别：
Agent 更像"动态安排任务"，工作流更像"固定流水线"。
本脚本按节点顺序展示 state 在 可视化工作流中的作用。
"""

import json

print("05 工作流状态机")
print("=" * 72)

# state 是工作流里的共享记录本。每个节点读它，也往里面写东西。
state = {
    "question": "服务开始前 3 小时退款，客服应该怎么提示？",
    "trace": [],
}
print("一、开始节点")
print(json.dumps(state, ensure_ascii=False, indent=2))

# 节点 1：意图识别。
q = state["question"]
if "退款" in q:
    state["intent"] = "退款咨询"
elif "身份证" in q:
    state["intent"] = "证件问题"
elif "没票" in q or "候补申请" in q:
    state["intent"] = "库存与候补申请"
else:
    state["intent"] = "通用咨询"
state["trace"].append({"node": "意图识别", "intent": state["intent"]})

# 节点 2：槽位抽取。
state["slots"] = {}
if "3 小时" in q or "3小时" in q:
    state["slots"]["hours_before_departure"] = 3
if state["intent"] == "退款咨询":
    state["slots"].setdefault("hours_before_departure", 3)
state["trace"].append({"node": "槽位抽取", "slots": state["slots"]})

# 节点 3：知识检索。
state["policy"] = {"keyword": "退款", "hits": ["退款和手续费以订单页面规则为准，客服不能承诺固定金额。"]}
state["trace"].append({"node": "知识检索", "policy": state["policy"]})

# 节点 4：工具调用。
hours = state["slots"].get("hours_before_departure", 3)
if hours <= 4:
    risk = "临近服务开始，规则和手续费更需要以订单页为准。"
else:
    risk = "仍需查看订单页规则。"
state["tool_result"] = {"hours_before_departure": hours, "risk": risk}
state["trace"].append({"node": "工具调用", "tool_result": state["tool_result"]})

# 节点 5：回答生成。
state["draft_answer"] = (
    f"当前距离服务开始约 {hours} 小时，退款前请先查看订单页面规则。"
    "手续费和是否可退需要以官方页面显示为准，客服不能承诺固定金额。"
)
state["trace"].append({"node": "回答生成", "draft_answer": state["draft_answer"]})

# 节点 6：质检与修正。
state["quality"] = {
    "has_boundary": "官方页面" in state["draft_answer"] and "不能承诺" in state["draft_answer"],
    "has_policy": bool(state["policy"]["hits"]),
    "has_tool_result": bool(state["tool_result"]),
    "has_safe_wording": not any(w in state["draft_answer"] for w in ["保证", "一定", "肯定", "确定可以", "绝对"]),
}
state["final_answer"] = state["draft_answer"]
if not state["quality"]["has_boundary"]:
    state["final_answer"] += " 最终以 官方页面为准。"
if not state["quality"]["has_safe_wording"]:
    state["final_answer"] = state["final_answer"].replace("保证", "建议").replace("一定", "通常").replace("肯定", "一般")
state["trace"].append({"node": "质检与修正", "quality": state["quality"]})

print("\n二、完整工作流 trace")
print(json.dumps(state["trace"], ensure_ascii=False, indent=2))

print("\n三、最终回答")
print(state["final_answer"])

print("\n四、和 可视化工具 的对应关系")
print("Start -> 意图识别/条件判断 -> 知识检索 -> 工具节点 -> LLM 节点 -> Answer")

print("\n五、质检维度说明")
print("   1. has_boundary: 回答是否包含官方页面引导")
print("   2. has_policy: 是否检索到了业务规则")
print("   3. has_tool_result: 工具节点是否正常返回")
print("   4. has_safe_wording: 是否避免了承诺性用词（保证/一定/肯定/绝对）")