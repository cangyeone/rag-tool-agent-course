"""02_计划器_Planner — 生成计划 → 执行 → 观察 → 必要时修正计划。

学习目标：理解 Planner 如何把复杂问题拆成多个子任务，动态调整执行策略。
"""

import json

print("02_计划器_Planner")
print("=" * 72)
print("Planner = 大任务的「拆解器」：把一句话的复杂需求拆成有序的工具调用计划。")
print()

question = "帮我查一下明天北京南到上海虹桥的票，如果有 G107 就订标准服务，没有就候补申请 G109。"

print(f"【用户需求】{question}\n")

# ========== Phase 1: 生成计划 ==========
print("【Phase 1: 生成执行计划】")
plan = [
    {"step": 1, "tool": "query_tickets",
     "args": {"from_station": "北京南", "to_station": "上海虹桥", "date": "2026-06-20"},
     "reason": "先拿到所有可用订单编号信息，判断 G107/G109 状态",
     "depends_on": [], "status": "pending"},
    {"step": 2, "tool": "check_condition",
     "args": {"condition": "G107 有票"},
     "reason": "根据 step1 结果判断走订票还是候补申请分支",
     "depends_on": [1], "status": "pending"},
    {"step": 3, "tool": "order_ticket",
     "args": {"train_no": "G107", "seat_type": "标准服务", "date": "2026-06-20"},
     "reason": "G107 有票则直接下单（条件分支A）",
     "depends_on": [1, 2], "status": "pending"},
    {"step": 4, "tool": "submit_waitlist",
     "args": {"train_no": "G109", "seat_type": "标准服务", "date": "2026-06-20"},
     "reason": "G107 无票则候补申请 G109（条件分支B）",
     "depends_on": [1, 2], "status": "pending"},
]
for p in plan:
    deps = f"依赖步骤: {p['depends_on']}" if p['depends_on'] else "无依赖"
    print(f"  Step {p['step']}: {p['tool']} | {p['reason']} | {deps}")

# ========== Phase 2: 逐步执行，动态修正 ==========
print("\n【Phase 2: 执行计划 + 动态修正】")
execution_log = []

# 模拟执行
simulated_ticket_data = {"trains": [
    {"no": "G107", "remaining": 0, "status": "售罄"},
    {"no": "G109", "remaining": 5, "status": "有票"},
]}

for step in plan:
    if step["step"] == 1:
        result = simulated_ticket_data
        step["status"] = "done"
        execution_log.append({"step": 1, "result": f"G107 售罄, G109 有 5 张"})

    elif step["step"] == 2:
        g107_available = any(t["no"] == "G107" and t["remaining"] > 0
                             for t in simulated_ticket_data["trains"])
        result = {"condition": "G107 有票", "result": g107_available}
        step["status"] = "done"
        execution_log.append({"step": 2, "result": f"条件判定: G107 无票 → 走候补申请分支"})

    elif step["step"] == 3:
        # G107 无票，所以 skip
        step["status"] = "skipped"
        execution_log.append({"step": 3, "result": "跳过：G107 无票，不执行订票"})

    elif step["step"] == 4:
        result = {"waitlist_id": "WL20260619001", "train_no": "G109",
                  "status": "候补申请已提交", "position": 3}
        step["status"] = "done"
        execution_log.append({"step": 4, "result": f"候补申请成功，排队第 {result['position']} 位"})

# ========== Phase 3: 计划修正演示 ==========
print("\n【Phase 3: 计划修正 —— 当工具执行失败时】")
revision_scenario = {
    "original_step": 1,
    "error": "query_tickets 接口返回超时",
    "revision": 'Planner 决定降级：先查缓存数据，同时提示用户「数据可能有延迟」',
    "new_plan": [
        {"step": "1b", "tool": "query_tickets_cache",
         "desc": "从 Redis 缓存拿最近一次查询结果（可能有 5 分钟延迟）"}
    ]
}
print(f"  原步骤 {revision_scenario['original_step']} 出错: {revision_scenario['error']}")
print(f"  Planner 修正: {revision_scenario['revision']}")
for s in revision_scenario["new_plan"]:
    print(f"    新增步骤 {s['step']}: {s['tool']} ({s['desc']})")

# ========== Phase 4: 执行总结 ==========
print("\n【执行总结】")
print(f"  计划总步骤: {len(plan)}")
done = sum(1 for s in plan if s["status"] == "done")
skipped = sum(1 for s in plan if s["status"] == "skipped")
print(f"  已执行: {done}, 已跳过: {skipped}")
print(f"  执行日志:")
for log_entry in execution_log:
    print(f"    Step {log_entry['step']}: {log_entry['result']}")

# 最终回答
print(f"\n【最终回答】")
print(f"  G107 明日已售罄，已为您提交 G109 候补申请订单（排队第 3 位）。")
print(f"  候补申请结果以 官方通知为准。")

print("\nPlanner 设计要点：")
print("1. 计划是一个 DAG（有向无环图）：step 之间有依赖关系")
print("2. 执行过程中如果某步失败，Planner 应该能生成修正计划")
print("3. 条件分支（if/else）在计划层面管理，不靠模型「临场发挥」")