"""03_多工具协同。

学习目标：在单次对话中编排 3+ 个工具，理解工具依赖图和并行执行。
焦点：工具依赖图、并行 vs 串行、结果聚合、工具冲突处理。

示例业务系统 场景：处理一个同时涉及查服务编号、查规则、查替代方案、风险评估的复杂查询。
"""

import json
import time
from collections import defaultdict

print("03 多工具协同 —— 工具依赖图与并行执行")
print("=" * 72)

# ── 1. 定义 5 个工具 ──
tools = {
    "service_point_lookup": {
        "desc": "服务点名→编码/拼音/所属城市",
        "input": ["service_point_name"],
        "output": ["service_point_code", "pinyin", "city"],
        "cost_ms": 30,
    },
    "train_query": {
        "desc": "服务编号→停靠站、时刻、库存",
        "input": ["order_id", "date"],
        "output": ["stops", "schedule", "seats_available"],
        "cost_ms": 80,
    },
    "policy_search": {
        "desc": "关键词→业务规则条文",
        "input": ["keyword", "top_k"],
        "output": ["rules", "source"],
        "cost_ms": 50,
    },
    "alternative_finder": {
        "desc": "起终点→同方向替代服务编号",
        "input": ["from_code", "to_code", "date"],
        "output": ["items", "seat_summary"],
        "cost_ms": 100,
        "depends_on": ["service_point_lookup"],
    },
    "risk_checker": {
        "desc": "回答草稿→风险检测",
        "input": ["draft_answer", "context"],
        "output": ["risk_level", "issues", "fix_suggestions"],
        "cost_ms": 20,
        "depends_on": ["train_query", "policy_search", "alternative_finder"],
    },
}

print("【工具注册表】")
for name, meta in tools.items():
    deps = meta.get("depends_on", [])
    dep_str = f" → 依赖: {deps}" if deps else ""
    print(f"   {name}: {meta['desc']} ({meta['cost_ms']}ms){dep_str}")

# ── 2. 构建依赖图 ──
print("\n【工具依赖图】")
# 画出 DAG
deps = {}
for name, meta in tools.items():
    deps[name] = meta.get("depends_on", [])
# 拓扑排序
def topo_sort(deps_graph):
    in_degree = {n: 0 for n in deps_graph}
    for n in deps_graph:
        for dep in deps_graph[n]:
            in_degree[n] += 1
    sorted_nodes = []
    remaining = set(deps_graph.keys())
    while remaining:
        zero_degree = [n for n in remaining if in_degree[n] == 0]
        if not zero_degree:
            break
        for n in sorted(zero_degree):
            remaining.remove(n)
            sorted_nodes.append(n)
            for other in deps_graph:
                if n in deps_graph[other]:
                    in_degree[other] -= 1
    # 分组：同一批无依赖的可以并行
    waves = []
    visited = set()
    in_deg = {n: len(deps_graph[n]) for n in deps_graph}
    while len(visited) < len(deps_graph):
        wave = [n for n in sorted(deps_graph) if n not in visited and in_deg[n] == 0]
        if not wave:
            break
        waves.append(wave)
        for n in wave:
            visited.add(n)
            for other in deps_graph:
                if n in deps_graph[other]:
                    in_deg[other] -= 1
    return waves

waves = topo_sort(deps)
print("   执行波次（同一波内可并行）：")
for i, wave in enumerate(waves):
    print(f"   第 {i+1} 波: {wave}")

# ── 3. 模拟工具执行 ──
print("\n【并行执行引擎】")

user_question = "ORD-1001 服务点A到服务点B标准服务没票了，我想知道能不能候补申请，同时帮我查有没有别的车"
print(f"   用户问题: {user_question}")

# 模拟工具函数
def run_service_point_lookup(name):
    return {"service_point_code": "SP-A", "pinyin": "beijingnan", "city": "北京"}

def run_train_query(order_id, date):
    return {"order_id": order_id, "date": date, "stops": ["服务点A", "济南西", "南京南", "服务点B"],
            "seats": {"标准服务": 0, "高级服务": 5, "商务座": 3}}

def run_policy_search(keyword, top_k):
    return {"keyword": keyword, "rules": [
        "候补申请不能保证成功，需以官方页面通知为准。",
        "每人最多可提交 4 个候补申请需求，不额外收费。",
        "候补申请订单在服务开始前 2 小时自动停止兑现。"
    ], "source": "示例业务系统 业务规则库 v3.2"}

def run_alternative_finder(from_code, to_code, date):
    return {"from": from_code, "to": to_code, "date": date,
            "items": [{"no": "订单B", "标准服务": 12, "高级服务": 8},
                       {"no": "ORD-1003", "标准服务": 5, "高级服务": 2},
                       {"no": "ORD-2001", "标准服务": 20, "高级服务": 15}]}

def run_risk_checker(draft, context):
    issues = []
    if "保证" in draft or "一定" in draft:
        issues.append("含承诺性词汇")
    if "官方" not in draft:
        issues.append("缺少官方引导")
    return {"risk_level": "HIGH" if issues else "LOW", "issues": issues}

tool_impls = {
    "service_point_lookup": lambda: run_service_point_lookup("服务点A"),
    "train_query": lambda: run_train_query("ORD-1001", "2026-06-19"),
    "policy_search": lambda: run_policy_search("候补申请", 3),
    "alternative_finder": lambda: run_alternative_finder("SP-A", "SHH", "2026-06-19"),
    "risk_checker": lambda: run_risk_checker("draft", {}),
}

# 执行
results = {}
total_time = 0
for wave_idx, wave in enumerate(waves):
    wave_start = time.time()
    print(f"\n   ⚡ 第 {wave_idx+1} 波（{'并行' if len(wave)>1 else '串行'}）:")
    for tool_name in wave:
        t_start = time.time()
        results[tool_name] = tool_impls[tool_name]()
        t_elapsed = (time.time() - t_start) * 1000
        print(f"   ├─ {tool_name}: {t_elapsed:.0f}ms → {json.dumps(results[tool_name], ensure_ascii=False)[:100]}...")
    wave_elapsed = (time.time() - wave_start) * 1000
    total_time += wave_elapsed
    print(f"   第 {wave_idx+1} 波耗时: {wave_elapsed:.0f}ms")

print(f"\n   总耗时: {total_time:.0f}ms （全串行预估: {sum(tools[t]['cost_ms'] for t in tools)}ms）")
speedup = sum(tools[t]['cost_ms'] for t in tools) / max(total_time, 1)
print(f"   并行加速比: {speedup:.1f}x")

# ── 4. 结果聚合与冲突处理 ──
print("\n【结果聚合】")
# 整合所有工具结果
aggregated = {
    "服务编号信息": {
        "ORD-1001": f"标准服务已无库存，高级服务余 {results['train_query']['seats']['高级服务']} 张",
    },
    "候补申请规则": results['policy_search']['rules'][:2],
    "替代方案": [f"{t['no']}(标准服务{t['标准服务']}张)" for t in results['alternative_finder']['items']],
}

print(json.dumps(aggregated, ensure_ascii=False, indent=2))

# 冲突检测：如果候补申请规则说不能保证 vs 替代方案有库存
print("\n【冲突检测】")
conflicts = []
if results['train_query']['seats']['标准服务'] == 0:
    conflicts.append("当前服务编号无票，需要候补申请或替代方案")
if len(results['alternative_finder']['items']) > 0 and results['train_query']['seats']['标准服务'] == 0:
    conflicts.append("有替代服务编号可用，建议优先推荐替代而非仅提候补申请")
for c in conflicts:
    print(f"   ⚠️ {c}")

# ── 5. 生成最终回答 ──
print("\n【最终回答生成】")
answer = (
    f"ORD-1001 标准服务当前已无库存（高级服务还有 {results['train_query']['seats']['高级服务']} 张）。"
    f"如果想等标准服务，可以提交候补申请——但候补申请不能保证成功，需以 官方页面通知为准。"
    f"同时为您查到同方向替代服务编号：{'、'.join([t['no'] for t in results['alternative_finder']['items'][:3]])}，"
    f"其中 ORD-2001 标准服务最充裕（20 张）。建议您登录 示例业务系统 查看实时库存后再做决定。"
)
print(f"   {answer}")

# 风险检测
risk = run_risk_checker(answer, aggregated)
print(f"\n   风险评估: {risk['risk_level']} — {', '.join(risk['issues']) if risk['issues'] else '无问题'}")

# ── 6. 工具调用统计 ──
print(f"\n{'=' * 72}")
print("工具协同总结:")
print(f"   调用工具数: {len(tools)} 个")
print(f"   执行波次: {len(waves)} 波")
print(f"   有依赖的工具: {sum(1 for t in tools.values() if t.get('depends_on'))} 个")
print(f"   无依赖可并行的: {sum(1 for t in tools.values() if not t.get('depends_on'))} 个")
print(f"   冲突项: {len(conflicts)} 条")
print("   关键原则: 分析依赖 → 拓扑排序 → 无依赖并行 → 聚合结果 → 冲突检测")