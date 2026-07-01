"""04_工具选择路由。

学习目标：根据用户问题分析所需工具，实现工具选择路由和工具级联调用。

脚本说明：
- 本脚本展示如何从问题中分析出需要哪些工具，并按依赖关系级联执行。
- 数据全部为模拟数据，不连接真实 示例业务系统 或生产系统。
"""

import json

print("04_工具选择路由")
print("=" * 72)
print("知识地图位置：05_工具路由与参数抽取 / lesson_01_工具注册表与参数规范")
print("演示目标：从问题分析出所需工具，实现路由选择和工具级联。")
print()

# ── 一、工具注册表（含输入输出 schema 和依赖关系） ──
tool_registry = {
    "service_point_lookup": {
        "name": "服务点查询工具",
        "desc": "根据服务点名称查询服务点编码和所属城市",
        "input_params": ["service_point_name"],
        "output_fields": ["service_point_code", "city", "service_point_type"],
        "requires": [],
        "mock_result": lambda name: {
            "服务点A": {"service_point_code": "SP-A", "city": "北京", "service_point_type": "业务服务站"},
            "服务点B": {"service_point_code": "SP-B", "city": "上海", "service_point_type": "业务服务站"},
            "服务点C": {"service_point_code": "SP-C", "city": "广州", "service_point_type": "业务服务站"},
            "服务点D": {"service_point_code": "SP-D", "city": "深圳", "service_point_type": "业务服务站"},
        }.get(name, {"service_point_code": "UNK", "city": "未知", "service_point_type": "未知"}),
    },
    "inventory_query": {
        "name": "库存查询工具",
        "desc": "查询指定日期和区间的库存信息",
        "input_params": ["from_location_code", "to_location_code", "date"],
        "output_fields": ["items", "availability"],
        "requires": ["service_point_lookup"],
        "mock_result": lambda f, t, d: {
            "items": [
                {"order_id": "ORD-1001", "seats_left": 0, "price_2nd": 553},
                {"order_id": "ORD-1002", "seats_left": 23, "price_2nd": 553},
                {"order_id": "D321", "seats_left": 156, "price_2nd": 426},
            ],
            "date": d,
        },
    },
    "refund_calculator": {
        "name": "退款计算工具",
        "desc": "计算退款费用和可退金额",
        "input_params": ["order_id", "order_amount", "days_before_departure"],
        "output_fields": ["refund_amount", "fee", "fee_rate"],
        "requires": [],
        "mock_result": lambda t, p, d: {
            "refund_amount": round(p * (1 - 0.05 if d > 15 else 0.2 if d > 2 else 0.5), 2),
            "fee": round(p * (0.05 if d > 15 else 0.2 if d > 2 else 0.5), 2),
            "fee_rate": "5%" if d > 15 else "20%" if d > 2 else "50%",
        },
    },
    "alternative_route": {
        "name": "替代路线工具",
        "desc": "当直达无票时，搜索中转方案",
        "input_params": ["from_city", "to_city", "date"],
        "output_fields": ["routes", "total_time", "transfer_city"],
        "requires": [],
        "mock_result": lambda f, t, d: {
            "routes": [
                {"path": f"{f}→济南→{t}", "total_time": "6h30m", "transfer": "济南"},
                {"path": f"{f}→南京→{t}", "total_time": "7h15m", "transfer": "南京"},
            ],
        },
    },
}

print("一、工具注册表（含依赖关系）")
for tid, t in tool_registry.items():
    deps = f"依赖: {t['requires']}" if t["requires"] else "无依赖"
    print(f"  [{tid}] {t['name']}: 入参={t['input_params']}, 出参={t['output_fields']}, {deps}")

# ── 二、工具选择路由器 ──
print("\n二、问题 → 工具选择路由")


def select_tools(question):
    """根据问题内容，选择需要调用的工具列表。"""
    selected = []

    # 规则1: 提到服务点名 → 需要服务点查询
    service_point_patterns = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京"]
    if any(s in question for s in service_point_patterns) and any(
            k in question for k in ["站", "到", "从", "出发"]
    ):
        selected.append("service_point_lookup")

    # 规则2: 提到"票"、"服务编号"、"G/D/C/K" → 需要库存查询
    import re
    if any(k in question for k in ["票", "服务编号", "库存", "有没有", "还有"]) or \
            re.search(r"[GDCKTZ]\d+", question):
        selected.append("inventory_query")

    # 规则3: 提到退款/变更/手续费 → 需要退款计算
    if any(k in question for k in ["退", "变更", "手续费", "取消订单"]):
        selected.append("refund_calculator")

    # 规则4: 提到"没票"、"候补申请"、"中转"、"替代" → 需要替代路线
    if any(k in question for k in ["没票", "候补申请", "中转", "替代", "其他服务编号", "换乘"]):
        selected.append("alternative_route")

    return selected


# ── 三、工具级联执行器 ──
print("\n三、工具级联执行（按依赖顺序）")


def execute_tool_chain(question, tool_names):
    """按依赖关系排序后级联执行工具。"""
    # 拓扑排序（简单实现）
    executed = set()
    results = {}
    all_tools_needed = set(tool_names)

    # 收集所有依赖
    for t in list(all_tools_needed):
        for dep in tool_registry[t]["requires"]:
            all_tools_needed.add(dep)

    # 按依赖顺序执行
    executed_order = []
    remaining = set(all_tools_needed)
    while remaining:
        ready = {t for t in remaining
                 if all(d in executed for d in tool_registry[t]["requires"])}
        if not ready:
            break
        for t in sorted(ready):
            executed_order.append(t)
            executed.add(t)
            remaining.remove(t)

    print(f"  执行顺序: {' → '.join(executed_order)}")

    # 模拟执行
    for t in executed_order:
        tool = tool_registry[t]
        print(f"\n  执行 [{t}] {tool['name']}...")

        # 准备参数（从之前工具的结果中获取）
        params = {}
        for p in tool["input_params"]:
            if p in params:
                continue
            # 尝试从前面工具结果中获取
            for prev_tool in executed_order[:executed_order.index(t)]:
                prev_result = results.get(prev_tool, {})
                if p in prev_result:
                    params[p] = prev_result[p]
                    break

        # 降级：从问题中获取参数
        if t == "service_point_lookup" and "service_point_name" not in params:
            # 从问题中提取服务点名
            for s in ["服务点A", "服务点B", "服务点C", "服务点D"]:
                if s in question:
                    params["service_point_name"] = s
                    break
        if t == "inventory_query":
            if "from_location_code" not in params and "service_point_lookup" in results:
                params["from_location_code"] = results["service_point_lookup"].get("service_point_code", "UNK")
            if "to_location_code" not in params:
                params["to_location_code"] = "SP-B"
            if "date" not in params:
                params["date"] = "2026-06-20"
        if t == "refund_calculator":
            if "order_id" not in params:
                import re
                m = re.search(r"[GDCKTZ]\d+", question)
                params["order_id"] = m.group(0) if m else "ORD-1001"
            if "order_amount" not in params:
                params["order_amount"] = 553
            if "days_before_departure" not in params:
                params["days_before_departure"] = 5
        if t == "alternative_route":
            if "from_city" not in params and "service_point_lookup" in results:
                params["from_city"] = results["service_point_lookup"].get("city", "北京")
            if "to_city" not in params:
                params["to_city"] = "上海"
            if "date" not in params:
                params["date"] = "2026-06-20"

        print(f"    入参: {json.dumps(params, ensure_ascii=False)}")
        # 按 input_params 顺序提取值，缺失的用默认值
        input_keys = tool["input_params"]
        pos_args = [params.get(k, "默认值") for k in input_keys]
        result = tool["mock_result"](*pos_args)
        results[t] = result
        print(f"    结果: {json.dumps(result, ensure_ascii=False)}")

    return results


# ── 四、完整测试 ──
test_questions = [
    "明天服务点A到服务点B还有业务服务票吗",
    "ORD-1001 没票了怎么办，能中转吗",
    "我的 ORD-1001 票要退，要收多少手续费",
    "帮我查一下服务点C站到服务点D的库存",
]

for i, q in enumerate(test_questions, 1):
    print(f"\n{'='*60}")
    print(f"测试 [{i}]: {q}")
    tools = select_tools(q)
    print(f"  选中工具: {tools}")
    if tools:
        results = execute_tool_chain(q, tools)
        print(f"\n  最终结果汇总:")
        for tid, r in results.items():
            print(f"    [{tid}] {json.dumps(r, ensure_ascii=False)}")

print("\n课堂可修改点：")
print("1. 修改 select_tools 规则，观察工具选择的变化。")
print("2. 在 mock_result 里增加错误模拟，观察级联失败时如何处理。")
print("3. 讨论：如果工具依赖形成循环怎么办？")