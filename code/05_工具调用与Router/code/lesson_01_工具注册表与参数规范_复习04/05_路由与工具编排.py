"""05_路由与工具编排。

学习目标：掌握完整的路由-工具编排流水线——路由选择 → 工具执行 → 结果观测 → 重路由 → 最终回答。

脚本说明：
- 本脚本模拟一个完整的 Agent 编排循环，包含路由决策、工具调用、结果评估和兜底重路由。
- 数据全部为模拟数据，不连接真实 示例业务系统 或生产系统。
"""

import json
import time
import random

print("05_路由与工具编排")
print("=" * 72)
print("知识地图位置：05_工具调用与Router / lesson_01_工具注册表与参数规范")
print("演示目标：展示完整的编排流程：路由→工具→观测→重路由→回答。")
print()

# ── 一、路由目标与工具映射 ──
route_tool_map = {
    "查票": {
        "tools": ["station_lookup", "ticket_query"],
        "fallback_route": "客服转接",
    },
    "买票": {
        "tools": ["station_lookup", "ticket_query", "order_creator"],
        "fallback_route": "查票",
    },
    "退变更": {
        "tools": ["refund_calculator", "order_lookup"],
        "fallback_route": "客服转接",
    },
    "站内服务": {
        "tools": ["station_info"],
        "fallback_route": "客服转接",
    },
    "账号问题": {
        "tools": ["account_lookup"],
        "fallback_route": "客服转接",
    },
    "客服转接": {
        "tools": ["human_handoff"],
        "fallback_route": None,
    },
}

# ── 二、模拟工具实现 ──
def station_lookup(name):
    stations = {"北京南": "VNP", "上海虹桥": "AOH", "广州南": "IZQ", "深圳北": "IOQ"}
    code = stations.get(name)
    if code:
        return {"success": True, "station_code": code, "station_name": name}
    return {"success": False, "error": f"未找到服务点: {name}"}

def ticket_query(from_code, to_code, date):
    if random.random() < 0.15:  # 15%概率模拟失败
        return {"success": False, "error": "查询超时，请稍后重试"}
    return {
        "success": True,
        "trains": [
            {"train_no": "G107", "seats_left": random.randint(0, 200), "price": 553},
            {"train_no": "G109", "seats_left": random.randint(0, 200), "price": 553},
        ],
    }

def refund_calculator(train_no, price, days_before):
    if days_before > 15:
        rate = 0.05
    elif days_before > 2:
        rate = 0.2
    else:
        rate = 0.5
    fee = round(price * rate, 2)
    return {"success": True, "refund": round(price - fee, 2), "fee": fee, "rate": f"{int(rate*100)}%"}

def order_lookup(order_id):
    if not order_id or order_id == "未知":
        return {"success": False, "error": "缺少订单号"}
    return {"success": True, "order_id": order_id, "status": "已支付", "train": "G107"}

def station_info(station_name):
    return {"success": True, "facilities": ["候车室", "餐厅", "便利店", "寄存处"]}

def account_lookup(phone=""):
    return {"success": True, "status": "正常", "bound": True}

def human_handoff(reason=""):
    return {"success": True, "message": f"已转接人工客服，原因: {reason}"}

def order_creator(from_code, to_code, date, train_no=""):
    return {"success": True, "order_id": "E" + str(random.randint(100000, 999999)), "status": "待支付"}

# 工具注册表
tool_executors = {
    "station_lookup": station_lookup,
    "ticket_query": ticket_query,
    "refund_calculator": refund_calculator,
    "order_lookup": order_lookup,
    "station_info": station_info,
    "account_lookup": account_lookup,
    "human_handoff": human_handoff,
    "order_creator": order_creator,
}

# ── 三、编排引擎 ──
print("一、编排引擎启动")
print()

def orchestrate(question, max_retries=2):
    """完整的编排流程。"""
    print(f"用户问题: {question}")
    print(f"{'─'*50}")

    # Step 1: 路由决策
    route = classify_route(question)
    print(f"[步骤1] 路由决策 → [{route}]")
    print(f"         对应工具: {route_tool_map[route]['tools']}")

    # Step 2: 提取参数
    params = extract_params(question, route)
    print(f"[步骤2] 参数提取 → {json.dumps(params, ensure_ascii=False)}")

    # Step 3: 依次执行工具
    all_results = {}
    all_success = True

    for tool_name in route_tool_map[route]["tools"]:
        time.sleep(0.1)  # 模拟延迟
        executor = tool_executors.get(tool_name)
        if not executor:
            print(f"[步骤3] 执行 {tool_name} → ❌ 工具未注册")
            all_success = False
            continue

        # 准备工具参数
        tool_params = prepare_tool_params(tool_name, params, all_results)
        print(f"[步骤3] 执行 {tool_name}({json.dumps(tool_params, ensure_ascii=False)})...")

        result = executor(**tool_params)
        all_results[tool_name] = result

        if result.get("success"):
            print(f"         ✅ 成功: {json.dumps(result, ensure_ascii=False)}")
        else:
            print(f"         ❌ 失败: {result.get('error', '未知错误')}")
            all_success = False

    # Step 4: 观测结果，判断是否需要重路由
    print(f"[步骤4] 结果观测...")
    if not all_success:
        fallback = route_tool_map[route]["fallback_route"]
        if fallback and max_retries > 0:
            print(f"         检测到工具失败 → 重路由至 [{fallback}] (剩余重试: {max_retries-1})")
            time.sleep(0.2)
            return orchestrate(question, max_retries - 1)  # 实际中应该传新的路由问题
        else:
            print(f"         无可用兜底路由，返回降级回答。")

    # Step 5: 生成最终回答
    print(f"[步骤5] 生成最终回答...")
    answer = generate_answer(route, all_results, all_success)
    print(f"         回答: {answer}")

    return {
        "route": route,
        "tools_called": list(all_results.keys()),
        "success": all_success,
        "answer": answer,
    }


# ── 四、辅助函数 ──
def classify_route(question):
    """分类路由（使用关键词规则）。"""
    if any(k in question for k in ["退", "变更", "手续费", "取消"]):
        return "退变更"
    if any(k in question for k in ["买", "下单", "订票", "下单"]):
        return "买票"
    if any(k in question for k in ["站", "寄存", "餐厅", "设施", "换乘", "地铁"]):
        return "站内服务"
    if any(k in question for k in ["密码", "账号", "登录", "认证"]):
        return "账号问题"
    if any(k in question for k in ["票", "订单编号", "库存", "时刻", "还有"]):
        return "查票"
    return "客服转接"


def extract_params(question, route):
    """根据路由类型提取参数。"""
    import re
    params = {}

    # 提取订单编号
    m = re.search(r"[GDCKTZ]\d+", question)
    if m:
        params["train_no"] = m.group(0)

    # 提取服务点
    for s in ["北京南", "上海虹桥", "广州南", "深圳北"]:
        if s in question:
            if "from_station" not in params:
                params["from_station"] = s
            elif "to_station" not in params:
                params["to_station"] = s

    # 默认值
    params.setdefault("date", "2026-06-20")
    params.setdefault("ticket_price", 553)
    params.setdefault("days_before", 5)
    params.setdefault("order_id", "未知")

    return params


def prepare_tool_params(tool_name, params, previous_results):
    """为具体工具准备参数。"""
    tool_params = {}

    if tool_name == "station_lookup":
        name = params.get("from_station", "北京南")
        tool_params["name"] = name
    elif tool_name == "ticket_query":
        from_result = previous_results.get("station_lookup", {})
        tool_params["from_code"] = from_result.get("station_code",
                                                    station_lookup(params.get("from_station", "北京南")).get(
                                                        "station_code", "VNP"))
        tool_params["to_code"] = "AOH"  # 模拟
        tool_params["date"] = params.get("date", "2026-06-20")
    elif tool_name == "refund_calculator":
        tool_params["train_no"] = params.get("train_no", "G107")
        tool_params["price"] = params.get("ticket_price", 553)
        tool_params["days_before"] = params.get("days_before", 5)
    elif tool_name == "order_lookup":
        tool_params["order_id"] = params.get("order_id", "未知")
    elif tool_name == "station_info":
        tool_params["station_name"] = params.get("from_station", "北京南")
    elif tool_name == "account_lookup":
        tool_params["phone"] = params.get("phone", "")
    elif tool_name == "human_handoff":
        tool_params["reason"] = "工具执行失败，兜底转人工"
    elif tool_name == "order_creator":
        tool_params["from_code"] = "VNP"
        tool_params["to_code"] = "AOH"
        tool_params["date"] = params.get("date", "2026-06-20")
        tool_params["train_no"] = params.get("train_no", "")

    return tool_params


def generate_answer(route, results, success):
    """根据路由类型和工具结果生成回答。"""
    if route == "查票":
        tq = results.get("ticket_query", {})
        if tq.get("success"):
            trains = tq.get("trains", [])
            train_str = "、".join(
                f"{t['train_no']}(余{t['seats_left']}张/¥{t['price']})" for t in trains)
            return f"查询到以下订单编号: {train_str}。建议尽早下单。"
        return "抱歉，库存查询暂时不可用，请稍后重试或转人工客服。"

    if route == "退变更":
        rc = results.get("refund_calculator", {})
        ol = results.get("order_lookup", {})
        if rc.get("success") and ol.get("success"):
            return (f"订单{ol['order_id']}({ol['train']})退款手续费¥{rc['fee']}"
                    f"(费率{rc['rate']})，可退¥{rc['refund']}。")
        return "请提供正确的订单号以查询退变更详情。"

    if route == "买票":
        oc = results.get("order_creator", {})
        if oc.get("success"):
            return f"已为您创建订单 {oc['order_id']}，状态: {oc['status']}。请在15分钟内完成支付。"
        return "下单失败，请检查信息后重试。"

    if route == "站内服务":
        si = results.get("station_info", {})
        if si.get("success"):
            return f"该站提供以下设施: {'、'.join(si['facilities'])}。"
        return "暂无该站信息。"

    if route == "账号问题":
        al = results.get("account_lookup", {})
        return "您的账号状态正常。" if al.get("success") else "请通过官方App进行账号验证。"

    return "正在为您转接人工客服，请稍候..."


# ── 五、批量测试 ──
test_questions = [
    "帮我查一下北京南到上海的 G107 还有票吗",
    "我要退 G107 的票，已经买了553块",
    "帮我买一张明天广州南到深圳北的业务服务票",
]

print()
for i, q in enumerate(test_questions, 1):
    print(f"\n{'='*60}")
    print(f"=== 测试 [{i}] ===")
    result = orchestrate(q)

print("\n课堂可修改点：")
print("1. 修改 classify_route 规则，观察路由改变对工具链的影响。")
print("2. 在 ticket_query 中故意让失败概率提高，观察重路由行为。")
print("3. 新增一个路由目标和对应工具，走通完整编排流程。")