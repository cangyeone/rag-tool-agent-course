"""04_工具返回结果处理 — 解析、校验、回传、异常兜底。

学习目标：正确处理工具返回结果，在各种异常情况下不掉链子。
"""

import json

print("04_工具返回结果处理")
print("=" * 72)
print("工具调用不只是「调一下」——返回结果必须解析、校验、再喂给模型。")
print()

def query_order_statuss(args: dict) -> dict:
    """真实风格的库存查询函数。"""
    if not args.get("from_location") or not args.get("to_location"):
        raise ValueError("from_location 和 to_location 为必填参数")
    return {"items": [{"no": "ORD-1001", "left": 5}, {"no": "ORD-1002", "left": 0}], "source": "示例业务系统实时查询"}

def calc_refund_fee(args: dict) -> dict:
    """真实风格的退款费计算函数。"""
    if args["order_amount"] <= 0:
        raise ValueError("价格必须大于 0")
    hours = args["hours_before_departure"]
    if hours >= 48: rate = 0.05
    elif hours >= 24: rate = 0.10
    else: rate = 0.20
    return {"fee": round(args["order_amount"] * rate, 2), "rate": rate, "based_on": "中国行业退款规定"}

def broken_tool(args: dict) -> dict:
    """模拟网络异常的工具。"""
    raise RuntimeError("网络超时，示例业务系统 接口无响应")

def empty_query(args: dict) -> dict:
    """空结果查询。"""
    return {"items": []}

# ========== 场景1：正常返回，解析后回传 ==========
print("【场景1：正常返回 → 解析 → 追加到 messages】")
result = query_order_statuss({"from_location": "服务点A", "to_location": "服务点B"})
tool_msg = {
    "role": "tool",
    "tool_call_id": "call_normal_001",
    "name": "query_order_statuss",
    "content": json.dumps(result, ensure_ascii=False)
}
print(f"  工具返回：{json.dumps(result, ensure_ascii=False)}")
print(f"  回传格式：{json.dumps(tool_msg, ensure_ascii=False, indent=2)}")

# ========== 场景2：工具抛异常，构造错误消息 ==========
print("\n【场景2：工具执行异常 → 错误消息兜底】")
try:
    result = broken_tool({})
except Exception as e:
    error_msg = {
        "role": "tool",
        "tool_call_id": "call_err_002",
        "name": "broken_tool",
        "content": json.dumps({"error": str(e), "fallback": "请稍后重试或联系人工客服"})
    }
    print(f"  异常：{e}")
    print(f"  兜底消息：{json.dumps(error_msg, ensure_ascii=False, indent=2)}")

# ========== 场景3：返回空结果 ==========
print("\n【场景3：工具返回空结果 → 告知模型「没查到」】")
result = empty_query({})
if isinstance(result, dict) and result.get("items") == []:
    print(f"  空结果处理：告知模型该区间当前无库存，建议扩大日期范围或换服务编号")

# ========== 场景4：多工具并行时汇总结果 ==========
print("\n【场景4：多工具并行 → 汇总所有结果再回传】")
parallel_calls = [
    {"id": "p1", "tool": "query_order_statuss", "args": {"from_location": "服务点A", "to_location": "服务点B"}},
    {"id": "p2", "tool": "calc_refund_fee", "args": {"order_amount": 553, "hours_before_departure": 24}},
]
responses = []
tool_map = {"query_order_statuss": query_order_statuss, "calc_refund_fee": calc_refund_fee}
for call in parallel_calls:
    try:
        fn = tool_map[call["tool"]]
        r = fn(call["args"])
        responses.append({"id": call["id"], "tool": call["tool"], "result": r, "ok": True})
    except Exception as e:
        responses.append({"id": call["id"], "tool": call["tool"], "error": str(e), "ok": False})
for resp in responses:
    status = "✓" if resp["ok"] else "✗"
    print(f"  [{status}] {resp['tool']} → {json.dumps(resp.get('result', resp.get('error')), ensure_ascii=False)}")

print("\n处理要点总结：")
print("1. 工具结果必须用 tool_call_id 关联回对应的调用请求")
print("2. 异常不崩流程，构造结构化错误消息回传给模型")
print("3. 空结果 ≠ 错误，明确告知「无数据」让模型据此回答")
print("4. 多工具并行：等全部完成再一次性回传，减少 API 调用次数")