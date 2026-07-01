"""05_工具测试集 — 单元测试、集成测试、Mock 响应。

学习目标：为工具编写可复用的测试用例，保证工具行为稳定可靠。
"""

import json
from typing import Callable, List, Dict, Any

print("05_工具测试集")
print("=" * 72)
print("工具也要写测试！模型调用工具 ≠ 工具就能正确执行。")
print()

# ========== 被测试的工具 ==========
def query_order_statuss(args: dict) -> dict:
    if not args.get("from_location") or not args.get("to_location"):
        raise ValueError("from_location 和 to_location 为必填参数")
    if args.get("date") and args["date"] < "2026-01-01":
        raise ValueError("只能查询 2026 年及以后的日期")
    return {"items": [{"no": "ORD-1001", "remaining": 5}, {"no": "ORD-1002", "remaining": 0}]}

def calc_refund_fee(args: dict) -> dict:
    if args["order_amount"] <= 0:
        raise ValueError("价格必须大于 0")
    hours = args["hours_before_departure"]
    if hours > 360: rate = 0.0
    elif hours >= 48: rate = 0.05
    elif hours >= 24: rate = 0.10
    else: rate = 0.20
    return {"fee": round(args["order_amount"] * rate, 2), "rate": rate}

# ========== 简易测试框架 ==========
class TestCase:
    def __init__(self, name, fn, args, expect, should_raise=None):
        self.name = name
        self.fn = fn
        self.args = args
        self.expect = expect
        self.should_raise = should_raise

def run_tests(tests: List[TestCase]) -> dict:
    """运行测试并返回报告。"""
    results = {"total": len(tests), "passed": 0, "failed": 0, "details": []}
    for tc in tests:
        try:
            if tc.should_raise:
                try:
                    tc.fn(tc.args)
                    results["failed"] += 1
                    results["details"].append(
                        {"name": tc.name, "status": "FAIL",
                         "msg": f"期望抛出 {tc.should_raise.__name__}，但没有"})
                except tc.should_raise:
                    results["passed"] += 1
                    results["details"].append({"name": tc.name, "status": "PASS"})
                except Exception as e:
                    results["failed"] += 1
                    results["details"].append(
                        {"name": tc.name, "status": "FAIL",
                         "msg": f"期望 {tc.should_raise.__name__}，实际抛出 {type(e).__name__}"})
            else:
                actual = tc.fn(tc.args)
                if actual == tc.expect:
                    results["passed"] += 1
                    results["details"].append({"name": tc.name, "status": "PASS"})
                else:
                    results["failed"] += 1
                    results["details"].append(
                        {"name": tc.name, "status": "FAIL",
                         "msg": f"期望 {tc.expect}，实际 {actual}"})
        except Exception as e:
            results["failed"] += 1
            results["details"].append(
                {"name": tc.name, "status": "ERROR", "msg": str(e)})
    return results

# ========== 单元测试用例 ==========
unit_tests = [
    # query_order_statuss 正常
    TestCase("查库存-正常", query_order_statuss,
             {"from_location": "服务点A", "to_location": "服务点B"},
             {"items": [{"no": "ORD-1001", "remaining": 5}, {"no": "ORD-1002", "remaining": 0}]}),
    # query_order_statuss 缺少参数
    TestCase("查库存-缺from", query_order_statuss,
             {"to_location": "服务点B"}, None, should_raise=ValueError),
    # query_order_statuss 日期太早
    TestCase("查库存-日期过早", query_order_statuss,
             {"from_location": "服务点A", "to_location": "服务点B", "date": "2025-12-01"},
             None, should_raise=ValueError),
    # calc_refund_fee 正常
    TestCase("退款-48h以上", calc_refund_fee,
             {"order_amount": 553, "hours_before_departure": 72},
             {"fee": 27.65, "rate": 0.05}),
    TestCase("退款-24h内", calc_refund_fee,
             {"order_amount": 553, "hours_before_departure": 12},
             {"fee": 110.6, "rate": 0.20}),
    # calc_refund_fee 异常
    TestCase("退款-负价格", calc_refund_fee,
             {"order_amount": -10, "hours_before_departure": 12},
             None, should_raise=ValueError),
]

# ========== 集成测试（组合调用） ==========
print("【单元测试报告】")
report = run_tests(unit_tests)
print(f"  {report['passed']}/{report['total']} 通过, {report['failed']} 失败")
for d in report["details"]:
    status_icon = "✓" if d["status"] == "PASS" else "✗"
    detail = f"  {status_icon} {d['name']}"
    if "msg" in d:
        detail += f" — {d['msg']}"
    print(detail)

# ========== 集成测试：端到端调用链 ==========
print("\n【集成测试：端到端调用链】")
def integration_test():
    """模拟一条完整的查询 → 退款计算链路"""
    orders = query_order_statuss({"from_location": "服务点A", "to_location": "服务点B"})
    assert len(orders["items"]) == 2, "应返回 2 趟车"
    assert orders["items"][0]["remaining"] >= 0, "库存数不能为负"

    refund = calc_refund_fee({"order_amount": 553, "hours_before_departure": 12})
    assert refund["fee"] > 0, "退款费应大于 0"
    assert refund["rate"] == 0.20, "24h 内费率应为 20%"
    return True

try:
    integration_test()
    print("  ✓ 集成测试通过：查库存 → 退款计算链路正常")
except AssertionError as e:
    print(f"  ✗ 集成测试失败：{e}")

print("\n【Mock 模式演示（不依赖真实 API）】")
# 在测试中用 Mock 替换真实工具函数，避免网络依赖
class MockTool:
    """Mock 工具：不调真实 API，返回预定义数据。"""
    @staticmethod
    def query_order_statuss(args):
        return {"items": [{"no": "ORD-1001", "remaining": 5}]}

    @staticmethod
    def calc_refund_fee(args):
        return {"fee": 27.65, "rate": 0.05}

# 注入 Mock
original_query = query_order_statuss
query_order_statuss = MockTool.query_order_statuss
mock_result = query_order_statuss({"from_location": "服务点A"})
print(f"  Mock 结果: {json.dumps(mock_result, ensure_ascii=False)}")
query_order_statuss = original_query  # 恢复

print("\n测试策略总结：")
print("1. 单元测试：每个工具独立测试，覆盖正常 + 异常路径")
print("2. 集成测试：端到端组合调用，验证数据流转")
print("3. Mock 测试：不依赖外部 API，保证测试稳定可重复")