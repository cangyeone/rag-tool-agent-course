"""04_工具错误处理 — 网络超时、参数非法、工具不存在、限流、重试策略。

学习目标：识别五种典型工具错误，实现分级处理与自动重试。
"""

import json
import time
import random

print("04_工具错误处理")
print("=" * 72)
print("工具调用必然出错。关键不是不出错，而是出错后怎么优雅地兜住。")
print()

# ========== 模拟五种典型错误 ==========
call_count = {"rate_limit": 0}  # 用于模拟限流计数

def mock_tool(tool_name, args):
    """模拟工具执行，可能抛出各种错误。"""
    if tool_name == "network_timeout":
        raise TimeoutError("连接 示例业务系统 接口超时，已等待 30s")
    if tool_name == "invalid_params":
        if not isinstance(args.get("order_amount"), (int, float)) or args["order_amount"] <= 0:
            raise ValueError(f"参数 order_amount 非法: {args.get('order_amount')}")
        return {"fee": args["order_amount"] * 0.05}
    if tool_name == "not_found":
        raise KeyError(f"工具 not_found 未注册")
    if tool_name == "rate_limited":
        call_count["rate_limit"] += 1
        if call_count["rate_limit"] <= 2:
            raise RuntimeError("429 Too Many Requests，请稍后重试")
        return {"data": "第 3 次终于成功了"}
    if tool_name == "partial_failure":
        return {"items": [{"no": "ORD-1001", "left": 5}],
                "warnings": ["ORD-1002 数据获取失败，仅展示部分结果"]}
    return {"ok": True}

# ========== 错误分类与处理策略 ==========
error_handlers = {
    TimeoutError: {"type": "可重试", "retry": True, "max_retries": 2,
                   "user_msg": "查询超时，正在重试..."},
    ValueError: {"type": "不可重试", "retry": False,
                 "user_msg": "参数错误，请检查输入"},
    KeyError: {"type": "不可重试", "retry": False,
               "user_msg": "系统暂不支持该操作"},
    RuntimeError: {"type": "可重试（限流）", "retry": True, "max_retries": 3,
                   "user_msg": "请求过于频繁，稍后重试..."},
}

def execute_with_retry(tool_name, args, max_retries=2):
    """带重试和错误分类的工具执行。"""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            result = mock_tool(tool_name, args)
            # 部分成功也算一种"软错误"
            if isinstance(result, dict) and "warnings" in result:
                print(f"  ⚠ 部分成功: {result['warnings']}")
                return result
            return result
        except Exception as e:
            last_error = e
            error_type = type(e)
            handler = error_handlers.get(error_type, {"retry": False, "user_msg": "未知错误"})

            print(f"  [尝试 {attempt+1}/{max_retries+1}] {handler['user_msg']} ({error_type.__name__}: {e})")

            if not handler.get("retry"):
                break
            if attempt < max_retries:
                wait = (2 ** attempt) * 0.5  # 指数退避
                time.sleep(wait)

    # 所有重试失败后的兜底
    return {"error": str(last_error), "fallback": "请稍后重试或转人工客服"}

# ========== 测试五种错误场景 ==========
test_scenarios = [
    ("network_timeout", {}, "网络超时"),
    ("invalid_params", {"order_amount": -100}, "参数非法"),
    ("not_found", {}, "工具未注册"),
    ("rate_limited", {}, "限流"),
    ("partial_failure", {}, "部分成功"),
]

print("【错误场景测试】")
for tool_name, args, desc in test_scenarios:
    print(f"\n  场景: {desc}")
    result = execute_with_retry(tool_name, args)
    print(f"  最终结果: {json.dumps(result, ensure_ascii=False)}")

# ========== 错误处理层级 ==========
print("\n【错误处理四层架构】")
layers = [
    ("L1 重试层", "网络超时、限流等瞬时错误自动重试，指数退避"),
    ("L2 降级层", "主工具挂了用备用工具或缓存数据顶上"),
    ("L3 兜底层", "所有路径都失败时，返回'系统繁忙，请稍后重试'"),
    ("L4 监控层", "记录每次错误到日志，P95 错误率超阈值告警"),
]
for layer, desc in layers:
    print(f"  {layer}：{desc}")