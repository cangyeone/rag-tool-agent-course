"""03_工具执行器 — 串行 vs 并行调用、超时控制、执行策略。

学习目标：理解工具执行器的运营调度逻辑，知道什么时候并行、什么时候串行、超时如何兜底。
"""

import json
import time
import threading

print("03_工具执行器")
print("=" * 72)
print("执行器 = 运营调度中心：决定工具的执行顺序、超时策略和结果汇总。")
print()

# ========== 模拟工具函数（带延迟） ==========
def query_tickets(args):
    time.sleep(0.4)
    return {"trains": [{"no": "G107", "remaining": 5}]}

def query_station(args):
    time.sleep(0.2)
    return {"code": "VNP", "name": "北京南"}

def calc_refund(args):
    time.sleep(0.3)
    return {"fee": round(args["price"] * 0.05, 1)}

tools_map = {
    "query_tickets": query_tickets,
    "query_station": query_station,
    "calc_refund": calc_refund,
}

# ========== 执行策略1：串行执行 ==========
print("【策略1：串行执行 —— 适合有依赖关系】")
tool_calls_sequential = [
    {"id": "c1", "name": "query_station", "args": {"name": "北京南"}},
    {"id": "c2", "name": "query_tickets", "args": {"from": "北京南"}},
]
t0 = time.time()
results_seq = []
for call in tool_calls_sequential:
    fn = tools_map.get(call["name"])
    if fn:
        r = fn(call["args"])
        results_seq.append({"id": call["id"], "tool": call["name"], "result": r})
cost_seq = time.time() - t0
for r in results_seq:
    print(f"  [{r['id']}] {r['tool']} → {json.dumps(r['result'], ensure_ascii=False)}")
print(f"  总耗时: {cost_seq:.2f}s（两次调用串行累加）")

# ========== 执行策略2：并行执行（多线程模拟） ==========
print("\n【策略2：并行执行 —— 适合无依赖、互相独立的工具调用】")
tool_calls_parallel = [
    {"id": "p1", "name": "query_station", "args": {"name": "北京南"}},
    {"id": "p2", "name": "query_tickets", "args": {"from": "北京南"}},
    {"id": "p3", "name": "calc_refund", "args": {"price": 553}},
]
t0 = time.time()
results_par = []
threads = []
lock = threading.Lock()

def execute(call):
    try:
        fn = tools_map.get(call["name"])
        if fn:
            r = fn(call["args"])
            with lock:
                results_par.append({"id": call["id"], "tool": call["name"], "result": r})
    except Exception as e:
        with lock:
            results_par.append({"id": call["id"], "tool": call["name"], "error": str(e)})

for call in tool_calls_parallel:
    t = threading.Thread(target=execute, args=(call,))
    threads.append(t)
    t.start()
for t in threads:
    t.join()
cost_par = time.time() - t0
for r in results_par:
    print(f"  [{r['id']}] {r['tool']} → {json.dumps(r['result'], ensure_ascii=False)}")
print(f"  总耗时: {cost_par:.2f}s（三次调用只取最慢的那个）")
print(f"  加速比: {cost_seq / cost_par:.1f}x")

# ========== 执行策略3：超时控制 ==========
print("\n【策略3：超时控制 —— 慢工具不能拖垮整个回答】")

def slow_tool(args):
    time.sleep(2.0)
    return {"data": "太慢了"}

tools_map["slow_search"] = slow_tool

def execute_with_timeout(tool_name, args, timeout=0.5):
    """带超时的工具执行。"""
    result_container = {"result": None, "error": None, "done": False}

    def target():
        try:
            fn = tools_map[tool_name]
            result_container["result"] = fn(args)
        except Exception as e:
            result_container["error"] = str(e)
        finally:
            result_container["done"] = True

    t = threading.Thread(target=target)
    t.start()
    t.join(timeout)
    if not result_container["done"]:
        return {"error": f"工具 {tool_name} 执行超时（>{timeout}s），已取消", "fallback": True}
    if result_container["error"]:
        return {"error": result_container["error"]}
    return result_container["result"]

# 正常工具
print(f"  query_station: {json.dumps(execute_with_timeout('query_station', {}, timeout=1.0), ensure_ascii=False)}")
# 超时工具
print(f"  slow_search: {json.dumps(execute_with_timeout('slow_search', {}, timeout=0.3), ensure_ascii=False)}")

print("\n执行器设计要点：")
print("1. 无依赖的工具调用 → 并行执行，降延迟")
print("2. 有依赖的（如前一个工具的输出是后一个的输入）→ 必须串行")
print("3. 每个工具都要设超时，避免一个慢请求拖垮整个对话")
print("4. 超时不是静默失败，必须返回结构化错误让模型知道发生了什么")