"""01_为什么需要Router。

学习目标：理解为什么单一模型不够用，需要通过 Router 按任务类型和成本进行分流。

脚本说明：
- 本脚本展示多模型场景下，无 Router vs 有 Router 的成本和效果差异。
- 数据全部为模拟数据，不连接真实 示例业务系统 或生产系统。
"""

import json

print("01_为什么需要Router")
print("=" * 72)
print("知识地图位置：05_工具路由与参数抽取 / lesson_01_工具注册表与参数规范")
print("演示目标：对比无路由和有路由两种方案，理解路由的核心价值。")
print()

# ── 一、模型注册表：不同模型有不同成本和能力 ──
models = {
    "cheap_fast": {
        "name": "轻量模型（本地部署）",
        "cost_per_1k": 0.0001,
        "latency_ms": 50,
        "capability": ["简单问答", "关键词匹配", "规则判断"],
        "max_tokens": 512,
    },
    "standard": {
        "name": "标准模型（云端API）",
        "cost_per_1k": 0.002,
        "latency_ms": 300,
        "capability": ["通用问答", "信息抽取", "意图识别"],
        "max_tokens": 4096,
    },
    "premium": {
        "name": "高级模型（旗舰API）",
        "cost_per_1k": 0.01,
        "latency_ms": 1200,
        "capability": ["复杂推理", "多步规划", "代码生成"],
        "max_tokens": 8192,
    },
    "private": {
        "name": "私有化模型（内网部署）",
        "cost_per_1k": 0.0,
        "latency_ms": 200,
        "capability": ["敏感数据处理", "内部知识问答", "权限校验"],
        "max_tokens": 2048,
    },
}

print("一、模型注册表（四个模型，不同成本与能力）")
for mid, m in models.items():
    print(f"  [{mid}] {m['name']}: 成本=${m['cost_per_1k']}/1k tokens, "
          f"延迟={m['latency_ms']}ms, 能力={m['capability']}")

# ── 二、模拟一批 示例业务系统 用户问题 ──
questions = [
    {"id": 1, "text": "示例服务点在哪个区？", "type": "simple_fact", "sensitive": False},
    {"id": 2, "text": "ORD-1001 次服务流程今天还有库存吗？请帮我查一下库存和候补申请策略，并比较同方向其他服务编号的价格。",
     "type": "complex_query", "sensitive": False},
    {"id": 3, "text": "我的订单号 E123456789 的退款进度如何？", "type": "sensitive_order", "sensitive": True},
    {"id": 4, "text": "儿童票怎么买？", "type": "simple_fact", "sensitive": False},
    {"id": 5, "text": "帮我规划从广州到哈尔滨的中转路线，要求总时间不超过12小时，避开夜间到达。",
     "type": "complex_planning", "sensitive": False},
]

print("\n二、模拟用户问题（共 {} 个）".format(len(questions)))
for q in questions:
    print(f"  [{q['id']}] ({q['type']}) {q['text'][:40]}...")

# ── 三、方案A：无 Router，全部用 premium 模型 ──
print("\n三、方案A：无路由 —— 所有问题都用高级模型")
total_cost_a = 0
total_latency_a = 0
for q in questions:
    tokens = len(q["text"])  # 简化：按字符数估算
    cost = tokens * models["premium"]["cost_per_1k"] / 1000
    latency = models["premium"]["latency_ms"]
    total_cost_a += cost
    total_latency_a += latency
    print(f"  [{q['id']}] → premium, tokens≈{tokens}, 成本≈${cost:.6f}, 延迟≈{latency}ms")
print(f"  总成本: ${total_cost_a:.6f}  |  总延迟: {total_latency_a}ms")

# ── 四、方案B：有 Router，根据问题类型分流 ──
print("\n四、方案B：有路由 —— 按问题类型选择最合适的模型")


def route_by_type(question_type, is_sensitive):
    """路由规则：根据问题类型和敏感性选择模型。"""
    if is_sensitive:
        return "private"
    if question_type in ("simple_fact",):
        return "cheap_fast"
    if question_type in ("complex_query",):
        return "standard"
    if question_type in ("complex_planning",):
        return "premium"
    return "standard"


total_cost_b = 0
total_latency_b = 0
for q in questions:
    model_id = route_by_type(q["type"], q["sensitive"])
    model = models[model_id]
    tokens = len(q["text"])
    cost = tokens * model["cost_per_1k"] / 1000
    latency = model["latency_ms"]
    total_cost_b += cost
    total_latency_b += latency
    print(f"  [{q['id']}] → {model_id}, tokens≈{tokens}, 成本≈${cost:.6f}, 延迟≈{latency}ms")
print(f"  总成本: ${total_cost_b:.6f}  |  总延迟: {total_latency_b}ms")

# ── 五、对比总结 ──
print("\n五、对比总结")
print(f"  成本节省: ${total_cost_a - total_cost_b:.6f} ({(1 - total_cost_b/total_cost_a)*100:.1f}%)")
print(f"  延迟降低: {total_latency_a - total_latency_b}ms ({(1 - total_latency_b/total_latency_a)*100:.1f}%)")
print(f"  敏感数据保护: 方案A ❌（敏感订单送到云端） | 方案B ✅（敏感数据走私有化模型）")

print("\n课堂可修改点：")
print("1. 修改路由规则 route_by_type，观察成本和延迟变化。")
print("2. 增加新的问题类型，看看路由规则是否需要扩展。")
print("3. 讨论：如果路由判断错误，会带来什么后果？")