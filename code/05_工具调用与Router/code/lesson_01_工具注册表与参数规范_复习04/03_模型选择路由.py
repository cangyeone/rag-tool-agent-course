"""03_模型选择路由。

学习目标：根据问题复杂度和敏感性，将请求路由到不同能力和成本的模型上。

脚本说明：
- 本脚本展示模型选择路由的完整流程：评估问题 → 选择模型 → 模拟调用 → 统计成本。
- 数据全部为模拟数据，不连接真实 示例业务系统 或生产系统。
"""

import json
import time
import random

print("03_模型选择路由")
print("=" * 72)
print("知识地图位置：05_工具调用与Router / lesson_01_工具注册表与参数规范")
print("演示目标：根据问题特征路由到最合适的模型，实现成本与效果的平衡。")
print()

# ── 一、模型注册表 ──
model_registry = {
    "local_fast": {
        "name": "本地轻量模型",
        "cost": 0.0,
        "latency_range": (10, 50),
        "suitable_for": ["简单查询", "关键词提取", "规则判断"],
        "max_complexity": 3,
    },
    "cloud_standard": {
        "name": "云端标准模型",
        "cost": 0.002,
        "latency_range": (100, 400),
        "suitable_for": ["通用问答", "信息抽取", "意图识别"],
        "max_complexity": 6,
    },
    "cloud_advanced": {
        "name": "云端高级模型",
        "cost": 0.01,
        "latency_range": (500, 1500),
        "suitable_for": ["复杂推理", "多步规划", "综合分析"],
        "max_complexity": 10,
    },
    "private_secure": {
        "name": "私有化安全模型",
        "cost": 0.0,
        "latency_range": (50, 200),
        "suitable_for": ["敏感数据", "内部查询", "权限校验"],
        "max_complexity": 8,
    },
}

print("一、模型注册表")
for mid, m in model_registry.items():
    print(f"  [{mid}] {m['name']}: 成本=${m['cost']}/1k, "
          f"延迟={m['latency_range'][0]}-{m['latency_range'][1]}ms, "
          f"最大复杂度={m['max_complexity']}")

# ── 二、问题复杂度评估器 ──
print("\n二、问题复杂度评估器")


def assess_complexity(question):
    """评估问题复杂度（1-10分）。"""
    score = 1
    # 长度因子
    if len(question) > 30:
        score += 2
    elif len(question) > 15:
        score += 1
    # 多条件因子
    conditions = ["并且", "而且", "同时", "另外", "还要", "以及"]
    score += sum(1 for c in conditions if c in question)
    # 推理因子
    reasoning_keywords = ["为什么", "怎么", "如何", "原因", "分析", "规划", "比较", "区别"]
    score += sum(1 for k in reasoning_keywords if k in question)
    # 数值计算因子
    if any(c.isdigit() for c in question):
        score += 1
    return min(score, 10)


def check_sensitive(question):
    """检查是否包含敏感信息。"""
    sensitive_keywords = ["订单号", "身份证", "手机号", "密码", "支付", "退款进度", "个人信息"]
    return any(k in question for k in sensitive_keywords)


# ── 三、模型选择路由 ──
print("\n三、模型选择路由器")


def route_to_model(question):
    """根据问题复杂度和敏感性选择模型。"""
    complexity = assess_complexity(question)
    is_sensitive = check_sensitive(question)

    if is_sensitive:
        # 敏感数据必须走私有化模型
        if complexity <= model_registry["private_secure"]["max_complexity"]:
            return "private_secure", complexity, "敏感数据-私有化"
        else:
            # 过于复杂，需要在私有化环境中降级处理
            return "private_secure", complexity, "敏感数据-私有化(复杂降级)"

    # 按复杂度匹配
    if complexity <= model_registry["local_fast"]["max_complexity"]:
        return "local_fast", complexity, "低复杂度-本地快速"
    elif complexity <= model_registry["cloud_standard"]["max_complexity"]:
        return "cloud_standard", complexity, "中等复杂度-云端标准"
    else:
        return "cloud_advanced", complexity, "高复杂度-云端高级"


# ── 四、测试 ──
test_cases = [
    "北京站在哪",
    "帮我查一下 G107 的价格和时刻表，并且比较一下标准服务和高级服务的区别",
    "订单号 E123456789 的退款进度如何",
    "帮我规划从广州到哈尔滨的中转路线，要求总时间不超过12小时，并且分析每种方案的优劣",
    "我的身份证号 110101199001011234 绑定的账号忘了密码怎么办",
    "儿童票多少钱",
]

total_cost = 0
total_latency = 0

print()
for i, q in enumerate(test_cases, 1):
    model_id, complexity, reason = route_to_model(q)
    model = model_registry[model_id]
    latency = random.randint(*model["latency_range"])
    cost = len(q) * model["cost"] / 1000
    total_cost += cost
    total_latency += latency

    print(f"  [{i}] \"{q}\"")
    print(f"       复杂度={complexity}/10, 敏感={'是' if check_sensitive(q) else '否'}")
    print(f"       → [{model_id}] {model['name']} ({reason})")
    print(f"       预估成本=${cost:.6f}, 预估延迟={latency}ms")
    print()

print(f"四、汇总统计")
print(f"  总请求数: {len(test_cases)}")
print(f"  总成本: ${total_cost:.6f}")
print(f"  总延迟: {total_latency}ms")
print(f"  私有化处理: {sum(1 for q in test_cases if check_sensitive(q))} 条")

print("\n课堂可修改点：")
print("1. 修改 assess_complexity 的评分规则，观察路由变化。")
print("2. 增加新的模型，调整 max_complexity 阈值。")
print("3. 讨论：复杂度和敏感性冲突时，如何权衡？")