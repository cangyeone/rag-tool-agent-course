"""02_路由策略设计。

学习目标：掌握三种路由策略——关键词匹配、正则模式、意图分类，理解各自的适用场景。

脚本说明：
- 本脚本对比三种路由策略在 示例业务系统 场景下的表现。
- 数据全部为模拟数据，不连接真实 示例业务系统 或生产系统。
"""

import json
import re

print("02_路由策略设计")
print("=" * 72)
print("知识地图位置：05_工具路由与参数抽取 / lesson_01_工具注册表与参数规范")
print("演示目标：对比三种路由策略，理解规则路由的设计思路。")
print()

# ── 一、定义目标路由目标（"工具"或"处理节点"） ──
routes = {
    "查库存": {"desc": "查询服务编号库存、价格、时刻表"},
    "买票": {"desc": "提交创建订单订单"},
    "退款与变更": {"desc": "退款、变更规则与操作"},
    "站内服务": {"desc": "服务网点设施、餐饮、换乘指引"},
    "账号问题": {"desc": "注册、登录、实名认证"},
    "客服转接": {"desc": "转人工客服"},
}

print("一、路由目标（6个处理节点）")
print(json.dumps(routes, ensure_ascii=False, indent=2))

# ── 二、测试问题集 ──
test_questions = [
    "明天北京到上海的 G 字头还有库存吗",
    "我要买一张后天从服务点C到服务点D的业务服务票",
    "已经买了的票怎么退，要收手续费吗",
    "北京西站有没有麦当劳",
    "我忘记密码了怎么找回",
    "ORD-1001 没票了能不能候补申请",
    "我要变更到下周一的同一服务编号",
    "火服务网点能不能寄存行李",
]
print("\n二、测试问题集")
for i, q in enumerate(test_questions, 1):
    print(f"  [{i}] {q}")

# ── 三、策略一：关键词匹配路由 ──
print("\n三、策略一：关键词匹配路由")

keyword_rules = {
    "查库存": ["票", "库存", "有没有", "还有", "时刻", "服务编号"],
    "买票": ["买", "创建订单", "订票", "创建订单", "购买"],
    "退款与变更": ["退", "变更", "改", "手续费", "候补申请"],
    "站内服务": ["站", "服务网点", "寄存", "餐厅", "麦当劳", "便利店", "换乘"],
    "账号问题": ["密码", "注册", "登录", "账号", "实名", "认证"],
    "客服转接": ["人工", "客服", "投诉"],
}


def keyword_route(question):
    scores = {}
    for route_name, keywords in keyword_rules.items():
        score = sum(1 for kw in keywords if kw in question)
        if score > 0:
            scores[route_name] = score
    if not scores:
        return "客服转接", 0
    best = max(scores, key=scores.get)
    return best, scores[best]


for q in test_questions:
    route, score = keyword_route(q)
    print(f"  '{q}' → [{route}] (匹配得分:{score})")

# ── 四、策略二：正则模式路由 ──
print("\n四、策略二：正则模式路由")

regex_rules = {
    "查库存": [
        (r"(还有|有没有).{0,4}票", 3),
        (r"(G|D|C|K|T|Z)\d+", 2),
        (r"(查询|查一下|帮我查).{0,6}(票|服务编号)", 3),
    ],
    "买票": [
        (r"(买|购|订).{0,4}票", 3),
        (r"我要.{0,4}张", 2),
    ],
    "退款与变更": [
        (r"(退|变更|改到|候补申请)", 3),
        (r"手续费", 2),
    ],
    "站内服务": [
        (r"(服务网点|火服务网点|北京\w{1,2}站|广州\w{1,2}站)", 2),
        (r"(寄存|餐厅|麦当劳|便利店|换乘|地铁)", 3),
    ],
    "账号问题": [
        (r"(密码|账号|登录|注册|实名|认证)", 3),
        (r"忘记", 2),
    ],
    "客服转接": [
        (r"(人工|客服|投诉|举报)", 3),
    ],
}


def regex_route(question):
    scores = {}
    for route_name, patterns in regex_rules.items():
        score = 0
        for pattern, weight in patterns:
            if re.search(pattern, question):
                score += weight
        if score > 0:
            scores[route_name] = score
    if not scores:
        return "客服转接", 0
    best = max(scores, key=scores.get)
    return best, scores[best]


for q in test_questions:
    route, score = regex_route(q)
    print(f"  '{q}' → [{route}] (匹配得分:{score})")

# ── 五、策略三：意图分类路由（模拟简单 NLP 分类器） ──
print("\n五、策略三：意图分类路由（模拟分类器）")

intent_patterns = {
    "查库存": ["票", "服务编号", "时刻", "库存", "价格", "班次"],
    "买票": ["买", "购", "订", "创建订单"],
    "退款与变更": ["退", "改", "候补申请", "取消"],
    "站内服务": ["站", "寄存", "餐厅", "设施", "换乘", "地铁"],
    "账号问题": ["密码", "账号", "登录", "认证", "验证"],
    "客服转接": ["人工", "客服", "投诉"],
}


def intent_classify(question):
    """模拟意图分类：先提取特征词，再计算 TF 相似度。"""
    q_words = set(question)
    scores = {}
    for intent, keywords in intent_patterns.items():
        overlap = len(q_words & set(keywords))
        total = len(keywords)
        scores[intent] = overlap / total if total > 0 else 0
    if max(scores.values()) == 0:
        return "客服转接", 0.0
    best = max(scores, key=scores.get)
    return best, scores[best]


for q in test_questions:
    route, confidence = intent_classify(q)
    print(f"  '{q}' → [{route}] (置信度:{confidence:.2f})")

# ── 六、策略对比 ──
print("\n六、三种策略对比")
print("  策略            优点                    缺点")
print("  关键词匹配      实现简单，速度快         同义词、否定句容易误判")
print("  正则模式        精确度高，可处理变体     规则维护成本高")
print("  意图分类        泛化能力强              需要训练数据，冷启动困难")

print("\n课堂可修改点：")
print("1. 新增一条测试问题，观察三种策略的结果是否一致。")
print("2. 修改关键词/正则规则的权重，看路由结果如何变化。")
print("3. 讨论：如果三种策略结果不一致，应该信哪个？")