"""05_RAG_加工具端到端 — 检索 + 工具调用 + 生成回答的完整流水线。

学习目标：跑通一条真实链路：用户问 → RAG搜规则 → 工具查实时 → 模型生成回答。
"""

import json
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("05_RAG_加工具端到端")
print("=" * 72)
print("端到端流水线：RAG 检索规则 + 工具查实时数据 + LLM 组织回答")
print()

# ========== 第1层：知识库（规则/政策类，不会频繁变化） ==========
knowledge_base = [
    {"title": "候补申请规则", "content": "候补申请不能保证成功，兑现取决于退款、变更和新增库存，以官方页面为准。"},
    {"title": "退款与变更规则", "content": "服务开始前15天以上免手续费，48h-15天收5%，24h-48h收10%，24h内收20%。"},
    {"title": "儿童票规则", "content": "1.2m以下免费，1.2-1.5m半价，1.5m以上全价。"},
]
print("【Layer 1: 知识库（规则文本）】")
for doc in knowledge_base:
    print(f"  · {doc['title']}：{doc['content'][:30]}...")

# ========== 第2层：工具集（实时/计算类，每次都可能变） ==========
tools = {
    "query_order_statuss": lambda args: {"items": [
        {"no": "ORD-1001", "seats": {"标准服务": 0, "高级服务": 3}, "price": 553},
        {"no": "ORD-1002", "seats": {"标准服务": 12, "高级服务": 0}, "price": 553},
    ]},
    "query_service_point": lambda args: {"code": "SP-A" if "服务点A" in args.get("name", "")
                                   else "SP-B" if "服务点B" in args.get("name", "")
                                   else "UNKNOWN"},
    "calc_refund_fee": lambda args: {"fee": args.get("price", 0) * 0.05
                                 if args.get("hours_before", 48) >= 48 else args.get("price", 0) * 0.10},
}

# ========== 端到端流水线 ==========
questions = [
    "我买了明天的 ORD-1001 标准服务，现在退款要扣多少钱？候补申请能保证买到新的吗？",
    "服务点A到服务点B的 ORD-1002 还有高级服务吗？",
]

for q in questions:
    print(f"\n{'=' * 60}")
    print(f"【用户问题】{q}")

    # Step A: RAG 检索 —— 从知识库找相关规则
    query_words = set(q.replace("？", "").replace("，", "").split())
    retrieved = []
    for doc in knowledge_base:
        score = sum(1 for w in query_words if w in doc["content"])
        if score > 0:
            retrieved.append(doc)
    retrieved.sort(key=lambda d: sum(1 for w in query_words if w in d["content"]), reverse=True)
    rag_context = retrieved[0]["content"] if retrieved else "未找到相关规则"
    print(f"  [RAG检索] 命中 → {retrieved[0]['title'] if retrieved else '无'}")

    # Step B: 工具调用 —— 决定调用哪个工具
    need_orders = any(w in q for w in ["票", "ORD-1001", "ORD-1002", "标准服务", "高级服务"])
    need_refund = any(w in q for w in ["退款", "扣", "手续费", "退"])
    need_service_point = any(w in q for w in ["服务点A", "服务点B"])

    tool_results = {}
    if need_service_point and need_orders:
        service_point = tools["query_service_point"]({"name": "服务点A"})
        tool_results["服务点"] = service_point
        orders = tools["query_order_statuss"]({})
        tool_results["库存"] = orders
        print(f"  [工具调用] 查询服务点编码 + 库存")
        print(f"    服务点编码：{service_point['code']}")
        print(f"    库存：{json.dumps(orders['items'], ensure_ascii=False)}")

    if need_refund:
        refund = tools["calc_refund_fee"]({"price": 553, "hours_before": 24})
        tool_results["退款费"] = refund
        print(f"  [工具调用] 计算退款费用 → {refund['fee']} 元")

    # Step C: 生成回答 —— 组合 RAG 依据 + 工具结果
    if need_refund and need_orders:
        answer = (f"根据《退款与变更规则》，24 小时内退款费用为价格的 20%，"
                  f"即 553×20% = {refund['fee']} 元。"
                  f"根据《候补申请规则》，候补申请不能保证成功。"
                  f"当前 ORD-1001 标准服务已无库存，高级服务还剩 3 张；"
                  f"ORD-1002 高级服务无库存，标准服务还有 12 张。"
                  f"最终以 示例业务系统 实时页面为准。")
    elif need_orders:
        has_first = any(t["seats"].get("高级服务", 0) > 0 for t in orders["items"] if t["no"] == "ORD-1002")
        answer = (f"ORD-1002 高级服务{'有库存' if has_first else '已无库存'}。"
                  f"当前可买服务编号：{json.dumps(orders['items'], ensure_ascii=False)}")
    else:
        answer = f"根据规则：{rag_context}"
    print(f"  [最终回答] {answer}")

print(f"\n{'=' * 60}")
print("\n端到端流水线总结：")
print("  ① RAG 检索 → 拿到政策/规则类知识作为「事实锚点」")
print("  ② 工具调用 → 拿到实时/计算类数据填补「时效性缺口」")
print("  ③ LLM 生成 → 组合两者，用自然语言组织给用户")
print("  三者齐备，才是真正可落地的 AI 应用。")