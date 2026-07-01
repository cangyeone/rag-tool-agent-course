"""04_RAG_加工具回答。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
学习目标：掌握将 RAG 检索结果与工具调用结果组合成完整回答的方法，
         包括模板化组装、引用来源标注、信息优先级排序。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import math
import jieba
import time
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

# ── 从课程目录加载真实文档 ──
def _load_real_docs():
    real_docs = []
    sample_files = [
        ("code/01_大模型基础/README.md", 350),
        ("code/02_模型接口与指令设计/README.md", 350),
        ("code/03_RAG知识库与检索/README.md", 350),
        ("README.md", 350),
    ]
    for rel_path, max_chars in sample_files:
        file_path = COURSE_ROOT / rel_path
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")[:max_chars]
            title = rel_path.split("/")[1] if rel_path.startswith("code/") else "课程总览"
            real_docs.append({"title": title, "source": rel_path, "content": content})
    return real_docs

docs = _load_real_docs()


question = "帮我查一下今天的ORD-1001次服务流程还有库存吗？如果不能买的话能不能候补申请？"

print("=" * 72)
print("04_RAG_加工具回答 —— 模板化回答组装")
print("=" * 72)

print(f"\n用户问题：{question}")

# ═══════════════════════════════════════════
# 一、执行 RAG 检索
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("一、RAG 检索结果")
print("━" * 60)

# 使用 BM25 检索相关文档
query_tokens = [w for w in jieba.cut(question) if w.strip()]
all_contents = [d["content"] for d in docs]
N = len(docs)
avgdl = sum(len(c) for c in all_contents) / N

def idf(term, docs_list):
    df = sum(1 for d in docs_list if term in d)
    return math.log((N + 1) / (df + 1)) + 1

def bm25(query_tokens, doc_text, docs_list):
    score = 0
    doc_len = len(doc_text)
    k1, b = 1.5, 0.75
    for term in query_tokens:
        tf = doc_text.count(term)
        if tf == 0:
            continue
        score += idf(term, docs_list) * (tf * (k1 + 1)) / (
            tf + k1 * (1 - b + b * (doc_len / avgdl)))
    return score

rag_scored = [(bm25(query_tokens, d["content"], all_contents), d) for d in docs]
rag_scored.sort(key=lambda x: x[0], reverse=True)

print(f"  检索到 {len([s for s, _ in rag_scored if s > 0])} 篇相关文档：")
for score, doc in rag_scored:
    if score > 0:
        print(f"    ✓ {doc['title']}（BM25={score:.4f}）")
        print(f"      内容：{doc['content']}")

# ═══════════════════════════════════════════
# 二、执行工具调用（模拟）
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("二、工具调用结果")
print("━" * 60)

def query_inventory(order_id, date="2025-06-19"):
    """模拟库存查询。"""
    return {
        "tool": "query_inventory",
        "params": {"order_id": order_id, "date": date},
        "result": {
            "order_id": order_id,
            "date": date,
            "from": "服务点A",
            "to": "服务点B",
            "seats": {"商务座": 3, "高级服务": 12, "标准服务": 0},
            "status": "标准服务已无库存",
        },
        "query_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

def query_service_point_info(service_point_name):
    """模拟服务点信息查询。"""
    return {
        "tool": "query_service_point_info",
        "params": {"service_point_name": service_point_name},
        "result": {
            "service_point_name": service_point_name,
            "telegram_code": "BJP",
            "lines": ["示例线路业务服务", "京津城际"],
        },
    }

# 执行工具调用
tool_results = []
tool_results.append(query_inventory("ORD-1001"))
tool_results.append(query_service_point_info("服务点A"))

for tr in tool_results:
    print(f"  工具：{tr['tool']}")
    print(f"  参数：{json.dumps(tr['params'], ensure_ascii=False)}")
    print(f"  结果：{json.dumps(tr['result'], ensure_ascii=False)}")
    print()

# ═══════════════════════════════════════════
# 三、模板化回答组装
# ═══════════════════════════════════════════
print("━" * 60)
print("三、模板化回答组装")
print("━" * 60)

# 提取关键信息
top_rag = rag_scored[0][1] if rag_scored and rag_scored[0][0] > 0 else None
order_info = tool_results[0]["result"] if tool_results else None
service_point_info = tool_results[1]["result"] if len(tool_results) > 1 else None

# ── 组装策略：工具信息优先（实时数据），RAG 信息做补充 ──

# 策略 1：直接拼接模板
print("\n  策略一：直接拼接模板")
answer_direct = f"""
您好！关于您的问题，以下是查询结果：

📌 实时库存查询（{tool_results[0]['query_time']}）：
ORD-1001 次服务流程 {order_info['from']} → {order_info['to']}，
目前商务座剩余 3 张，高级服务剩余 12 张，标准服务已无库存。

📋 相关规定（来源：《{top_rag['title']}》）：
{top_rag['content']}

💡 建议：
1. 您可以考虑购买高级服务或商务座
2. 提交候补申请（标准服务），有人退款时会自动兑现
3. 最终以 官方页面显示为准
"""
print(answer_direct.strip())

# 策略 2：结构化回答（JSON 格式）
print("\n  策略二：结构化回答（JSON 格式，便于下游处理）")

structured_answer = {
    "question": question,
    "answer_type": "混合回答（RAG + 工具）",
    "real_time_data": {
        "source": "工具调用",
        "tool": tool_results[0]["tool"],
        "data": order_info,
    },
    "knowledge_base": {
        "source": "RAG检索",
        "documents": [
            {
                "title": top_rag["title"],
                "content": top_rag["content"],
                "relevance_score": round(rag_scored[0][0], 4),
            }
        ] if top_rag else [],
    },
    "final_response": answer_direct.strip().replace("\n", " "),
    "citations": [top_rag["title"]] if top_rag else [],
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}

print(json.dumps(structured_answer, ensure_ascii=False, indent=2))

# ═══════════════════════════════════════════
# 四、信息优先级与冲突处理
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("四、信息优先级与冲突处理")
print("━" * 60)

print("""
  信息源优先级（从高到低）：
    1. 工具调用结果（实时、确定性强）
       → 库存数据、订单状态、支付结果

    2. RAG 检索结果（知识解释、规则说明）
       → 规章条款、FAQ、操作流程

    3. LLM 自身知识（兜底，需标注不确定性）
       → "根据我的理解…建议您以示例业务系统官方为准"

  冲突处理策略：
    当 RAG 结果与工具结果矛盾时：
    - 工具结果优先（实时数据 > 静态知识）
    - 明确标注差异：如"规章规定X，但当前系统显示Y"
    - 建议用户以官方为准
""")

# ═══════════════════════════════════════════
# 五、不同场景的回答模板
# ═══════════════════════════════════════════
print("━" * 60)
print("五、不同场景的回答模板对比")
print("━" * 60)

scenarios = [
    ("仅有 RAG（无工具）", "查询规章类问题",
     "根据《{doc_title}》，{doc_content}。如有疑问请咨询客服。"),
    ("仅有工具（无 RAG）", "查询实时状态",
     "查询结果：{tool_data}。数据更新时间 {time}。"),
    ("RAG + 工具", "复杂问题（既有规则又需实时数据）",
     "查询到 {tool_data}。根据《{doc_title}》，{doc_content}。综合建议：..."),
]

for label, usage, template in scenarios:
    print(f"\n  {label}：")
    print(f"    适用：{usage}")
    print(f"    模板：{template}")

print("\n" + "=" * 72)
print("学习要点：")
print("  1. RAG + 工具 = 静态知识 + 动态数据 → 完整回答")
print("  2. 模板化组装保证回答格式统一、引用可追溯")
print("  3. 信息优先级：工具 > RAG > LLM 自身知识")
print("  4. 结构化 JSON 输出便于上游系统消费")