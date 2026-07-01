"""07 几百个工具：端到端调用链路。

把本节课所有步骤串起来：
1. 规则过滤
2. 向量召回
3. LLM rerank
4. schema 注入
5. 模型生成 tool call
6. 工具执行
7. 审计记录
"""

from __future__ import annotations

import json

from tool_catalog_common import (
    inject_full_schema,
    rule_filter,
    simulated_llm_rerank,
    tool_registry,
    vector_recall,
)

print("07 几百个工具：端到端调用链路")
print("=" * 80)


def simulate_model_tool_call(user_query: str, injected_schemas: list[dict]) -> dict:
    """模拟模型在少量 schema 中选择工具并生成参数。"""
    tool_names = [schema["function"]["name"] for schema in injected_schemas]

    if "邮件" in user_query and "gmail_search_email" in tool_names:
        return {
            "name": "gmail_search_email",
            "arguments": {
                "query": "培训课程材料",
                "from_user": "张三",
                "date_range": "last_7_days",
                "has_attachment": True,
            },
        }

    if "演示材料" in user_query and "presentation_create_doc" in tool_names:
        return {
            "name": "presentation_create_doc",
            "arguments": {
                "title": "RAG 工具调用培训",
                "outline": ["背景", "工具调用", "几百个工具如何筛选", "案例"],
                "page_count": 20,
                "style": "企业内训",
            },
        }

    if "代码仓库" in user_query and "repo_search_code" in tool_names:
        return {
            "name": "repo_search_code",
            "arguments": {
                "query": "DeepSeek tool_calls arguments",
                "path": "rag-tool-agent-course",
                "file_glob": "*.py",
            },
        }

    if "知识库" in user_query and "rag_search_knowledge" in tool_names:
        return {
            "name": "rag_search_knowledge",
            "arguments": {
                "query": "候补申请规则",
                "top_k": 5,
                "source_filter": "课程资料",
            },
        }

    return {
        "name": injected_schemas[0]["function"]["name"],
        "arguments": {"input": user_query},
    }


def execute_tool(tool_call: dict) -> dict:
    """模拟工具执行。真实系统这里会调用 Gmail、Slides、代码搜索或知识库。"""
    name = tool_call["name"]
    args = tool_call["arguments"]

    if name == "gmail_search_email":
        return {"status": "ok", "matches": 3, "summary": "找到 3 封张三上周发来的带附件邮件。", "arguments": args}

    if name == "presentation_create_doc":
        return {"status": "ok", "deck_id": "demo_deck_001", "page_count": args.get("page_count"), "arguments": args}

    if name == "repo_search_code":
        return {"status": "ok", "matches": 2, "summary": "找到 2 处 tool_calls arguments 解析代码。", "arguments": args}

    if name == "rag_search_knowledge":
        return {"status": "ok", "chunks": 5, "summary": "召回 5 段候补申请规则资料。", "arguments": args}

    return {"status": "ok", "summary": f"模拟执行 {name}", "arguments": args}


def run(user_query: str) -> None:
    print("\n" + "=" * 80)
    print("用户问题：", user_query)
    print("=" * 80)

    candidates, groups, rule_reasons = rule_filter(user_query)
    recalled = vector_recall(user_query, candidates, top_k=12)
    reranked = simulated_llm_rerank(user_query, recalled, top_n=5)
    injected = inject_full_schema(reranked)
    tool_call = simulate_model_tool_call(user_query, injected)
    result = execute_tool(tool_call)

    print("\n1. 规则过滤")
    print("命中工具组：", groups if groups else "未命中")
    print("候选数量：", len(candidates), "/", len(tool_registry))

    print("\n2. 向量召回 top-12")
    for item in recalled[:5]:
        print(f"  {item['tool']['name']:<24} score={item['vector_score']:.3f}")
    print("  ...")

    print("\n3. LLM rerank top-5")
    for item in reranked:
        print(f"  {item['tool']['name']:<24} score={item['rerank_score']} reason={item['reason']}")

    print("\n4. schema 注入")
    print([schema["function"]["name"] for schema in injected])

    print("\n5. 模型生成 tool call")
    print(json.dumps(tool_call, ensure_ascii=False, indent=2))

    print("\n6. 工具执行结果")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n7. 调用链路压缩")
    print(f"{len(tool_registry)} 个工具 → {len(candidates)} 个规则候选 → {len(recalled)} 个向量候选 → {len(reranked)} 个 rerank 候选 → 1 个工具调用")


queries = [
    "帮我找一下张三上周发来的邮件，看看附件里有没有培训课程材料。",
    "帮我做一个 20 页 演示材料，主题是 RAG 工具调用培训。",
    "在代码仓库里搜索 DeepSeek tool_calls 相关代码，并看看哪里解析 arguments。",
    "帮我查一下知识库里关于候补申请的规则。",
]

for query in queries:
    run(query)

print("\n课堂结论")
print("这一节讲的不是单个 tool call，而是几百个工具时如何把选择范围压缩到少数几个。")
print("最终给模型的不是全部工具，而是经过规则、向量、rerank 后留下的少量完整 schema。")
