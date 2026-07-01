"""06 调用执行与审计日志。

筛选工具只是前半段。
真正调用工具时，还要记录：
- 为什么选这个工具
- 注入了哪些 schema
- 模型生成了哪些 arguments
- 工具执行结果是什么
- 是否需要人工确认

本脚本用模拟 tool call 展示审计日志结构。
"""

from __future__ import annotations

import json

from tool_catalog_common import inject_full_schema, rule_filter, simulated_llm_rerank, vector_recall

print("06 调用执行与审计日志")
print("=" * 80)

query = "在代码仓库里搜索 DeepSeek tool_calls 相关代码，并看看哪里解析 arguments。"
candidates, groups, rule_reasons = rule_filter(query)
recalled = vector_recall(query, candidates, top_k=12)
reranked = simulated_llm_rerank(query, recalled, top_n=5)
injected = inject_full_schema(reranked)

selected_tool = reranked[0]["tool"]
model_tool_call = {
    "name": selected_tool["name"],
    "arguments": {
        "query": "DeepSeek tool_calls arguments",
        "path": "rag-tool-agent-course",
        "file_glob": "*.py",
    },
}

tool_result = {
    "matches": [
        "code/12_专题_多轮对话机制/code/02_DeepSeek多轮对话_tool_calls_thinking.py",
        "code/04_工具调用与智能体/code/lesson_01_RAG加工具调用/03_DeepSeek真实工具调用.py",
    ],
    "summary": "找到 DeepSeek tool_calls 和 arguments 解析相关示例。",
}

audit_log = {
    "user_query": query,
    "rule_groups": groups,
    "rule_reasons": rule_reasons,
    "vector_top_k": [
        {"name": item["tool"]["name"], "score": round(item["vector_score"], 3)}
        for item in recalled
    ],
    "rerank_top_n": [
        {"name": item["tool"]["name"], "score": item["rerank_score"], "reason": item["reason"]}
        for item in reranked
    ],
    "injected_tool_names": [schema["function"]["name"] for schema in injected],
    "model_tool_call": model_tool_call,
    "tool_result": tool_result,
    "risk_level": "low",
    "need_human_confirm": False,
}

print("审计日志：")
print(json.dumps(audit_log, ensure_ascii=False, indent=2))

print("\n结论")
print("几百个工具场景必须记录选择过程。")
print("否则工具误调用时，很难定位是规则、召回、rerank、schema 还是模型 arguments 出了问题。")
