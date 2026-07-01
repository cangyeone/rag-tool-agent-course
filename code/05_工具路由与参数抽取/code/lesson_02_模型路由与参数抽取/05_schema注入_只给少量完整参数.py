"""05 schema 注入：只给最终少量工具的完整参数。

几百个工具时，不要把所有 full_schema 都塞给模型。
完整 schema 只在最后一步注入，通常保留 3-10 个工具。
"""

from __future__ import annotations

import json

from tool_catalog_common import (
    estimate_schema_chars,
    inject_full_schema,
    rule_filter,
    simulated_llm_rerank,
    tool_registry,
    vector_recall,
)

print("05 schema 注入：只给少量完整参数")
print("=" * 80)

query = "在代码仓库里搜索 DeepSeek tool_calls 相关代码，并看看哪里解析 arguments。"
candidates, groups, _ = rule_filter(query)
recalled = vector_recall(query, candidates, top_k=12)
reranked = simulated_llm_rerank(query, recalled, top_n=5)
injected = inject_full_schema(reranked)

print("用户问题：", query)
print("规则命中工具组：", groups)
print(f"全量工具数量：{len(tool_registry)}")
print(f"规则候选数量：{len(candidates)}")
print(f"向量召回数量：{len(recalled)}")
print(f"rerank 保留数量：{len(reranked)}")
print(f"最终注入 full_schema 数量：{len(injected)}")

print("\n长度对比：")
print(f"全部工具 full_schema 字符数：{estimate_schema_chars(tool_registry, full=True)}")
print(f"最终注入 schema 字符数：{len(json.dumps(injected, ensure_ascii=False))}")

print("\n最终注入给模型的 tools 参数：")
print(json.dumps(injected, ensure_ascii=False, indent=2)[:3500])
print("...（课堂输出截断）")

print("\n结论")
print("schema 注入是最后一步。")
print("模型正式生成 tool call 时，只看最终 3-10 个完整 schema。")
