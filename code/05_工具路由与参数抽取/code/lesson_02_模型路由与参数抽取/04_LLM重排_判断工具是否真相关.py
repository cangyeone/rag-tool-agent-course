"""04 LLM rerank：判断工具是否真的相关。

向量召回只是“像不像”，不一定“该不该用”。

比如用户说“找邮件里的培训课程材料”，向量检索可能召回 演示材料 创建工具。
但真实意图是查邮件附件，不是创建 演示材料。

LLM rerank 用来判断工具是否真的适合当前任务。
本脚本用透明规则模拟 rerank，真实系统可换成小模型或主模型。
"""

from __future__ import annotations

from tool_catalog_common import rule_filter, simulated_llm_rerank, vector_recall

print("04 LLM rerank：判断工具是否真的相关")
print("=" * 80)

query = "帮我找一下张三上周发来的邮件，看看附件里有没有培训课程材料。"
candidates, groups, _ = rule_filter(query)
recalled = vector_recall(query, candidates, top_k=10)
reranked = simulated_llm_rerank(query, recalled, top_n=5)

print("用户问题：", query)
print("规则命中工具组：", groups)

print("\n向量召回结果：")
for item in recalled:
    print(f"- {item['tool']['name']:<24} vector={item['vector_score']:.3f}")

print("\nrerank 后保留：")
for rank, item in enumerate(reranked, start=1):
    tool = item["tool"]
    print(f"{rank}. {tool['name']:<24} score={item['rerank_score']}")
    print("   理由：", item["reason"])

print("\n结论")
print("LLM rerank 不是再做一次检索，而是判断“这个工具能不能完成当前任务”。")
print("rerank 阶段仍然不给完整 schema，只给短描述和 short_schema。")
