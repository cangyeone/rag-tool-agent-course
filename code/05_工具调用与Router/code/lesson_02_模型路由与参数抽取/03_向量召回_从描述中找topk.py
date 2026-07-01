"""03 向量召回：从工具描述中找 top-k。

规则过滤之后，候选工具仍然可能有几十个。
向量召回只看轻量描述，不看 full_schema。

课堂里用词袋余弦相似度模拟 embedding 检索。
真实系统可以替换成 bge-m3、OpenAI embedding、DashVector、FAISS 等。
"""

from __future__ import annotations

from tool_catalog_common import rule_filter, vector_recall

print("03 向量召回：从描述中找 top-k")
print("=" * 80)

query = "帮我找一下张三上周发来的邮件，看看附件里有没有培训课程材料。"
candidates, groups, reasons = rule_filter(query)
recalled = vector_recall(query, candidates, top_k=8)

print("用户问题：", query)
print("规则命中工具组：", groups)
print(f"规则候选数量：{len(candidates)}")

print("\n向量召回 top-8：")
for rank, item in enumerate(recalled, start=1):
    tool = item["tool"]
    print(f"{rank:02d}. {tool['name']:<24} group={tool['group']:<10} score={item['vector_score']:.3f}")
    print("    ", tool["description"])

print("\n结论")
print("向量召回解决的是表达变化问题，例如“找邮件附件”和 gmail_search_email 是语义相关。")
print("这一步仍然只处理轻量 description，不注入完整 schema。")
