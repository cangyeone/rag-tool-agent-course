"""01 工具分组与轻量注册表。

本节开始不再讲“怎么写一个工具”，而是讲“几百个工具怎么调用”。

第一步是把工具注册表拆成两层：
1. 轻量 tool card：用于规则过滤、向量检索、rerank。
2. 完整 full_schema：只在最后注入少量工具时使用。
"""

from __future__ import annotations

from collections import Counter

from tool_catalog_common import estimate_schema_chars, tool_registry

print("01 工具分组与轻量注册表")
print("=" * 80)

group_counter = Counter(tool["group"] for tool in tool_registry)

print(f"\n工具总数：{len(tool_registry)}")
print("工具组分布：")
for group, count in group_counter.most_common():
    print(f"  {group:<12} {count:>3} 个")

print("\n轻量 tool card 示例：")
sample = tool_registry[0]
print({
    "name": sample["name"],
    "group": sample["group"],
    "description": sample["description"],
    "short_schema": sample["short_schema"],
})

print("\n完整 full_schema 示例：")
print(sample["full_schema"])

full_chars = estimate_schema_chars(tool_registry, full=True)
short_chars = estimate_schema_chars(tool_registry, full=False)

print("\n长度对比：")
print(f"  全部 full_schema 字符数：{full_chars}")
print(f"  全部轻量 tool card 字符数：{short_chars}")
print(f"  轻量描述约为 full_schema 的 {short_chars / full_chars:.1%}")

print("\n结论")
print("几百个工具时，前几步只用轻量 tool card。")
print("完整 schema 留到最后，只给少量候选工具。")
