"""02 规则过滤：先定位工具组。

规则过滤负责抓明显信号：
- 用户说“邮件” → Gmail tool group
- 用户说“演示材料” → presentation tool group
- 用户说“代码仓库” → code / filesystem tools
- 用户说“知识库” → RAG tools

这一步速度快、成本低，不需要大模型。
"""

from __future__ import annotations

from tool_catalog_common import rule_filter, tool_registry

print("02 规则过滤：先定位工具组")
print("=" * 80)

queries = [
    "帮我找一下张三上周发来的邮件，看看附件里有没有培训课程材料。",
    "帮我做一个 20 页 演示材料，主题是 RAG 工具调用培训。",
    "在代码仓库里搜索 DeepSeek tool_calls 相关代码。",
    "查一下知识库里关于候补申请的规则。",
    "帮我整理一下明天上午的事情。",
]

for query in queries:
    candidates, groups, reasons = rule_filter(query)
    print("\n用户问题：", query)
    print("命中工具组：", groups if groups else "未命中")
    print("规则理由：")
    for reason in reasons:
        print("  -", reason)
    print(f"候选工具数：{len(candidates)} / {len(tool_registry)}")
    print("候选前 5 个：", [tool["name"] for tool in candidates[:5]])

print("\n结论")
print("规则过滤不是最终选择工具，只是先把搜索范围缩小。")
print("没命中规则时要有兜底，不能直接报错。")
