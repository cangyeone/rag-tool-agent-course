"""03 先用规则缩小范围。

如果有很多工具，不要一上来就全部交给模型。
可以先用简单规则把范围缩小。
"""

print("03 先用规则缩小范围")
print("=" * 60)

tool_groups = {
    "docs": ["search_doc", "summarize_doc", "compare_docs"],
    "slides": ["make_slide_outline", "polish_slide_text"],
    "files": ["list_files", "read_file", "write_file"],
    "tasks": ["make_todo", "assign_owner", "check_status"],
}

rules = {
    "docs": ["文档", "资料", "知识库", "检索", "查询"],
    "slides": ["PPT", "幻灯片", "演示", "页面"],
    "files": ["文件", "目录", "代码", "脚本"],
    "tasks": ["待办", "任务", "负责人", "进度"],
}

questions = [
    "帮我查一下 RAG 资料里 overlap 是什么",
    "把这段内容改成三页 PPT 大纲",
    "列出当前目录里的 Python 文件",
    "给这件事生成一个待办并指定负责人",
    "这个问题我也说不清楚",
]

for question in questions:
    matched_group = None
    for group, words in rules.items():
        for word in words:
            if word.lower() in question.lower():
                matched_group = group
                break
        if matched_group:
            break

    print("\n问题：", question)
    if matched_group:
        print("命中工具组：", matched_group)
        print("候选工具：", tool_groups[matched_group])
    else:
        print("没有命中规则：先让用户补充，或交给通用问答。")

print("\n可以改哪里：")
print("1. 给 rules 增加自己的关键词。")
print("2. 给 tool_groups 增加更多工具。")
print("3. 加一条新问题，看会路由到哪里。")
