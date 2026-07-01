"""02 工具清单和参数检查。

工具多起来以后，需要写清楚每个工具：
1. 叫什么。
2. 做什么。
3. 需要哪些参数。
4. 参数缺了怎么办。
"""

print("02 工具清单和参数检查")
print("=" * 60)

tool_list = [
    {
        "name": "search_doc",
        "desc": "按关键词查找文档片段",
        "required": ["keyword"],
    },
    {
        "name": "create_note",
        "desc": "把输入内容整理成一条简短备注",
        "required": ["text"],
    },
    {
        "name": "make_todo",
        "desc": "根据一句话生成待办项",
        "required": ["task", "owner"],
    },
]


def check_args(tool, args):
    """检查一个工具所需参数是否都在 args 里。"""
    missing = []
    for field in tool["required"]:
        if field not in args or args[field] in ("", None):
            missing.append(field)
    return missing


print("\n工具清单：")
for tool in tool_list:
    print(f"- {tool['name']}: {tool['desc']}，参数 {tool['required']}")

calls = [
    {"tool": "search_doc", "args": {"keyword": "RAG"}},
    {"tool": "create_note", "args": {}},
    {"tool": "make_todo", "args": {"task": "整理课程资料"}},
]

print("\n检查三个调用：")
for call in calls:
    tool = next(item for item in tool_list if item["name"] == call["tool"])
    missing = check_args(tool, call["args"])
    if missing:
        print(f"- {call['tool']} 缺少参数：{missing}")
    else:
        print(f"- {call['tool']} 参数完整：{call['args']}")

print("\n可以改哪里：")
print("1. 给 tool_list 增加新工具。")
print("2. 给某个工具增加 required 字段。")
print("3. 在 calls 里故意漏参数，看检查结果。")
