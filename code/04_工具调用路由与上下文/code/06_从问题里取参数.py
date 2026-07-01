"""06 从问题里取参数。

模型选中工具以后，还要把用户问题里的字段取出来。
这个脚本先用规则和正则表达式做一个小例子。
"""

import re

print("06 从问题里取参数")
print("=" * 60)

questions = [
    "帮我检索 RAG 的 overlap，返回 3 条",
    "把主题 工具调用 整理成 5 页大纲",
    "读取文件 docs/intro.md",
]

for question in questions:
    print("\n问题：", question)

    if "检索" in question or "查询" in question:
        tool = "search_doc"
        top_k_match = re.search(r"返回\s*(\d+)\s*条", question)
        top_k = int(top_k_match.group(1)) if top_k_match else 5
        query = question.replace("帮我", "").replace("检索", "").split("返回")[0].strip()
        args = {"query": query, "top_k": top_k}

    elif "大纲" in question:
        tool = "make_slide_outline"
        pages_match = re.search(r"(\d+)\s*页", question)
        pages = int(pages_match.group(1)) if pages_match else 3
        topic_match = re.search(r"主题\s*(.*?)\s*整理", question)
        topic = topic_match.group(1).strip() if topic_match else "未命名主题"
        args = {"topic": topic, "pages": pages}

    elif "读取文件" in question:
        tool = "read_file"
        path = question.split("读取文件", 1)[1].strip()
        args = {"path": path}

    else:
        tool = "unknown"
        args = {}

    print("工具：", tool)
    print("参数：", args)

print("\n可以改哪里：")
print("1. 增加新问题。")
print("2. 改正则表达式。")
print("3. 给每个参数加默认值。")
