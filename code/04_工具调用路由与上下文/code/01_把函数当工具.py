"""01 把函数当工具。

这段代码不接模型，先把“工具调用”这件事拆开看：
1. 工具就是普通函数。
2. 工具有名字、说明和参数。
3. 程序根据名字找到函数，再传入参数。
"""

print("01 把函数当工具")
print("=" * 60)

# 一份很小的资料。真实项目里，这里可能来自数据库、知识库或业务接口。
docs = [
    {"title": "RAG", "text": "RAG 会先检索资料，再把资料交给模型回答。"},
    {"title": "Embedding", "text": "Embedding 会把文本变成向量，方便做相似度计算。"},
    {"title": "Tool", "text": "工具调用让模型可以使用外部函数，而不只是生成文字。"},
]


def search_doc(keyword):
    """从 docs 里找包含关键词的资料。"""
    hits = []
    for item in docs:
        if keyword.lower() in item["title"].lower() or keyword.lower() in item["text"].lower():
            hits.append(item)
    return hits


def make_short_note(text):
    """把一段文本裁成一句短说明。"""
    return text[:40] + ("..." if len(text) > 40 else "")


# 工具表：名字 -> 函数。
# 初学时可以先不用复杂 class，字典就够看清结构。
tools = {
    "search_doc": search_doc,
    "make_short_note": make_short_note,
}

print("\n可用工具：")
for name in tools:
    print(" -", name)

print("\n调用 search_doc：")
result = tools["search_doc"]("Embedding")
print(result)

print("\n调用 make_short_note：")
note = tools["make_short_note"]("工具调用的重点不是函数有多复杂，而是输入、输出和边界要清楚。")
print(note)

print("\n可以改哪里：")
print("1. 改 docs，换成自己的资料。")
print("2. 改 keyword，观察 search_doc 返回什么。")
print("3. 新增一个普通函数，再放进 tools 字典。")
