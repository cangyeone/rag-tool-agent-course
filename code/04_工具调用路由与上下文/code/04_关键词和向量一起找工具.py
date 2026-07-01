"""04 关键词和向量一起找工具。

这里不用真实向量模型，先用一个很小的词集合相似度代替。
重点是看清思路：关键词能抓精确词，语义相似能抓相近表达。
"""

print("04 关键词和向量一起找工具")
print("=" * 60)

tools = [
    {"name": "search_doc", "desc": "检索文档、知识库、资料片段"},
    {"name": "summarize_text", "desc": "把长文本压成短摘要"},
    {"name": "make_slide_outline", "desc": "把材料整理成演示大纲"},
    {"name": "read_file", "desc": "读取本地文件内容"},
    {"name": "make_todo", "desc": "生成待办事项和负责人"},
]

query = "我有一段很长的资料，想压短一点"


def tokenize(text):
    """非常简陋的分词：按常见词做包含判断。"""
    vocab = ["检索", "文档", "知识库", "资料", "摘要", "压短", "大纲", "演示", "文件", "待办", "负责人"]
    return {word for word in vocab if word in text}


def keyword_score(query, desc):
    """字面命中几个词，就给几分。"""
    score = 0
    for word in tokenize(query):
        if word in desc:
            score += 1
    return score


def set_similarity(query, desc):
    """用词集合重叠比例模拟语义相似度。"""
    a = tokenize(query)
    b = tokenize(desc)
    if not a or not b:
        return 0
    return len(a & b) / len(a | b)


rows = []
for tool in tools:
    k_score = keyword_score(query, tool["desc"])
    v_score = set_similarity(query, tool["desc"])
    final = k_score + v_score
    rows.append((final, k_score, v_score, tool["name"], tool["desc"]))

rows.sort(reverse=True)

print("问题：", query)
print("\n候选工具排序：")
for final, k_score, v_score, name, desc in rows:
    print(f"- {name:<20} 总分={final:.2f} 关键词={k_score} 相似度={v_score:.2f} 说明={desc}")

print("\n可以改哪里：")
print("1. 改 query，观察排序变化。")
print("2. 改 tools 里的 desc。")
print("3. 把 set_similarity 换成真实 Embedding 相似度。")
