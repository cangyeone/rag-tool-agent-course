"""07 短记忆和摘要。

多轮对话不能无限增长。
一个简单办法是：
1. 最近几轮原样保留。
2. 更早的内容压成几条事实。
"""

print("07 短记忆和摘要")
print("=" * 60)

messages = [
    ("user", "我想做一个个人知识库。"),
    ("assistant", "可以先准备 Markdown 或 PDF 文档。"),
    ("user", "向量模型用 BGE-m3。"),
    ("assistant", "好的，索引阶段使用 BGE-m3。"),
    ("user", "检索时要同时用关键词。"),
    ("assistant", "可以做混合检索，再用 RRF 合并排序。"),
    ("user", "最后做一个网页问答入口。"),
]

important_memory = []
recent_messages = []

for role, text in messages:
    if "BGE-m3" in text:
        important_memory.append("向量模型：BGE-m3")
    if "关键词" in text or "混合检索" in text:
        important_memory.append("检索方式：关键词 + 向量混合检索")
    if "网页" in text:
        important_memory.append("交付形式：网页问答入口")

recent_messages = messages[-3:]

# 去重但保持顺序。
clean_memory = []
for item in important_memory:
    if item not in clean_memory:
        clean_memory.append(item)

print("\n保留下来的记忆：")
for item in clean_memory:
    print(" -", item)

print("\n最近几轮对话：")
for role, text in recent_messages:
    print(f"{role}: {text}")

print("\n下一次给模型的上下文可以这样拼：")
for item in clean_memory:
    print("[记忆]", item)
for role, text in recent_messages:
    print(f"[{role}] {text}")

print("\n可以改哪里：")
print("1. 改 messages。")
print("2. 改记忆提取规则。")
print("3. 把最近 3 轮改成最近 5 轮。")
