"""08 一个小型任务执行器。

这个脚本把前面几件事串起来：
1. 读用户问题。
2. 用规则选工具。
3. 抽取参数。
4. 执行工具。
5. 记录过程。
"""

print("08 一个小型任务执行器")
print("=" * 60)

docs = [
    "RAG 会先检索资料，再让模型基于资料回答。",
    "工具调用适合处理查询、计算、读写文件这类明确动作。",
    "上下文管理会保留最近对话和关键记忆。",
]


def search_doc(query, top_k=2):
    hits = []
    for text in docs:
        score = sum(1 for word in query.split() if word in text)
        if score > 0 or query in text:
            hits.append((score, text))
    hits.sort(reverse=True)
    return [text for score, text in hits[:top_k]] or docs[:top_k]


def make_note(text):
    return "备注：" + text[:50]


tools = {
    "search_doc": search_doc,
    "make_note": make_note,
}

question = "帮我检索 RAG，并返回 2 条资料"
log = []

print("用户问题：", question)

# 一、路由
if "检索" in question or "查" in question:
    tool_name = "search_doc"
else:
    tool_name = "make_note"
log.append(f"选择工具：{tool_name}")

# 二、抽取参数
if tool_name == "search_doc":
    query = question.replace("帮我", "").replace("检索", "").split("并")[0].strip()
    top_k = 2
    args = {"query": query, "top_k": top_k}
else:
    args = {"text": question}
log.append(f"参数：{args}")

# 三、执行
result = tools[tool_name](**args)
log.append(f"工具返回：{result}")

# 四、生成一个简单回答
if tool_name == "search_doc":
    answer = "找到这些资料：\n" + "\n".join(f"- {item}" for item in result)
else:
    answer = result
log.append("生成回答")

print("\n执行日志：")
for item in log:
    print(" -", item)

print("\n最终回答：")
print(answer)

print("\n可以改哪里：")
print("1. 改 question。")
print("2. 给 tools 增加新函数。")
print("3. 给日志增加耗时、工具名、参数和结果。")
