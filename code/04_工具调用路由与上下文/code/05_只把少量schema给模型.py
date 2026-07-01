"""05 只把少量 schema 给模型。

工具很多时，不要把完整参数定义全塞进上下文。
先筛出少量候选，再给这些候选的完整 schema。
"""

print("05 只把少量 schema 给模型")
print("=" * 60)

full_schemas = {
    "search_doc": {
        "desc": "检索文档",
        "params": {"query": "检索问题", "top_k": "返回条数"},
    },
    "summarize_text": {
        "desc": "生成摘要",
        "params": {"text": "原文", "max_words": "最多多少字"},
    },
    "make_slide_outline": {
        "desc": "生成演示大纲",
        "params": {"topic": "主题", "pages": "页数"},
    },
    "read_file": {
        "desc": "读取文件",
        "params": {"path": "文件路径"},
    },
    "make_todo": {
        "desc": "生成待办",
        "params": {"task": "任务内容", "owner": "负责人"},
    },
}

# 上一步路由已经选出来的候选工具。
selected = ["search_doc", "summarize_text"]

print("候选工具：", selected)
print("\n准备注入给模型的 schema：")

schema_for_model = {}
for name in selected:
    schema_for_model[name] = full_schemas[name]
    print(f"\n{name}")
    print("说明：", full_schemas[name]["desc"])
    print("参数：")
    for key, desc in full_schemas[name]["params"].items():
        print(f"  - {key}: {desc}")

print("\n没有注入的工具：")
for name in full_schemas:
    if name not in selected:
        print(" -", name)

print("\n可以改哪里：")
print("1. 改 selected，观察注入内容变化。")
print("2. 给 full_schemas 增加工具。")
print("3. 把 selected 的数量限制成最多 3 个。")
