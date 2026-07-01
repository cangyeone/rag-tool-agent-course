"""02_文档切片。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
学习目标：理解文档切片的三种策略（固定长度、按句切、按段切），
         掌握 chunk_size 对检索效果的影响。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import textwrap
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

# ── 从课程目录加载真实文档 ──
def _load_real_docs():
    real_docs = []
    sample_files = [
        ("code/01_AI基础与模型发展/README.md", 350),
        ("code/02_大模型接口与业务指令/README.md", 350),
        ("code/03_RAG知识库与检索策略/README.md", 350),
        ("README.md", 350),
    ]
    for rel_path, max_chars in sample_files:
        file_path = COURSE_ROOT / rel_path
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")[:max_chars]
            title = rel_path.split("/")[1] if rel_path.startswith("code/") else "课程总览"
            real_docs.append({"title": title, "source": rel_path, "content": content})
    return real_docs

docs = _load_real_docs()


# 拼接为长文本
full_text = " ".join(doc["content"] for doc in docs)

print("=" * 72)
print("02_文档切片 —— 三种分块策略对比")
print("=" * 72)

print("\n完整文本（拼接后，", len(full_text), "字符）：")
print(textwrap.fill(full_text, width=60))
print()

# ═══════════════════════════════════════════
# 策略一：固定长度切片
# ═══════════════════════════════════════════
print("━" * 50)
print("策略一：固定长度切片（Fixed-Length Chunking）")
print("━" * 50)

def fixed_chunk(text, chunk_size):
    """按固定字符数切分，超出部分截断。"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

for chunk_size in [20, 35, 50]:
    print(f"\n  chunk_size={chunk_size}：")
    chunks = fixed_chunk(full_text, chunk_size)
    for i, ch in enumerate(chunks, 1):
        print(f"    [{i}] {ch}")

# ═══════════════════════════════════════════
# 策略二：按句子切片
# ═══════════════════════════════════════════
print("\n" + "━" * 50)
print("策略二：按句子切片（Sentence-Based Chunking）")
print("━" * 50)

def sentence_chunk(text, max_sentences=1):
    """按中文标点切句，可组合多句为一个 chunk。"""
    import re
    sentences = re.split(r'[。！？；]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    buffer = ""
    count = 0
    for sent in sentences:
        buffer += sent + "。"
        count += 1
        if count >= max_sentences:
            chunks.append(buffer.strip())
            buffer = ""
            count = 0
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks

print("\n  按单句切分：")
chunks = sentence_chunk(full_text, max_sentences=1)
for i, ch in enumerate(chunks, 1):
    print(f"    [{i}] {ch}")

print("\n  按两句一组切分：")
chunks = sentence_chunk(full_text, max_sentences=2)
for i, ch in enumerate(chunks, 1):
    print(f"    [{i}] {ch}")

# ═══════════════════════════════════════════
# 策略三：按文档（段落）切片
# ═══════════════════════════════════════════
print("\n" + "━" * 50)
print("策略三：按文档/段落切片（Document-Level Chunking）")
print("━" * 50)
print("\n  每个文档作为独立 chunk（保留文档边界）：")
for i, doc in enumerate(docs, 1):
    print(f"    [{i}] {doc['title']}: {doc['content']}")

# ═══════════════════════════════════════════
# 对比分析
# ═══════════════════════════════════════════
print("\n" + "━" * 50)
print("对比分析")
print("━" * 50)
comparison = [
    ("固定长度", "实现简单，速度快",
      "语义可能被截断，如【候补申请不能保】被切断"),
    ("按句子", "保留完整语义单元",
     "句子长度不一，chunk 大小不均匀"),
    ("按文档", "上下文完整，适合 FAQ",
     "文档太长时超出模型上下文窗口"),
]
print(f"\n  {'策略':<12}{'优点':<22}{'缺点'}")
for name, pro, con in comparison:
    print(f"  {name:<12}{pro:<22}{con}")

print("\n" + "=" * 72)
print("学习要点：")
print("  1. 固定长度切片 → 速度优先，适合初步实验")
print("  2. 按句子切片 → 语义优先，适合问答场景")
print("  3. 按文档切片 → 适合 FAQ 类短文档")
print("  4. 真实系统通常结合多种策略")