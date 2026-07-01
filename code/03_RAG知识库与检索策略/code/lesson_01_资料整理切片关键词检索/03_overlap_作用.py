"""03_overlap_作用。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
学习目标：直观理解 overlap（重叠窗口）如何防止关键信息被切断，
         通过 side-by-side 对比不同 overlap 值的切片结果。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

学习目标：直观理解 overlap（重叠窗口）如何防止关键信息被切断。

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


full_text = " ".join(doc["content"] for doc in docs)

print("=" * 72)
print("03_overlap 作用 —— Side-by-Side 切片对比")
print("=" * 72)

print(f"\n原始文本（{len(full_text)}字符）：")
print(textwrap.fill(full_text, width=60))

# ── 重叠切片函数 ──
def chunk_with_overlap(text, chunk_size, overlap):
    """带重叠的滑动窗口切片。"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return chunks

# ═══════════════════════════════════════════
# 并排对比：overlap = 0, 8, 16 (chunk_size=30)
# ═══════════════════════════════════════════
chunk_size = 30
overlap_values = [0, 8, 16]

print("\n" + "━" * 72)
print(f"并排对比：chunk_size={chunk_size}，不同 overlap 值")
print("━" * 72)

# 计算所有切片
all_chunks = {}
for ov in overlap_values:
    all_chunks[ov] = chunk_with_overlap(full_text, chunk_size, ov)

# 确定最大行数
max_lines = max(len(v) for v in all_chunks.values())

# 打印表头
header = f"  {'序号':<5}"
for ov in overlap_values:
    header += f"overlap={ov:<3} {'':<{chunk_size}}"
print(header)
print("  " + "-" * 60)

# 打印每一行
for row in range(max_lines):
    line = f"  {row+1:<5}"
    for ov in overlap_values:
        ch = all_chunks[ov][row] if row < len(all_chunks[ov]) else ""
        if ch:
            line += f"{ch:<{chunk_size+4}}"
        else:
            line += " " * (chunk_size + 4)
    print(line)

# ═══════════════════════════════════════════
# 关键信息切断分析
# ═══════════════════════════════════════════
print("\n" + "━" * 72)
print('关键信息「切断」分析：overlap=0 的问题')
print("━" * 72)

# 演示一句话被切断的情况
demo_text = "候补申请不能保证成功，兑现结果取决于退款、变更和新增库存。"
print(f"\n  原始句子：{demo_text}")
print(f"\n  overlap=0 的切片（chunk_size=12）：")
chunks_no = chunk_with_overlap(demo_text, 12, 0)
for i, ch in enumerate(chunks_no, 1):
    print(f"    [{i}] {ch}")

print(f"\n  overlap=6 的切片（chunk_size=12）：")
chunks_yes = chunk_with_overlap(demo_text, 12, 6)
for i, ch in enumerate(chunks_yes, 1):
    print(f"    [{i}] {ch}")

# ═══════════════════════════════════════════
# Overlap 与 chunk 数量关系
# ═══════════════════════════════════════════
print("\n" + "━" * 72)
print("Overlap 与 Chunk 数量关系")
print("━" * 72)

print(f"\n  {'Overlap':<10}{'Chunk数':<10}{'总字符（含重叠）':<20}{'重叠比例'}")
for ov in [0, 4, 8, 12, 16, 20]:
    chunks = chunk_with_overlap(full_text, chunk_size=25, overlap=ov)
    raw_len = len(full_text)
    chunked_len = sum(len(c) for c in chunks)
    ratio = (chunked_len - raw_len) / raw_len * 100 if raw_len > 0 else 0
    print(f"  {ov:<10}{len(chunks):<10}{chunked_len:<20}{ratio:.1f}%")

# ═══════════════════════════════════════════
# 搜索命中对比
# ═══════════════════════════════════════════
print("\n" + "━" * 72)
print('搜索命中对比：关键词「退变更」能否被完整命中')
print("━" * 72)

query = "退变更"
print(f"\n  搜索词：'{query}'")
for ov in overlap_values:
    chunks = chunk_with_overlap(full_text, chunk_size=chunk_size, overlap=ov)
    hits = [(i, ch) for i, ch in enumerate(chunks, 1) if query in ch]
    print(f"\n  overlap={ov}（{len(chunks)}个chunk）：")
    if hits:
        for idx, ch in hits:
            pos = ch.index(query)
            start = max(0, pos - 5)
            end = min(len(ch), pos + len(query) + 5)
            context = ch[start:end]
            print(f'    chunk[{idx}] \"...{context}...\" ')
    else:
        print(f"    未命中！关键词可能被切断了。")

print("\n" + "=" * 72)
print("学习要点：")
print("  1. overlap=0 容易切断关键信息，导致检索漏召回")
print("  2. overlap 越大，信息完整性越好，但存储成本增加")
print("  3. 典型 overlap 取 chunk_size 的 10%~25%")
print("  4. 需要在召回率与存储效率之间找到平衡")