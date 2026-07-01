"""01_文本向量化。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
学习目标：理解「文本 → 向量」的核心概念，掌握哈希向量
         （Hash Vector）这个最简单的向量化方法，为嵌入模型打下基础。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━‘



Q AI是什么？ 
A 这一部分用于解释 AI、传统机器学习、深度学习、大模型之间的关系，并把字符、token、messages、流式输出等基础概念落到可运行代码上。 

->

固定长度向量v 

"""

import math
import hashlib
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

# ── 从课程目录加载真实文档 ──
def _load_real_docs():
    real_docs = []
    sample_files = [
        ("code/01_大模型基础/README.md", 350),
        ("code/02_模型接口与指令设计/README.md", 350),
        ("code/03_RAG知识库与检索/README.md", 350),
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


question = "候补申请一定能成功吗？"

print("=" * 72)
print("01_文本向量化 —— 从文本到向量")
print("=" * 72)

# ═══════════════════════════════════════════
# 概念：什么是向量？
# ═══════════════════════════════════════════
print("\n一、什么是向量？")
print("  向量 = 一组有序的数字，表示文本在「语义空间」中的坐标。")
print("  例如：'候补申请规则' → [0.3, 0.7, 0.1, 0, ...]")
print("  维度 = 向量的长度。维度越高，表达能力越强，计算量也越大。")
print("  典型维度：128, 384, 768, 1024, 4096")

# ═══════════════════════════════════════════
# 方法一：字符哈希向量（Char-Hash）
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("方法一：字符哈希向量（8维）—— 最简单、可解释")
print("━" * 60)

DIM = 8

def char_hash_vector(text, dim=DIM):
    """把每个字符的 Unicode 码点映射到向量维度上累加，再归一化。"""
    vec = [0.0] * dim
    for ch in text:
        bucket = ord(ch) % dim
        vec[bucket] += 1.0
    # L2 归一化
    length = math.sqrt(sum(x * x for x in vec))
    if length == 0:
        return [0.0] * dim
    return [round(x / length, 4) for x in vec]

# 向量化所有文本
texts = [question] + [d["content"] for d in docs]
titles = ["[查询]" + question] + [d["title"] for d in docs]

print(f"\n  维度={DIM}，归一化方式=L2")
print(f"\n  {'文本':<22}{'向量'}")
print("  " + "-" * 50)
for title, text in zip(titles, texts):
    vec = char_hash_vector(text)
    print(f"  {title:<22}{vec}")

# ═══════════════════════════════════════════
# 方法二：词级哈希向量（Word-Level Hash）
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("方法二：词级哈希向量（16维）—— 用分词 + MD5 哈希")
print("━" * 60)

WORD_DIM = 16

def word_hash_vector(text, dim=WORD_DIM):
    """先 jieba 分词，再对每个词做 MD5 哈希映射到向量维度。"""
    import jieba
    vec = [0.0] * dim
    words = list(jieba.cut(text))
    for word in words:
        if not word.strip():
            continue
        h = hashlib.md5(word.encode("utf-8")).hexdigest()
        bucket = int(h, 16) % dim
        vec[bucket] += 1.0
    length = math.sqrt(sum(x * x for x in vec))
    if length == 0:
        return [0.0] * dim
    return [round(x / length, 4) for x in vec]

print(f"\n  维度={WORD_DIM}，哈希方式=MD5")
for title, text in zip(titles, texts):
    vec = word_hash_vector(text)
    import jieba
    words = list(jieba.cut(text))
    print(f"  {title[:20]}: 分词={list(words)[:6]}... → {vec}")

# ═══════════════════════════════════════════
# 相似度计算（用哈希向量做初次检索）
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("用哈希向量做相似度检索")
print("━" * 60)

query_vec = char_hash_vector(question)
doc_vecs = [char_hash_vector(d["content"]) for d in docs]

def cosine(vec_a, vec_b):
    return sum(a * b for a, b in zip(vec_a, vec_b))

print(f"\n  查询：{question}")
print(f"  查询向量：{query_vec}\n")

scores = []
for i, (doc, vec) in enumerate(zip(docs, doc_vecs)):
    sim = cosine(query_vec, vec)
    # 同时计算欧氏距离
    euclidean = math.sqrt(sum((a - b) ** 2 for a, b in zip(query_vec, vec)))
    scores.append((sim, euclidean, doc["title"], vec))

scores.sort(key=lambda x: x[0], reverse=True)

print(f"  {'排名':<6}{'文档':<16}{'余弦相似度':<12}{'欧氏距离':<12}")
print("  " + "-" * 48)
for rank, (sim, euc, title, _) in enumerate(scores, 1):
    bar = "█" * min(int(sim * 20), 20)
    print(f"  {rank:<6}{title:<16}{sim:<12.4f}{euc:<12.4f} {bar}")

# ═══════════════════════════════════════════
# 维度选择的影响
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("维度选择对区分度的影响")
print("━" * 60)

# 计算不同维度下文档之间的最小相似度（区分度）
print(f"\n  {'维度':<8}{'最小相似度':<14}{'最大相似度':<14}{'区分度（max-min）'}")
for dim in [4, 8, 16, 32, 64]:
    vecs = []
    for doc in docs:
        vec = [0.0] * dim
        for ch in doc["content"]:
            vec[ord(ch) % dim] += 1.0
        length = math.sqrt(sum(x * x for x in vec)) or 1
        vecs.append([x / length for x in vec])

    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            s = sum(a * b for a, b in zip(vecs[i], vecs[j]))
            sims.append(s)

    min_sim, max_sim = min(sims) if sims else 0, max(sims) if sims else 0
    print(f"  {dim:<8}{min_sim:<14.4f}{max_sim:<14.4f}{max_sim - min_sim:.4f}")

print("\n  结论：维度越高，不同文档之间的相似度差异越大，区分度越好。")

# ═══════════════════════════════════════════
# 嵌入模型简介
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("从哈希向量到嵌入模型")
print("━" * 60)
print("""
  哈希向量：
    - 优点：快速、可解释、不需要模型
    - 缺点：无法捕捉语义相似性，'买票'和'创建订单'完全不同

  嵌入模型（Embedding Model）：
    - 基于神经网络训练，能捕捉语义
    - 常用中文模型：BGE-m3, text2vec-large-chinese, m3e
    - 同样的含义 → 相似的向量
    - 例如：'候补申请' 和 '排队买票' 在嵌入空间中会很接近

  调用示例（伪代码）：
    from openai import OpenAI
    client = OpenAI()
    vec = client.embeddings.create(
        model="text-embedding-3-small",
        input="候补申请不能保证成功"
    ).data[0].embedding
    # vec 是一个 1536 维的浮点数列表
""")

print("=" * 72)
print("学习要点：")
print("  1. 向量化 = 把文本变成数字，让机器能计算相似度")
print("  2. 哈希向量是最简单的向量化方法，适合教学演示")
print("  3. 生产环境使用嵌入模型（BGE-m3 等）")
print("  4. 维度越高区分度越好，但有边际效应")