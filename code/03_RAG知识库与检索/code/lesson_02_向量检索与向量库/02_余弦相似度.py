"""02_余弦相似度。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
学习目标：深入理解余弦相似度的数学原理，区分点积与余弦相似度，
         通过角度可视化理解「相似」的几何含义。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
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
print("02_余弦相似度 —— 数学原理与可视化理解")
print("=" * 72)

# ═══════════════════════════════════════════
# 一、构造哈希向量（复用上一课的生成逻辑）
# ═══════════════════════════════════════════
print("\n一、构造向量（8 维哈希向量）")

DIM = 8

def make_vector(text):
    vec = [0.0] * DIM
    for ch in text:
        vec[ord(ch) % DIM] += 1.0
    length = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [round(x / length, 4) for x in vec]

all_texts = [question] + [d["content"] for d in docs]
all_vectors = [make_vector(t) for t in all_texts]

query_vec = all_vectors[0]
print(f"\n  查询向量（{question}）：")
print(f"  {query_vec}")

# ═══════════════════════════════════════════
# 二、点积（Dot Product）vs 余弦相似度
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("二、点积 vs 余弦相似度 —— 关键区别")
print("━" * 60)

print("""
  点积（Dot Product）：
    公式：a·b = Σ aᵢ × bᵢ
    范围：[-∞, +∞]（与向量长度有关）
    含义：向量长度 × 投影长度

  余弦相似度（Cosine Similarity）：
    公式：cos(θ) = (a·b) / (||a|| · ||b||)
    范围：[-1, +1]
    含义：两个向量夹角的余弦值，只关心方向，不关心长度

  关键区别：余弦相似度做了「归一化」，
  消除了向量长度的影响，更关注语义方向。
""")

# 演示：同一方向但不同长度的向量
print("  演示：同一方向，不同长度的向量")
v1 = [1.0, 2.0, 3.0]          # 长度 ≈ 3.74
v2 = [2.0, 4.0, 6.0]          # 长度 ≈ 7.48（v1 的 2 倍）
dot = sum(a * b for a, b in zip(v1, v2))
len1 = math.sqrt(sum(x * x for x in v1))
len2 = math.sqrt(sum(x * x for x in v2))
cos_sim = dot / (len1 * len2)
print(f"    v1={v1}")
print(f"    v2={v2}")
print(f"    点积={dot:.4f}  → v2 比 v1 长，点积也翻倍")
print(f"    余弦相似度={cos_sim:.4f}  → 完全相同方向，不受长度影响")

# ═══════════════════════════════════════════
# 三、余弦相似度 —— 逐文档计算
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("三、逐文档计算余弦相似度")
print("━" * 60)

def cosine(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))

def euclidean_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

def angle_from_cosine(cos_val):
    """将余弦值转为角度（度）。"""
    cos_val = max(-1.0, min(1.0, cos_val))
    return math.degrees(math.acos(cos_val))

title_list = ["[查询]" + question] + [d["title"] for d in docs]

print(f"\n  {'文档':<22}{'余弦相似度':<12}{'夹角':<10}{'欧氏距离':<12}")
print("  " + "-" * 54)

for i, (title, vec) in enumerate(zip(title_list[1:], all_vectors[1:])):
    cos_sim = cosine(query_vec, vec)
    angle = angle_from_cosine(cos_sim)
    euc_dist = euclidean_distance(query_vec, vec)
    # 可视化相似度条
    bar_len = int((cos_sim + 1) / 2 * 20)
    bar = "█" * bar_len + "░" * (20 - bar_len)
    print(f"  {title:<22}{cos_sim:<12.4f}{angle:<10.1f}°{euc_dist:<12.4f} {bar}")

# ═══════════════════════════════════════════
# 四、角度可视化
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("四、角度直观理解")
print("━" * 60)

print("""
  余弦相似度 → 夹角 → 语义关系：
    1.0     →    0° → 完全相同方向（同义）
    0.7~0.9 → 26°~45° → 高度相关
    0.5~0.7 → 45°~60° → 中等相关
    0.0~0.5 → 60°~90° → 弱相关
    0.0     →   90° → 正交（无关）
   -1.0     →  180° → 完全相反（反义）
""")

# ═══════════════════════════════════════════
# 五、不同查询的相似度对比
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("五、同一批文档对不同查询的余弦相似度")
print("━" * 60)

test_queries = [
    "候补申请一定能成功吗？",
    "退款怎么办理？",
    "北京到上海的服务点有哪些？",
    "地质雷达是什么？",
    "今天天气怎么样？",  # 无关查询
]

doc_vecs = all_vectors[1:]  # 跳过查询向量本身

for q in test_queries:
    qvec = make_vector(q)
    print(f"\n  查询：'{q}'")

    sims = []
    for d, v in zip(docs, doc_vecs):
        c = cosine(qvec, v)
        sims.append((c, d["title"]))

    sims.sort(reverse=True)
    best = sims[0]
    bar = "█" * min(int(best[0] * 30), 30)
    print(f"    最佳匹配：{best[1]}（余弦={best[0]:.4f}） {bar}")
    if best[0] < 0.3:
        print(f"    ⚠ 相似度很低（<0.3），知识库可能没有相关内容")

# ═══════════════════════════════════════════
# 六、所有文档对之间的相似度矩阵
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("六、文档间的相似度矩阵")
print("━" * 60)

print(f"\n  {'':<16}", end="")
for d in docs:
    print(f"{d['title'][:6]:<10}", end="")
print()

for i, d1 in enumerate(docs):
    print(f"  {d1['title']:<16}", end="")
    for j, d2 in enumerate(docs):
        c = cosine(doc_vecs[i], doc_vecs[j])
        print(f"{c:<10.4f}", end="")
    print()

print("\n  解读：对角线全为 1.0（自相似），离开对角线越小越不相关。")

# ═══════════════════════════════════════════
# 七、余弦相似度的问题
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("七、余弦相似度的局限")
print("━" * 60)

print("""
  1. 对向量质量敏感 —— 哈希向量语义能力弱，需嵌入模型
  2. 不区分「重要词」和「停用词」—— 所有维度平等对待
  3. 高维稀疏向量会退化为接近正交
  4. 无法捕捉顺序和语法结构
""")

print("=" * 72)
print("学习要点：")
print("  1. 余弦相似度 = 方向相似，点积 = 长度×方向")
print("  2. 0° 完全同义，90° 无关，180° 完全反义")
print("  3. 归一化向量时，余弦相似度 = 点积")
print("  4. 实际系统先用余弦相似度召回 Top-K，再重排序")