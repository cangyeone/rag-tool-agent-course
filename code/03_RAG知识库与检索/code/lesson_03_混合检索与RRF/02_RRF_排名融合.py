"""02_RRF_排名融合。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
学习目标：掌握 RRF（Reciprocal Rank Fusion）排名融合算法，
         理解 k 参数对融合结果的影响，对比线性加权 vs RRF 的优势。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from collections import Counter
import math
import jieba
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
print("02_RRF_排名融合 —— k 参数敏感性分析")
print("=" * 72)

# ═══════════════════════════════════════════
# 一、生成两个排序列表（关键词 + 向量）
# ═══════════════════════════════════════════
print("\n一、分别计算关键词排序和向量排序")

# 关键词排名（BM25）
query_tokens = [w for w in jieba.cut(question) if w.strip()]
all_contents = [d["content"] for d in docs]
N = len(docs)
avgdl = sum(len(c) for c in all_contents) / N

def idf(term, docs_list):
    df = sum(1 for d in docs_list if term in d)
    return math.log((N + 1) / (df + 1)) + 1

def bm25(query_tokens, doc_text, docs_list):
    score = 0
    doc_len = len(doc_text)
    k1, b = 1.5, 0.75
    for term in query_tokens:
        tf = doc_text.count(term)
        if tf == 0:
            continue
        score += idf(term, docs_list) * (tf * (k1 + 1)) / (
            tf + k1 * (1 - b + b * (doc_len / avgdl)))
    return score

keyword_scored = [(bm25(query_tokens, d["content"], all_contents), d["title"])
                  for d in docs]
keyword_scored.sort(key=lambda x: x[0], reverse=True)
keyword_ranks = {title: rank for rank, (_, title) in enumerate(keyword_scored, 1)}

# 向量排名（余弦相似度）
DIM = 8
def make_vector(text):
    vec = [0.0] * DIM
    for ch in text:
        vec[ord(ch) % DIM] += 1.0
    length = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / length for x in vec]

def cosine(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))

query_vec = make_vector(question)
vector_scored = [(cosine(query_vec, make_vector(d["content"])), d["title"])
                 for d in docs]
vector_scored.sort(key=lambda x: x[0], reverse=True)
vector_ranks = {title: rank for rank, (_, title) in enumerate(vector_scored, 1)}

print(f"\n  关键词排序：{' > '.join(t for _, t in keyword_scored)}")
print(f"  向量排序：  {' > '.join(t for _, t in vector_scored)}")

# ═══════════════════════════════════════════
# 二、RRF 算法核心
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("二、RRF 算法原理")
print("━" * 60)

print("""
  RRF 公式：score(d) = Σ 1 / (k + rank_i(d))

  其中：
  - rank_i(d) 是文档 d 在第 i 个排序列表中的排名（从 1 开始）
  - k 是平滑参数，防止排名为 1 时得分过大
  - 对所有排序列表求和

  核心思想：
  - 排名靠前 → 得分高（分母小）
  - 多个列表排名都靠前 → 得分叠加 → 总排名大幅提升
  - k 控制排名差异的影响程度
""")

# ═══════════════════════════════════════════
# 三、k 参数敏感性分析
# ═══════════════════════════════════════════
print("━" * 60)
print("三、k 参数敏感性分析：不同 k 值下的融合结果")
print("━" * 60)

def rrf_fusion(rank_lists, k):
    """执行 RRF 融合。rank_lists 是多个排序列表的集合。"""
    rrf_score = Counter()
    doc_names = set()
    for rank_list in rank_lists:
        for rank, name in enumerate(rank_list, 1):
            rrf_score[name] += 1.0 / (k + rank)
            doc_names.add(name)
    return rrf_score.most_common()

keyword_list = [t for _, t in keyword_scored]
vector_list = [t for _, t in vector_scored]

k_values = [1, 5, 10, 30, 60, 120, 200]
print(f"\n  {'k值':<8}", end="")
for k in k_values:
    print(f"{'k=' + str(k):<20}", end="")
print()
print("  " + "-" * (8 + 20 * len(k_values)))

# 展示每个 k 值下的前 3 名
for rank_pos in range(4):
    line = f"  {'#' + str(rank_pos + 1):<8}"
    for k in k_values:
        results = rrf_fusion([keyword_list, vector_list], k)
        if rank_pos < len(results):
            name, score = results[rank_pos]
            line += f"{name[:14]:<16}{score:.4f} "
        else:
            line += " " * 20
    print(line)

# ═══════════════════════════════════════════
# 四、k 值的实际影响
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("四、k 值的含义与选择")
print("━" * 60)

print("""
  k=1:   排名差异被极度放大。
         rank 1 的文档得分为 1/(1+1)=0.5
         rank 4 的文档得分仅为 1/(1+4)=0.2
         适合：两个列表高度一致时

  k=60:  经典取值，排名差异被适度平滑。
         rank 1: 1/61≈0.0164
         rank 4: 1/64≈0.0156（差距不大）
         适合：两个列表有分歧时的通用场景

  k=200: 极度平滑，几乎只看文档是否出现在列表中，
         不太关心具体排名。
         适合：列表较多（3+）或者排名不可靠时
""")

# ═══════════════════════════════════════════
# 五、RRF vs 线性加权对比
# ═══════════════════════════════════════════
print("━" * 60)
print("五、RRF vs 线性加权对比")
print("━" * 60)

# 获取两个列表的排名
kw_order = keyword_list
vec_order = vector_list

# RRF 融合（k=60）
rrf_result = rrf_fusion([kw_order, vec_order], k=60)

print(f"\n  {'方法':<20}{'排序结果'}")
print("  " + "-" * 52)

# 关键词
print(f"  {'关键词':<20}{' > '.join(kw_order)}")
# 向量
print(f"  {'向量':<20}{' > '.join(vec_order)}")
# RRF
rrf_names = [name for name, _ in rrf_result]
print(f"  {'RRF(k=60)':<20}{' > '.join(rrf_names)}")

# 对比分析
print(f"\n  排名变化分析：")
for doc in docs:
    title = doc["title"]
    kw_r = kw_order.index(title) + 1
    vec_r = vec_order.index(title) + 1
    rrf_r = rrf_names.index(title) + 1 if title in rrf_names else "-"
    change = ""
    if isinstance(rrf_r, int):
        if rrf_r < min(kw_r, vec_r):
            change = "↑ 融合后提升"
        elif rrf_r > max(kw_r, vec_r):
            change = "↓ 融合后降低"
        else:
            change = "→ 在两者之间"
    print(f"    {title}: 关键词#{kw_r} → 向量#{vec_r} → RRF#{rrf_r} {change}")

# ═══════════════════════════════════════════
# 六、多个排序列表的 RRF
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("六、多列表 RRF（3 个排序列表）")
print("━" * 60)

# 模拟第三个排序列表：按文档长度排序
length_list = sorted(docs, key=lambda d: len(d["content"]), reverse=False)
length_order = [d["title"] for d in length_list]
print(f"\n  第三列表（按文档长度）：{' > '.join(length_order)}")

three_list_result = rrf_fusion([kw_order, vec_order, length_order], k=60)
print(f"\n  三列表 RRF 结果（k=60）：")
for rank, (name, score) in enumerate(three_list_result, 1):
    bar = "█" * int(score * 200)
    print(f"    #{rank} {name:<16} RRF={score:.4f} {bar}")

print("\n" + "=" * 72)
print("学习要点：")
print("  1. RRF = 对排名取倒数求和，避免了分数归一化的麻烦")
print("  2. k 值控制平滑程度：k 越小排名差异越大，k 越大越平滑")
print("  3. 经典取值 k=60，适合大多数场景")
print("  4. RRF 天然支持 3+ 个排序列表的融合")