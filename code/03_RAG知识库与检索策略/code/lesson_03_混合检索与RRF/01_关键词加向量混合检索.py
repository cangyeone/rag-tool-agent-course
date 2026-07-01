"""01_关键词加向量混合检索。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
学习目标：将关键词检索与向量检索的结果进行线性加权融合，
         理解不同权重分配对最终排序的影响，找到最优混合比例。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

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


question = "候补申请一定能成功吗？"
print("=" * 72)
print("01_关键词加向量混合检索 —— 线性加权融合")
print("=" * 72)

print(f"\n用户问题：{question}")

# ═══════════════════════════════════════════
# 一、关键词检索（BM25）
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("一、关键词检索得分（BM25）")
print("━" * 60)

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
        idf_val = idf(term, docs_list)
        score += idf_val * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
    return score

keyword_scores = []
for doc in docs:
    ks = bm25(query_tokens, doc["content"], all_contents)
    keyword_scores.append(ks)
    print(f"  {doc['title']}: BM25={ks:.4f}")

# ═══════════════════════════════════════════
# 二、向量检索（哈希向量 + 余弦相似度）
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("二、向量检索得分（余弦相似度）")
print("━" * 60)

DIM = 8

def make_vector(text, dim=DIM):
    vec = [0.0] * dim
    for ch in text:
        vec[ord(ch) % dim] += 1.0
    length = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [round(x / length, 4) for x in vec]

def cosine(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))

query_vec = make_vector(question)
vector_scores = []
for doc in docs:
    vs = cosine(query_vec, make_vector(doc["content"]))
    vector_scores.append(vs)
    print(f"  {doc['title']}: Cosine={vs:.4f}")

# ═══════════════════════════════════════════
# 三、得分归一化（Min-Max）
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("三、得分归一化（Min-Max Normalization）")
print("━" * 60)

def minmax_normalize(scores):
    """将得分线性映射到 [0, 1] 区间。"""
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]

norm_kw = minmax_normalize(keyword_scores)
norm_vec = minmax_normalize(vector_scores)

print(f"\n  {'文档':<16}{'BM25原始':<12}{'BM25归一':<12}{'Cos原始':<12}{'Cos归一'}")
print("  " + "-" * 56)
for i, doc in enumerate(docs):
    print(f"  {doc['title']:<16}{keyword_scores[i]:<12.4f}{norm_kw[i]:<12.4f}"
          f"{vector_scores[i]:<12.4f}{norm_vec[i]:<12.4f}")

print("\n  注意：归一化后两种得分处于同一量纲，可以直接加权求和。")

# ═══════════════════════════════════════════
# 四、混合加权融合
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("四、混合加权融合 —— 不同权重对比")
print("━" * 60)

weight_configs = [
    (0.0, 1.0, "纯向量"),
    (0.3, 0.7, "偏向量"),
    (0.5, 0.5, "均衡"),
    (0.7, 0.3, "偏关键词"),
    (1.0, 0.0, "纯关键词"),
]

def hybrid_score(kw_norm, vec_norm, w_kw, w_vec):
    """线性加权：w_kw * keyword + w_vec * vector"""
    return w_kw * kw_norm + w_vec * vec_norm

for w_kw, w_vec, label in weight_configs:
    print(f"\n  权重配置：{label}（关键词={w_kw}, 向量={w_vec}）")

    scores = []
    for i, doc in enumerate(docs):
        hs = hybrid_score(norm_kw[i], norm_vec[i], w_kw, w_vec)
        scores.append((hs, doc))

    scores.sort(key=lambda x: x[0], reverse=True)
    for rank, (score, doc) in enumerate(scores, 1):
        bar = "█" * int(score * 30)
        print(f"    #{rank} {doc['title']:<16} 混合分={score:.4f} {bar}")

# ═══════════════════════════════════════════
# 五、排序一致性分析
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("五、排序一致性分析 —— 关键词 vs 向量 vs 混合")
print("━" * 60)

# 按各类得分排序得到排名
keyword_rank = sorted(
    range(len(docs)), key=lambda i: keyword_scores[i], reverse=True)
vector_rank = sorted(
    range(len(docs)), key=lambda i: vector_scores[i], reverse=True)
hybrid_rank_balanced = sorted(
    range(len(docs)),
    key=lambda i: hybrid_score(norm_kw[i], norm_vec[i], 0.5, 0.5),
    reverse=True)

print(f"\n  {'文档':<16}{'关键词排名':<12}{'向量排名':<12}{'混合(均衡)排名'}")
print("  " + "-" * 52)
for i in range(len(docs)):
    kw_r = keyword_rank.index(i) + 1
    vec_r = vector_rank.index(i) + 1
    hyb_r = hybrid_rank_balanced.index(i) + 1
    agreement = "✓ 一致" if kw_r == vec_r else f"✗ 不一致（差{abs(kw_r-vec_r)}位）"
    print(f"  {docs[i]['title']:<16}{kw_r:<12}{vec_r:<12}{hyb_r:<12}{agreement}")

# ═══════════════════════════════════════════
# 六、权重的实际影响
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("六、权重选择指南")
print("━" * 60)

print("""
  关键词权重高时：
    ✓ 精确匹配优先（如搜索"G107"能精确命中）
    ✓ 适合查询中有关键数字、术语的场景
    ✗ 无法捕捉语义相近的内容（"买票"找不到"下单"）

  向量权重高时：
    ✓ 语义扩展能力强（同义词、近义词都能召回）
    ✓ 适合自然语言问题
    ✗ 可能召回语义相关但实际不相关的内容

  建议策略：
    1. 初始用 0.5:0.5 均衡跑一遍
    2. 根据业务场景调优（客服：偏向量；日志查询：偏关键词）
    3. 最终使用 RRF 融合（下一课学习），避免权重调参
""")

print("=" * 72)
print("学习要点：")
print("  1. 线性加权 = w_kw × 关键词分 + w_vec × 向量分")
print("  2. 归一化是混合的前提，确保两个分数在同一量纲")
print("  3. 权重需要根据场景调优，没有万能值")
print("  4. RRF（下一课）可以免去权重调参的麻烦")