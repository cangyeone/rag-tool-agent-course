"""05_召回质量检查。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
学习目标：掌握检索召回质量的评估方法——精确率、召回率、
         MRR（平均倒数排名）、NDCG（归一化折损累计增益）。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import math
import time
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


DIM = 8

def make_vector(text, dim=DIM):
    vec = [0.0] * dim
    for ch in text:
        vec[ord(ch) % dim] += 1.0
    length = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [round(x / length, 4) for x in vec]

def cosine(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))

print("=" * 72)
print("05_召回质量检查 —— Precision, Recall, MRR, NDCG")
print("=" * 72)

# ═══════════════════════════════════════════
# 一、构建测试集：查询 + 人工标注的相关文档
# ═══════════════════════════════════════════
print("\n一、构建评估测试集")
print("  人工标注：对每个查询，标记哪些文档是真正相关的。")

# 测试查询及其相关文档（用 title 标识）
test_queries = [
    {
        "query": "候补申请能保证成功吗？",
        "relevant": ["候补申请规则"],  # 只有这一篇真正相关
        "partially": ["退款与变更提醒"],    # 部分相关
    },
    {
        "query": "退款和变更怎么办理？",
        "relevant": ["退款与变更提醒"],
        "partially": ["候补申请规则"],
    },
    {
        "query": "示例服务点在哪？",
        "relevant": ["服务点说明"],
        "partially": [],
    },
    {
        "query": "GPR是什么设备？",
        "relevant": ["GPR 安全检查"],
        "partially": [],
    },
    {
        "query": "示例业务系统的退款规则",
        "relevant": ["退款与变更提醒", "候补申请规则"],
        "partially": [],
    },
]

for tc in test_queries:
    print(f"  Q: '{tc['query']}' → 相关: {tc['relevant']}，"
          f"部分相关: {tc['partially']}")

# ═══════════════════════════════════════════
# 二、执行检索（使用上一课的余弦检索函数）
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("二、执行检索并收集结果")
print("━" * 60)

# 为所有文档生成向量（如果没有向量库文件，实时生成）
doc_vectors = [make_vector(d["content"]) for d in docs]

def search(query_text, top_k=4):
    query_vec = make_vector(query_text)
    scores = []
    for i, (doc, vec) in enumerate(zip(docs, doc_vectors)):
        sim = cosine(query_vec, vec)
        scores.append((sim, doc))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [doc["title"] for _, doc in scores[:top_k]]

retrieval_results = {}
for tc in test_queries:
    results = search(tc["query"], top_k=4)
    retrieval_results[tc["query"]] = results
    print(f"  Q: '{tc['query']}'")
    print(f"  检索结果: {results}（期望相关: {tc['relevant']}）")
    print()

# ═══════════════════════════════════════════
# 三、Precision & Recall @K
# ═══════════════════════════════════════════
print("━" * 60)
print("三、Precision（精确率）& Recall（召回率）@K")
print("━" * 60)

def precision_at_k(retrieved, relevant, k):
    """前 K 个结果中相关文档的占比。"""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for r in top_k if r in relevant) / len(top_k)

def recall_at_k(retrieved, relevant, k):
    """相关文档中被召回的比例。"""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for r in top_k if r in relevant) / len(relevant)

print(f"\n  {'查询':<24}{'P@1':<8}{'P@3':<8}{'R@1':<8}{'R@3':<8}")
print("  " + "-" * 42)

total_p1 = total_p3 = total_r1 = total_r3 = 0

for tc in test_queries:
    retrieved = retrieval_results[tc["query"]]
    relevant = tc["relevant"]
    p1 = precision_at_k(retrieved, relevant, 1)
    p3 = precision_at_k(retrieved, relevant, 3)
    r1 = recall_at_k(retrieved, relevant, 1)
    r3 = recall_at_k(retrieved, relevant, 3)
    total_p1 += p1
    total_p3 += p3
    total_r1 += r1
    total_r3 += r3
    print(f"  {tc['query']:<24}{p1:<8.2f}{p3:<8.2f}{r1:<8.2f}{r3:<8.2f}")

n = len(test_queries)
print("  " + "-" * 42)
print(f"  {'平均':<24}{total_p1/n:<8.2f}{total_p3/n:<8.2f}{total_r1/n:<8.2f}{total_r3/n:<8.2f}")

# ═══════════════════════════════════════════
# 四、MRR（Mean Reciprocal Rank）
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("四、MRR（平均倒数排名）")
print("━" * 60)

print("  定义：第一个相关文档排名的倒数。")
print("  MRR 越高说明第一个正确答案出现得越早。")

def mrr(retrieval_results_dict, test_queries_list):
    rr_values = []
    for tc in test_queries_list:
        retrieved = retrieval_results_dict[tc["query"]]
        relevant = tc["relevant"]
        rr = 0.0
        for rank, doc_title in enumerate(retrieved, 1):
            if doc_title in relevant:
                rr = 1.0 / rank
                break
        rr_values.append((tc["query"], rr, rank if rr > 0 else 0))
    return rr_values, sum(rr for _, r, _ in rr_values) / len(rr_values)

rr_values, mrr_score = mrr(retrieval_results, test_queries)

print(f"\n  {'查询':<26}{'首个相关排名':<14}{'RR'}")
print("  " + "-" * 44)
for query, rr, rank in rr_values:
    print(f"  {query:<26}{'#' + str(rank) if rank > 0 else '未找到':<14}{rr:.4f}")
print("  " + "-" * 44)
print(f"  MRR = {mrr_score:.4f}")
print(f"\n  解读：MRR={mrr_score:.2f}，表示平均在", end="")
if mrr_score > 0:
    print(f"第 {1/mrr_score:.1f} 位找到正确答案。")

# ═══════════════════════════════════════════
# 五、NDCG（Normalized Discounted Cumulative Gain）
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("五、NDCG（归一化折损累计增益）")
print("━" * 60)

print("  定义：考虑排序位置权重的评估指标。靠前的正确结果贡献更大。")
print("  DCG = Σ (2^rel_i - 1) / log₂(rank_i + 1)")
print("  NDCG = DCG / IDCG（IDCG = 理想排序下的 DCG）\n")

def ndcg_at_k(retrieved, relevant, partially, k=4):
    """计算 NDCG@K。
    relevant: 完全相关（gain=3）
    partially: 部分相关（gain=1）
    不相关（gain=0）
    """
    gains = []
    for r in retrieved[:k]:
        if r in relevant:
            gains.append(3)
        elif r in partially:
            gains.append(1)
        else:
            gains.append(0)

    dcg = 0.0
    for i, gain in enumerate(gains):
        dcg += gain / math.log2(i + 2)  # i+2 因为 rank 从 1 开始

    # 理想 DCG：把所有相关（gain=3）排最前面，部分相关排中间
    ideal_gains = sorted([3] * len(relevant) + [1] * len(partially),
                         reverse=True)
    # 补齐到 k 个
    ideal_gains = ideal_gains[:k] + [0] * max(0, k - len(ideal_gains))

    idcg = 0.0
    for i, gain in enumerate(ideal_gains):
        idcg += gain / math.log2(i + 2)

    return dcg, idcg, dcg / idcg if idcg > 0 else 0.0

print(f"  {'查询':<24}{'DCG@4':<10}{'IDCG@4':<10}{'NDCG@4'}")
print("  " + "-" * 46)

total_ndcg = 0
for tc in test_queries:
    retrieved = retrieval_results[tc["query"]]
    dcg, idcg, ndcg = ndcg_at_k(retrieved, tc["relevant"], tc["partially"])
    total_ndcg += ndcg
    print(f"  {tc['query']:<24}{dcg:<10.4f}{idcg:<10.4f}{ndcg:.4f}")

print("  " + "-" * 46)
print(f"  {'平均 NDCG@4':<24}{'':<10}{'':<10}{total_ndcg/len(test_queries):.4f}")

# ═══════════════════════════════════════════
# 六、综合评估报告
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("六、综合评估报告")
print("━" * 60)

report = f"""
  召回质量评估报告
  ──────────────────
  测试查询数：{len(test_queries)}
  文档总数：{len(docs)}
  向量维度：{DIM}

  指标汇总：
    P@1       = {total_p1/n:.4f}  （第一条是否相关）
    P@3       = {total_p3/n:.4f}  （前三条中相关比例）
    R@1       = {total_r1/n:.4f}  （相关文档中排第一的比例）
    R@3       = {total_r3/n:.4f}  （相关文档中前三召回比例）
    MRR       = {mrr_score:.4f}  （平均倒数排名）
    NDCG@4    = {total_ndcg/n:.4f}  （归一化折损累计增益）

  结论：
    - 哈希向量（{DIM}维）的语义表达能力有限
    - 使用嵌入模型（如 BGE-m3）后，指标预期翻倍
    - NDCG > 0.5 可视为可接受的检索质量
"""

print(report)

print("=" * 72)
print("学习要点：")
print("  1. Precision = 检索结果中相关的比例（查得准不准）")
print("  2. Recall = 相关文档中被找到的比例（查得全不全）")
print("  3. MRR = 第一个正确答案排名的倒数（排序质量）")
print("  4. NDCG = 考虑排序位置的累积增益（综合质量）")