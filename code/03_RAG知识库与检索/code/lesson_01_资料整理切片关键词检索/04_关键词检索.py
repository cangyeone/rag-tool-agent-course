"""04_关键词检索。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
学习目标：掌握关键词检索的基本算法（TF 词频、BM25 思想），
         理解词法匹配的优势与局限。
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
print("04_关键词检索 —— 从 TF 到 BM25")
print("=" * 72)

print(f"\n用户问题：{question}")

# ── 中文分词 ──
def tokenize(text):
    return [w for w in jieba.cut(text) if w.strip()]

query_tokens = tokenize(question)
print(f"\n分词结果：{' | '.join(query_tokens)}")

# ═══════════════════════════════════════════
# 方法一：简单词频匹配（TF Scoring）
# ═══════════════════════════════════════════
print("\n" + "━" * 50)
print("方法一：简单词频匹配（TF）")
print("━" * 50)

def tf_score(query_tokens, doc_text):
    doc_lower = doc_text
    score = 0
    detail = {}
    for term in query_tokens:
        count = doc_lower.count(term)
        score += count
        detail[term] = count
    return score, detail

print(f"\n  {'文档':<16}{'得分':<6}{'各词命中详情'}")
print("  " + "-" * 54)
for doc in docs:
    score, detail = tf_score(query_tokens, doc["content"])
    detail_str = " ".join(f"{k}:{v}" for k, v in detail.items() if v > 0)
    print(f"  {doc['title']:<16}{score:<6}{detail_str}")

print("\n  ⚠ TF 的问题：长文档天然得分高，但不一定更相关。")

# ═══════════════════════════════════════════
# 方法二：TF-IDF 思想简化版
# ═══════════════════════════════════════════
print("\n" + "━" * 50)
print("方法二：TF-IDF 简化版")
print("━" * 50)

all_contents = [d["content"] for d in docs]
N = len(docs)

def idf(term, docs_list):
    df = sum(1 for d in docs_list if term in d)
    return math.log((N + 1) / (df + 1)) + 1

print("\n  计算 IDF 值：")
for term in query_tokens:
    idf_val = idf(term, all_contents)
    df = sum(1 for d in all_contents if term in d)
    print(f"    '{term}': 出现在 {df}/{N} 篇 → IDF={idf_val:.4f}")

def tfidf_score(query_tokens, doc_text, docs_list):
    score = 0
    detail = {}
    for term in query_tokens:
        tf = doc_text.count(term)
        idf_val = idf(term, docs_list)
        contribution = tf * idf_val
        score += contribution
        if contribution > 0:
            detail[term] = f"tf={tf}×idf={idf_val:.2f}={contribution:.2f}"
    return score, detail

print(f"\n  {'文档':<16}{'TF-IDF得分':<12}{'贡献详情'}")
print("  " + "-" * 60)
for doc in docs:
    score, detail = tfidf_score(query_tokens, doc["content"], all_contents)
    detail_str = " | ".join(f"{k}:{v}" for k, v in detail.items())
    print(f"  {doc['title']:<16}{score:<12.4f}{detail_str if detail_str else '无命中'}")

# ═══════════════════════════════════════════
# 方法三：BM25 简化版
# ═══════════════════════════════════════════
print("\n" + "━" * 50)
print("方法三：BM25 简化版（含文档长度归一化）")
print("━" * 50)

avgdl = sum(len(d) for d in all_contents) / len(all_contents)
k1, b = 1.5, 0.75

def bm25_score(query_tokens, doc_text, doc_idx, docs_list):
    score = 0
    doc_len = len(doc_text)
    detail = {}
    for term in query_tokens:
        tf = doc_text.count(term)
        if tf == 0:
            continue
        idf_val = idf(term, docs_list)
        # BM25 的 TF 饱和公式
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
        term_score = idf_val * numerator / denominator
        score += term_score
        detail[term] = f"tf={tf}→{term_score:.3f}"
    return score, detail

print(f"\n  参数：avgdl={avgdl:.1f}, k1={k1}, b={b}")
print(f"\n  {'文档':<16}{'BM25得分':<12}{'贡献详情'}")
print("  " + "-" * 60)
results = []
for i, doc in enumerate(docs):
    score, detail = bm25_score(query_tokens, doc["content"], i, all_contents)
    results.append((score, doc, detail))
    detail_str = " | ".join(f"{k}:{v}" for k, v in detail.items())
    print(f"  {doc['title']:<16}{score:<12.4f}{detail_str if detail_str else '无命中'}")

# ═══════════════════════════════════════════
# 最终排序
# ═══════════════════════════════════════════
print("\n" + "━" * 50)
print("三种方法最终排序对比")
print("━" * 50)

methods = {"TF（词频）": lambda d: tf_score(query_tokens, d["content"])[0],
           "TF-IDF": lambda d: tfidf_score(query_tokens, d["content"], all_contents)[0],
           "BM25": lambda d: bm25_score(query_tokens, d["content"], 0, all_contents)[0]}

for method_name, score_fn in methods.items():
    ranked = sorted(docs, key=score_fn, reverse=True)
    ranking = " > ".join(d["title"] for d in ranked)
    print(f"  {method_name:<12}: {ranking}")

print("\n" + "=" * 72)
print("学习要点：")
print("  1. TF 简单但偏向长文档")
print("  2. TF-IDF 用 IDF 惩罚常见词，提升稀有词权重")
print("  3. BM25 引入文档长度归一化 + TF 饱和，更贴近实际")
print('  4. 关键词检索的局限：无法理解同义词和语义（如「买票」≠「创建订单」）')