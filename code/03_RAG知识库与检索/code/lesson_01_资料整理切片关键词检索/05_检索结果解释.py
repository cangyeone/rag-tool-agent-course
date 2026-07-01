"""05_检索结果解释。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
学习目标：学会解释检索结果——为什么某条文档排在前面？
         掌握置信度计算、命中词高亮、结果可解释性三要素。
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
print("05_检索结果解释 —— 置信度与可解释性")
print("=" * 72)

print(f"\n用户问题：{question}")

# ── 分词 ──
query_tokens = [w for w in jieba.cut(question) if w.strip()]
print(f"分词结果：{' | '.join(query_tokens)}")

# ═══════════════════════════════════════════
# 计算 BM25 得分 + 解释
# ═══════════════════════════════════════════
all_contents = [d["content"] for d in docs]
N = len(docs)
avgdl = sum(len(c) for c in all_contents) / N

def idf(term, docs_list):
    df = sum(1 for d in docs_list if term in d)
    return math.log((N + 1) / (df + 1)) + 1

def bm25_with_explanations(query_tokens, doc_text, docs_list):
    score = 0
    term_details = []
    doc_len = len(doc_text)
    k1, b = 1.5, 0.75
    for term in query_tokens:
        tf = doc_text.count(term)
        if tf == 0:
            continue
        idf_val = idf(term, docs_list)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
        term_score = idf_val * numerator / denominator
        score += term_score
        # 找出该词在文档中的所有位置（用于高亮）
        positions = []
        idx = 0
        while True:
            idx = doc_text.find(term, idx)
            if idx == -1:
                break
            positions.append(idx)
            idx += 1
        term_details.append({
            "term": term,
            "tf": tf,
            "idf": round(idf_val, 3),
            "term_score": round(term_score, 4),
            "positions": positions
        })
    return round(score, 4), term_details

print("\n" + "━" * 60)
print("检索结果详细解释（BM25）")
print("━" * 60)

results = []
for doc in docs:
    score, details = bm25_with_explanations(query_tokens, doc["content"], all_contents)
    results.append((score, doc, details))

results.sort(key=lambda x: x[0], reverse=True)

for rank, (score, doc, details) in enumerate(results, 1):
    print(f"\n{'─' * 60}")
    print(f"排名 #{rank}：{doc['title']}")
    print(f"BM25 得分：{score}")
    print(f"\n原文：{doc['content']}")

    # 高亮命中词
    if details:
        print("\n命中词分析：")
        for det in details:
            bar = "█" * min(int(det["term_score"] * 20), 20)
            print(f"  '{det['term']}': TF={det['tf']}, IDF={det['idf']}, "
                  f"贡献={det['term_score']} {bar}")

        # 显示高亮文本
        highlighted = doc["content"]
        for det in details:
            highlighted = highlighted.replace(
                det["term"], f"【{det['term']}】")
        print(f"\n高亮文本：{highlighted}")
    else:
        print("  该文档未命中任何查询词。")

# ═══════════════════════════════════════════
# 置信度计算与分级
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("置信度分级")
print("━" * 60)

max_score = results[0][0] if results else 1

print(f"\n  最高分：{max_score}（归一化基准）")
print(f"\n  {'排名':<6}{'文档':<16}{'原始分':<10}{'置信度':<10}{'等级'}")
print("  " + "-" * 50)

for rank, (score, doc, details) in enumerate(results, 1):
    confidence = score / max_score * 100 if max_score > 0 else 0
    if confidence >= 70:
        grade = "★★★★★ 高置信"
    elif confidence >= 40:
        grade = "★★★☆☆ 中置信"
    elif confidence > 0:
        grade = "★☆☆☆☆ 低置信"
    else:
        grade = "☆☆☆☆☆ 无关联"

    print(f"  {rank:<6}{doc['title']:<16}{score:<10}{confidence:<10.1f}%{grade}")

# ═══════════════════════════════════════════
# 回答生成建议
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("基于检索结果的回答生成建议")
print("━" * 60)

top_doc = results[0][1]
print(f"\n  首选依据：{top_doc['title']}")
print(f"  依据内容：{top_doc['content']}")
print(f"\n  建议回答：")
print(f"  根据《{top_doc['title']}》，{top_doc['content']}")
print(f"\n  同时，以下文档也可辅助回答：")
for rank, (score, doc, details) in enumerate(results[1:], 2):
    if score > 0:
        print(f"    #{rank} {doc['title']}（得分 {score}，可参考）")

print("\n" + "=" * 72)
print("学习要点：")
print("  1. 每个检索结果都应有「为什么排在这」的解释")
print("  2. 命中词高亮帮助用户验证相关性")
print("  3. 置信度分级决定回答策略（直接回答 vs 追问澄清）")
print("  4. 多文档互补可提高回答完整性")