"""07_BGE-m3真实向量检索与RRF。

本脚本读取 06 生成的真实向量库，并演示：
1. BM25 关键词检索
2. BGE-m3 + FAISS 向量检索
3. RRF 排名融合

运行方式：
    cd rag-tool-agent-course
    python code/03_RAG知识库与检索/code/lesson_03_混合检索与RRF/07_BGE-m3真实向量检索与RRF.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import faiss
import jieba
import numpy as np
from sentence_transformers import SentenceTransformer

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = COURSE_ROOT / "open_models" / "bge-m3"
VECTOR_DIR = SCRIPT_DIR / "真实BGE向量库"
FAISS_PATH = VECTOR_DIR / "bge_m3_示例业务系统.index"
JSONL_PATH = VECTOR_DIR / "bge_m3_metadata.jsonl"
META_PATH = VECTOR_DIR / "bge_m3_index_meta.json"
BUILD_SCRIPT = SCRIPT_DIR / "06_BGE-m3真实向量库写入.py"

print("=" * 72)
print("07_BGE-m3真实向量检索与RRF —— BM25 + BGE + RRF")
print("=" * 72)

if not MODEL_DIR.is_dir():
    raise SystemExit(f"未找到本地 BGE-m3 模型：{MODEL_DIR.relative_to(COURSE_ROOT)}")

if not (FAISS_PATH.exists() and JSONL_PATH.exists() and META_PATH.exists()):
    print("\n未找到真实 BGE 向量库，先自动运行 06_BGE-m3真实向量库写入.py。")
    subprocess.run([sys.executable, str(BUILD_SCRIPT)], cwd=str(COURSE_ROOT), check=True)

meta = json.loads(META_PATH.read_text(encoding="utf-8"))
print("\n一、向量库信息")
print(json.dumps(meta, ensure_ascii=False, indent=2))

index = faiss.read_index(str(FAISS_PATH))
docs = []
for line in JSONL_PATH.read_text(encoding="utf-8").splitlines():
    if line.strip():
        docs.append(json.loads(line))

print(f"FAISS 向量数：{index.ntotal}")
print(f"元数据条数：{len(docs)}")

print("\n二、加载 BGE-m3 查询模型")
start = time.perf_counter()
model = SentenceTransformer(str(MODEL_DIR), device="cpu")
print(f"模型加载耗时：{(time.perf_counter() - start) * 1000:.1f} ms")


def tokenize(text: str) -> list[str]:
    return [w.strip() for w in jieba.cut(text) if len(w.strip()) >= 2]


def bm25_search(query: str, top_k: int = 5) -> list[tuple[float, dict]]:
    tokens = tokenize(query)
    contents = [doc["content"] for doc in docs]
    n_docs = len(docs)
    avgdl = sum(len(c) for c in contents) / max(1, n_docs)

    def idf(term: str) -> float:
        df = sum(1 for content in contents if term in content)
        return math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

    scores = []
    for doc in docs:
        content = doc["content"]
        doc_len = len(content)
        score = 0.0
        for term in tokens:
            tf = content.count(term)
            if tf == 0:
                continue
            k1 = 1.5
            b = 0.75
            score += idf(term) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
        scores.append((score, doc))

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]


def bge_search(query: str, top_k: int = 5) -> list[tuple[float, dict]]:
    query_vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    query_vec = np.asarray(query_vec, dtype="float32")
    scores, row_ids = index.search(query_vec, top_k)

    results = []
    for score, row_id in zip(scores[0], row_ids[0]):
        if row_id < 0:
            continue
        results.append((float(score), docs[int(row_id)]))
    return results


def rrf_fusion(rank_lists: list[list[str]], k: int = 60) -> list[tuple[float, str]]:
    scores = Counter()
    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list, 1):
            scores[doc_id] += 1.0 / (k + rank)
    return scores.most_common()


def doc_by_id(doc_id: str) -> dict:
    for doc in docs:
        if doc["id"] == doc_id:
            return doc
    raise KeyError(doc_id)


queries = [
    "候补申请一定能成功吗？",
    "退款费用和变更要注意什么？",
    "这个课程里 RAG 的流程是什么？",
    "学生优惠票需要满足什么条件？",
    "地质雷达图像能不能直接让模型判断？",
]

print("\n三、BM25、BGE 向量检索与 RRF 对比")
for query in queries:
    print("\n" + "-" * 72)
    print("查询：", query)

    bm25_results = bm25_search(query, top_k=5)
    bge_results = bge_search(query, top_k=5)

    bm25_ids = [doc["id"] for _, doc in bm25_results]
    bge_ids = [doc["id"] for _, doc in bge_results]
    fused = rrf_fusion([bm25_ids, bge_ids], k=60)[:5]

    print("\nBM25 关键词检索：")
    for rank, (score, doc) in enumerate(bm25_results, 1):
        print(f"#{rank} score={score:.4f} | {doc['title']}")

    print("\nBGE-m3 向量检索：")
    for rank, (score, doc) in enumerate(bge_results, 1):
        print(f"#{rank} score={score:.4f} | {doc['title']}")

    print("\nRRF 融合结果：")
    for rank, (doc_id, score) in enumerate(fused, 1):
        doc = doc_by_id(doc_id)
        preview = doc["content"].replace("\n", " ")[:110]
        print(f"#{rank} rrf={score:.6f} | {doc['title']} | {doc['source']}")
        print("   ", preview)

print("\n课堂观察点")
print("1. BM25 对明确词、编号、服务点名称更敏感。")
print("2. BGE-m3 对同义表达和语义近似更友好。")
print("3. RRF 不直接比较两种分数大小，只融合排名，适合把 BM25 和向量检索放在一起。")
print("4. 真实向量库至少需要三类文件：向量索引、元数据、索引说明。")