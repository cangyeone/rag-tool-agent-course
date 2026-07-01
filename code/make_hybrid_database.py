"""make_hybrid_database.py —— 构建向量 + BM25 混合检索数据库（多进程加速版）。

运行方式：
    cd rag-tool-agent-course
    python code/make_hybrid_database.py [--workers N]

参数说明：
    切片长度 2000 字符，overlap 500 字符。
    --workers: 并行工作进程数（默认：CPU核心数）
    输出目录：code/hybrid_kb/
输出文件：
    - faiss.index: FAISS 向量索引
    - vectors.npy: 向量矩阵
    - metadata.jsonl: 元数据
    - bm25_index.pkl: BM25 倒排索引
    - index_meta.json: 索引说明
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import faiss
import jieba
import numpy as np
from sentence_transformers import SentenceTransformer

COURSE_ROOT = Path(__file__).resolve().parent.parent
if not (COURSE_ROOT / "code").is_dir():
    COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

CODE_ROOT = COURSE_ROOT / "code"
MODEL_DIR = COURSE_ROOT / "open_models" / "bge-m3"
OUTPUT_DIR = CODE_ROOT / "hybrid_kb"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 500

FAISS_PATH = OUTPUT_DIR / "faiss.index"
NPY_PATH = OUTPUT_DIR / "vectors.npy"
JSONL_PATH = OUTPUT_DIR / "metadata.jsonl"
BM25_PATH = OUTPUT_DIR / "bm25_index.pkl"
META_PATH = OUTPUT_DIR / "index_meta.json"

print("=" * 72)
print("make_hybrid_database —— 向量 + BM25 混合检索数据库构建（多进程加速版）")
print("=" * 72)

# 解析命令行参数
parser = argparse.ArgumentParser(description="构建混合检索数据库")
parser.add_argument("--workers", type=int, default=None,
                    help="并行工作进程数（默认：CPU核心数）")
args = parser.parse_args()
NUM_WORKERS = args.workers or multiprocessing.cpu_count()
print(f"\n配置：使用 {NUM_WORKERS} 个并行工作进程")

if not MODEL_DIR.is_dir():
    raise SystemExit(f"未找到本地 BGE-m3 模型：{MODEL_DIR}")


# ========== 1. 遍历 Markdown 文档 ==========

def collect_markdown_files(root: Path) -> list[Path]:
    """递归收集 root 下所有 .md 文件，跳过隐藏目录和输出目录。"""
    skip_parts = {"__pycache__", ".venv", "rag_kb", "make_database_output", "hybrid_kb"}
    files: list[Path] = []
    for md_path in sorted(root.rglob("*.md")):
        rel = md_path.relative_to(root)
        if any(part.startswith(".") or part in skip_parts for part in rel.parts):
            continue
        files.append(md_path)
    return files


md_files = collect_markdown_files(CODE_ROOT)
if not md_files:
    raise SystemExit("未在 code/ 下找到任何 .md 文件。")

print(f"\n一、发现 {len(md_files)} 个 Markdown 文件")
for f in md_files:
    print(f"    {f.relative_to(CODE_ROOT)}")


# ========== 2. 切片 ==========

def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """将长文本按段落边界切分，长段落用滑动窗口切分，保留 overlap。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip() and len(p.strip()) >= 10]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip() if current else para
            continue
        if current:
            chunks.append(current)
        if len(para) <= chunk_size:
            current = para
        else:
            for i in range(0, max(len(para) - overlap, 1), chunk_size - overlap):
                piece = para[i:i + chunk_size]
                if len(piece) >= 20:
                    chunks.append(piece)
            current = ""
    if current:
        chunks.append(current)

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    merged: list[str] = []
    for idx, chunk in enumerate(chunks):
        if idx == 0:
            merged.append(chunk)
            continue
        prev_tail = chunks[idx - 1][-overlap:]
        merged.append(f"{prev_tail}\n{chunk}")
    return merged


print("\n二、文本切片")
all_chunks: list[dict] = []
for f in md_files:
    try:
        text = f.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        continue
    if len(text) < 20:
        continue

    source_rel = str(f.relative_to(COURSE_ROOT))
    title = f.stem
    chunk_texts = split_text(text)
    for pos, chunk_text in enumerate(chunk_texts, start=1):
        all_chunks.append({
            "title": title,
            "source": source_rel,
            "chunk_index": pos,
            "content": chunk_text,
            "content_length": len(chunk_text),
        })

print(f"    共切出 {len(all_chunks)} 个文本片段 (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

if len(all_chunks) < 5:
    raise SystemExit("切片数量过少，请检查 Markdown 文件内容。")

print("    前 5 条摘要：")
for i, chunk in enumerate(all_chunks[:5], 1):
    preview = chunk["content"].replace("\n", " ")[:80]
    print(f"      [{i}] {chunk['source']} | 长度={chunk['content_length']} | {preview}...")


# ========== 3. 中文分词（为 BM25 准备）- 多进程版本 ==========

print("\n三、中文分词（为 BM25 索引准备）- 多进程加速")

def tokenize(text: str) -> list[str]:
    """使用 jieba 进行中文分词，过滤短词。"""
    return [w.strip() for w in jieba.cut(text) if len(w.strip()) >= 2]


def tokenize_chunk(chunk_data: tuple[int, dict]) -> tuple[int, list[str]]:
    """对单个 chunk 进行分词（用于多进程）。"""
    idx, chunk = chunk_data
    tokens = tokenize(chunk["content"])
    return idx, tokens


start = time.perf_counter()
chunk_items = list(enumerate(all_chunks))

if NUM_WORKERS > 1 and len(chunk_items) > 10:
    # 多进程分词
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(tokenize_chunk, chunk_items, chunksize=max(1, len(chunk_items) // (NUM_WORKERS * 4)))
    # 按原始顺序整理结果
    results.sort(key=lambda x: x[0])
    for idx, tokens in results:
        all_chunks[idx]["tokens"] = tokens
else:
    # 单进程分词（数据量小时）
    for idx, chunk in chunk_items:
        chunk["tokens"] = tokenize(chunk["content"])

tokenize_ms = (time.perf_counter() - start) * 1000
print(f"    分词完成，耗时 {tokenize_ms:.0f} ms ({NUM_WORKERS} 进程)")
print(f"    示例：'{all_chunks[0]['content'][:30]}...' → {' / '.join(all_chunks[0]['tokens'][:8])}")


# ========== 4. 构建 BM25 倒排索引 ==========

print("\n四、构建 BM25 倒排索引")

N = len(all_chunks)
avgdl = sum(len(chunk["tokens"]) for chunk in all_chunks) / N

# 计算 IDF
df_counter = defaultdict(int)
for chunk in all_chunks:
    unique_tokens = set(chunk["tokens"])
    for token in unique_tokens:
        df_counter[token] += 1

idf_cache = {}
for term, df in df_counter.items():
    idf_cache[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

# 构建倒排索引：term -> list of (chunk_id, tf)
inverted_index = defaultdict(list)
for chunk_id, chunk in enumerate(all_chunks):
    tf_counter = defaultdict(int)
    for token in chunk["tokens"]:
        tf_counter[token] += 1
    for term, tf in tf_counter.items():
        inverted_index[term].append((chunk_id, tf))

print(f"    文档总数 N = {N}")
print(f"    平均文档长度 avgdl = {avgdl:.1f} 词")
print(f"    词汇表大小 = {len(idf_cache)}")
print(f"    倒排索引条目 = {len(inverted_index)}")


# ========== 5. 加载 BGE-m3 并编码（多进程分批） ==========

print(f"\n五、加载 BGE-m3 模型")
print(f"    模型路径: {MODEL_DIR.relative_to(COURSE_ROOT)}")
start = time.perf_counter()
model = SentenceTransformer(str(MODEL_DIR), device="cpu")
load_ms = (time.perf_counter() - start) * 1000
print(f"    加载耗时: {load_ms:.0f} ms")

texts = [chunk["content"] for chunk in all_chunks]
print(f"\n六、生成向量 ({len(texts)} 条) - 多进程分批编码")

# 计算最优 batch_size
base_batch = 4
optimal_batch = base_batch * NUM_WORKERS

start = time.perf_counter()
vectors = model.encode(
    texts, 
    batch_size=optimal_batch, 
    normalize_embeddings=True, 
    show_progress_bar=True,
    num_workers=NUM_WORKERS  # sentence-transformers 内部多进程支持
)
vectors = np.asarray(vectors, dtype="float32")
encode_ms = (time.perf_counter() - start) * 1000

dim = vectors.shape[1]
print(f"    向量矩阵: {vectors.shape}")
print(f"    向量维度: {dim}")
print(f"    编码耗时: {encode_ms:.0f} ms (batch_size={optimal_batch}, workers={NUM_WORKERS})")


# ========== 6. 写入 FAISS 与元数据 ==========

print(f"\n七、写入混合数据库")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# FAISS 向量索引
index = faiss.IndexFlatIP(dim)
index.add(vectors)
faiss.write_index(index, str(FAISS_PATH))
np.save(NPY_PATH, vectors)

print(f"    FAISS 索引: {FAISS_PATH.relative_to(COURSE_ROOT)} ({index.ntotal} 条)")
print(f"    向量矩阵:   {NPY_PATH.relative_to(COURSE_ROOT)} ({vectors.shape})")

# 元数据（不含 tokens，tokens 太大）
with JSONL_PATH.open("w", encoding="utf-8") as f:
    for row_id, chunk in enumerate(all_chunks):
        meta = {
            "row_id": row_id,
            "title": chunk["title"],
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "content": chunk["content"],
            "content_length": chunk["content_length"],
        }
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
print(f"    元数据:     {JSONL_PATH.relative_to(COURSE_ROOT)} ({len(all_chunks)} 条)")

# BM25 倒排索引（pickle）
bm25_data = {
    "inverted_index": dict(inverted_index),
    "idf_cache": idf_cache,
    "N": N,
    "avgdl": avgdl,
    "k1": 1.5,
    "b": 0.75,
}
with BM25_PATH.open("wb") as f:
    pickle.dump(bm25_data, f)
print(f"    BM25 索引:  {BM25_PATH.relative_to(COURSE_ROOT)} (词汇表 {len(idf_cache)} 词)")

# 索引说明
meta = {
    "version": "1.0",
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "embedding_model": "bge-m3",
    "model_path": str(MODEL_DIR.relative_to(COURSE_ROOT)),
    "vector_dim": int(dim),
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "normalize_embeddings": True,
    "similarity": "inner_product（向量已归一化，等价于 cosine）",
    "total_chunks": len(all_chunks),
    "source_files": len(md_files),
    "bm25_params": {
        "k1": 1.5,
        "b": 0.75,
        "avgdl": float(avgdl),
        "N": N,
    },
    "files": {
        "faiss_index": FAISS_PATH.name,
        "vector_matrix": NPY_PATH.name,
        "metadata": JSONL_PATH.name,
        "bm25_index": BM25_PATH.name,
    },
}
META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"    索引说明:   {META_PATH.relative_to(COURSE_ROOT)}")

print(f"\n八、完成")
print(f"    共 {len(md_files)} 个 Markdown 文件 → {len(all_chunks)} 个文本片段")
print(f"    向量维度: {dim}，BM25 词汇表: {len(idf_cache)} 词")
print(f"    输出目录: {OUTPUT_DIR.relative_to(COURSE_ROOT)}/")
print(f"    下一步: 使用 hybrid_bot.py 进行混合检索问答")