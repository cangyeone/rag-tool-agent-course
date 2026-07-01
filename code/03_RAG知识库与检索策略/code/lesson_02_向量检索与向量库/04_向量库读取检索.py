"""04_向量库读取检索。

本脚本读取 03_向量库写入.py 生成的真实向量库：
1. 加载 FAISS 索引
2. 加载 JSONL 元数据
3. 用同一个 BGE-m3 模型把问题向量化
4. 执行 Top-K 向量检索
5. 打印分数、来源和命中文本

运行方式：
    cd rag-tool-agent-course
    python code/03_RAG知识库与检索策略/code/lesson_02_向量检索与向量库/04_向量库读取检索.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import faiss
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
BUILD_SCRIPT = SCRIPT_DIR / "03_向量库写入.py"

print("=" * 72)
print("04_向量库读取检索 —— BGE-m3 + FAISS Top-K")
print("=" * 72)

if not MODEL_DIR.is_dir():
    raise SystemExit(f"未找到本地 BGE-m3 模型：{MODEL_DIR.relative_to(COURSE_ROOT)}")

if not (FAISS_PATH.exists() and JSONL_PATH.exists() and META_PATH.exists()):
    print("\n未找到真实 BGE 向量库，先自动运行 03_向量库写入.py。")
    subprocess.run([sys.executable, str(BUILD_SCRIPT)], cwd=str(COURSE_ROOT), check=True)

print("\n一、加载向量库文件")
meta = json.loads(META_PATH.read_text(encoding="utf-8"))
print(json.dumps(meta, ensure_ascii=False, indent=2))

start = time.perf_counter()
index = faiss.read_index(str(FAISS_PATH))
load_index_ms = (time.perf_counter() - start) * 1000

metadata = []
for line in JSONL_PATH.read_text(encoding="utf-8").splitlines():
    if line.strip():
        metadata.append(json.loads(line))

print(f"FAISS 加载耗时：{load_index_ms:.2f} ms")
print(f"索引向量数：{index.ntotal}")
print(f"元数据条数：{len(metadata)}")

print("\n二、加载 BGE-m3 查询模型")
start = time.perf_counter()
model = SentenceTransformer(str(MODEL_DIR), device="cpu")
load_model_ms = (time.perf_counter() - start) * 1000
print(f"模型加载耗时：{load_model_ms:.1f} ms")


def search(query: str, top_k: int = 4) -> list[tuple[float, dict]]:
    """把查询文本编码为 BGE-m3 向量，然后交给 FAISS 检索。"""
    query_vec = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_vec = np.asarray(query_vec, dtype="float32")

    scores, row_ids = index.search(query_vec, top_k)

    results = []
    for score, row_id in zip(scores[0], row_ids[0]):
        if row_id < 0:
            continue
        row = metadata[int(row_id)]
        results.append((float(score), row))
    return results


questions = [
    "候补申请一定能成功吗？",
    "退款手续费和变更要注意什么？",
    "这个课程里 RAG 的流程是什么？",
    "学生优惠票需要满足什么条件？",
    "地质雷达图像能不能直接让模型判断？",
]

print("\n三、执行真实向量检索")
for question in questions:
    start = time.perf_counter()
    results = search(question, top_k=4)
    search_ms = (time.perf_counter() - start) * 1000

    print("\n" + "-" * 60)
    print("查询：", question)
    print(f"检索耗时：{search_ms:.2f} ms")

    for rank, (score, row) in enumerate(results, 1):
        preview = row["content"].replace("\n", " ")[:120]
        print(f"#{rank} score={score:.4f} | {row['title']} | {row['source']}")
        print("   ", preview)

print("\n四、和简单哈希向量的区别")
print("1. 哈希向量只根据字符或词落在哪个桶里，不能真正理解同义表达。")
print("2. BGE-m3 是训练出来的 embedding 模型，可以把语义相近的文本放得更近。")
print("3. FAISS 负责快速找最近向量；RAG 系统再把命中的原文交给大模型生成回答。")
print("4. 向量库不是只保存 numpy 数组，还要有索引、元数据、来源和版本信息。")