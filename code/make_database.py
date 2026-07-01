"""make_database.py —— 遍历 code 下所有 Markdown 文档，构建 FAISS 向量数据库。

运行方式：
    cd rag-tool-agent-course
    python code/make_database.py

参数说明：
    切片长度 2000 字符，overlap 500 字符。
    输出目录：code/make_database_output/
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

COURSE_ROOT = Path(__file__).resolve().parent.parent
if not (COURSE_ROOT / "code").is_dir():
    COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

CODE_ROOT = COURSE_ROOT / "code"
MODEL_DIR = COURSE_ROOT / "open_models" / "bge-m3"
OUTPUT_DIR = CODE_ROOT / "make_database_output"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 500

FAISS_PATH = OUTPUT_DIR / "faiss.index"
NPY_PATH = OUTPUT_DIR / "vectors.npy"
JSONL_PATH = OUTPUT_DIR / "metadata.jsonl"
META_PATH = OUTPUT_DIR / "index_meta.json"

print("=" * 72)
print("make_database —— 遍历 Markdown → 切片 → BGE-m3 → FAISS")
print("=" * 72)

if not MODEL_DIR.is_dir():
    raise SystemExit(f"未找到本地 BGE-m3 模型：{MODEL_DIR}")


# ========== 1. 遍历 Markdown 文档 ==========

def collect_markdown_files(root: Path) -> list[Path]:
    """递归收集 root 下所有 .md 文件，跳过隐藏目录和输出目录。"""
    skip_parts = {"__pycache__", ".venv", "rag_kb", "make_database_output"}
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


# ========== 3. 加载 BGE-m3 并编码 ==========

print(f"\n三、加载 BGE-m3 模型")
print(f"    模型路径: {MODEL_DIR.relative_to(COURSE_ROOT)}")
start = time.perf_counter()
model = SentenceTransformer(str(MODEL_DIR), device="cpu")
load_ms = (time.perf_counter() - start) * 1000
print(f"    加载耗时: {load_ms:.0f} ms")

texts = [chunk["content"] for chunk in all_chunks]
print(f"\n四、生成向量 ({len(texts)} 条)")
start = time.perf_counter()
vectors = model.encode(texts, batch_size=4, normalize_embeddings=True, show_progress_bar=True)
vectors = np.asarray(vectors, dtype="float32")
encode_ms = (time.perf_counter() - start) * 1000

dim = vectors.shape[1]
print(f"    向量矩阵: {vectors.shape}")
print(f"    向量维度: {dim}")
print(f"    编码耗时: {encode_ms:.0f} ms")


# ========== 4. 写入 FAISS 与元数据 ==========

print(f"\n五、写入 FAISS 向量数据库")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

index = faiss.IndexFlatIP(dim)
index.add(vectors)
faiss.write_index(index, str(FAISS_PATH))
np.save(NPY_PATH, vectors)

print(f"    FAISS 索引: {FAISS_PATH.relative_to(COURSE_ROOT)} ({index.ntotal} 条)")
print(f"    向量矩阵:   {NPY_PATH.relative_to(COURSE_ROOT)} ({vectors.shape})")

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
    "files": {
        "faiss_index": FAISS_PATH.name,
        "vector_matrix": NPY_PATH.name,
        "metadata": JSONL_PATH.name,
    },
}
META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"    索引说明:   {META_PATH.relative_to(COURSE_ROOT)}")

print(f"\n六、完成")
print(f"    共 {len(md_files)} 个 Markdown 文件 → {len(all_chunks)} 个文本片段 → {dim} 维向量")
print(f"    输出目录: {OUTPUT_DIR.relative_to(COURSE_ROOT)}/")
print(f"    下一步: 使用 FAISS 索引进行向量检索")