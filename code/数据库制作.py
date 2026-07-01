"""数据库制作 —— 遍历 code 下所有 Markdown 文档，切片后写入 FAISS 向量库。

功能：
1. 递归遍历 code 目录下所有 .md 文件
2. 对每个文档按 2000 字符切片，相邻切片间保留 500 字符 overlap
3. 使用 BGE-m3 模型编码为向量
4. 写入 FAISS 索引 + JSONL 元数据（含文档来源路径）

运行方式：
    cd rag-tool-agent-course
    python code/数据库制作.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── 路径配置 ─────────────────────────────────────────────────────────────────
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

CODE_DIR = COURSE_ROOT / "code"
MODEL_DIR = COURSE_ROOT / "open_models" / "bge-m3"

# 输出目录
VECTOR_DIR = CODE_DIR / "faiss_database"
FAISS_PATH = VECTOR_DIR / "documents.index"
NPY_PATH = VECTOR_DIR / "document_vectors.npy"
JSONL_PATH = VECTOR_DIR / "metadata.jsonl"
META_PATH = VECTOR_DIR / "index_meta.json"

# ── 切片参数 ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 500


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """将文本按指定长度切片，相邻切片之间保留 overlap 长度的重叠。"""
    if not text.strip():
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def collect_markdown_files(root: Path) -> list[Path]:
    """递归收集 root 目录下所有 .md 文件。"""
    md_files = sorted(root.rglob("*.md"))
    return md_files


def build_documents(md_files: list[Path]) -> list[dict]:
    """读取所有 Markdown 文件并切片，返回文档列表。"""
    docs: list[dict] = []
    for file_path in md_files:
        rel_path = str(file_path.relative_to(COURSE_ROOT))
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception as e:
            print(f"  [跳过] 读取失败: {rel_path} ({e})")
            continue

        if not text:
            continue

        chunks = chunk_text(text)
        for chunk_idx, chunk in enumerate(chunks):
            docs.append({
                "id": f"doc_{len(docs) + 1:05d}",
                "source_file": rel_path,
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
                "content": chunk,
            })

    return docs


# ── 主流程 ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("数据库制作 —— 遍历 Markdown + BGE-m3 + FAISS")
    print("=" * 72)
    print(f"扫描目录：{CODE_DIR}")
    print(f"切片参数：chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")

    # 1. 收集 Markdown 文件
    print("\n一、收集 Markdown 文件")
    md_files = collect_markdown_files(CODE_DIR)
    print(f"共找到 {len(md_files)} 个 Markdown 文件")
    if not md_files:
        raise SystemExit("未找到任何 .md 文件，请检查目录。")

    # 2. 切片
    print("\n二、文档切片")
    docs = build_documents(md_files)
    if not docs:
        raise SystemExit("切片后没有任何有效内容。")
    print(f"共切分为 {len(docs)} 个文本块")

    # 打印前 5 条预览
    for i, doc in enumerate(docs[:5], 1):
        preview = doc["content"][:60].replace("\n", " ")
        print(f"  {i:02d}. [{doc['source_file']}] chunk {doc['chunk_index']}/{doc['total_chunks']} | {preview}...")
    if len(docs) > 5:
        print(f"  ... 共 {len(docs)} 条")

    # 3. 加载模型
    print("\n三、加载 BGE-m3 模型")
    if not MODEL_DIR.is_dir():
        raise SystemExit(f"未找到本地 BGE-m3 模型：{MODEL_DIR}")
    print(f"模型路径：{MODEL_DIR}")
    start = time.perf_counter()
    model = SentenceTransformer(str(MODEL_DIR), device="cpu")
    print(f"加载耗时：{(time.perf_counter() - start) * 1000:.1f} ms")

    # 4. 编码向量
    print("\n四、生成文本向量")
    texts = [doc["content"] for doc in docs]
    start = time.perf_counter()
    vectors = model.encode(
        texts,
        batch_size=8,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    vectors = np.asarray(vectors, dtype="float32")
    print(f"向量矩阵形状：{vectors.shape}")
    print(f"编码耗时：{(time.perf_counter() - start) * 1000:.1f} ms")

    # 5. 写入 FAISS 索引
    print("\n五、写入 FAISS 索引")
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    dim = vectors.shape[1]

    # 向量已归一化，内积等价于余弦相似度
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, str(FAISS_PATH))
    np.save(NPY_PATH, vectors)

    print(f"FAISS 索引类型：IndexFlatIP")
    print(f"索引向量数：{index.ntotal}")
    print(f"向量维度：{dim}")
    print(f"索引文件：{FAISS_PATH}")
    print(f"向量矩阵：{NPY_PATH}")

    # 6. 写入元数据
    print("\n六、写入元数据")
    with JSONL_PATH.open("w", encoding="utf-8") as f:
        for row_id, doc in enumerate(docs):
            metadata = {
                "row_id": row_id,
                "id": doc["id"],
                "source_file": doc["source_file"],
                "chunk_index": doc["chunk_index"],
                "total_chunks": doc["total_chunks"],
                "content": doc["content"],
                "content_length": len(doc["content"]),
            }
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    # 7. 写入索引说明
    meta = {
        "version": "1.0",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_model": "bge-m3",
        "model_path": str(MODEL_DIR),
        "vector_dim": int(dim),
        "normalize_embeddings": True,
        "similarity": "inner_product（向量已归一化，等价于 cosine）",
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "total_markdown_files": len(md_files),
        "total_chunks": len(docs),
        "scan_directory": str(CODE_DIR),
        "faiss_index": FAISS_PATH.name,
        "vector_matrix": NPY_PATH.name,
        "metadata": JSONL_PATH.name,
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"元数据文件：{JSONL_PATH}")
    print(f"索引说明：{META_PATH}")

    # 8. 完成
    print("\n" + "=" * 72)
    print("数据库制作完成！")
    print(f"  Markdown 文件数：{len(md_files)}")
    print(f"  切片总数：{len(docs)}")
    print(f"  向量维度：{dim}")
    print(f"  输出目录：{VECTOR_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()