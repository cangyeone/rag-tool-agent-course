"""vectorstore —— FAISS 向量数据库 + SQLite 元数据管理。

提供函数：
    build_vectorstore(...) -> dict
        遍历 knowledge 目录，解析文档 → 切片 → BGE-m3 编码 → FAISS 索引 + SQLite 元数据。

    load_vectorstore(db_dir: Path, model_dir: Path) -> dict
        加载已有的向量库，返回 index, metadata, model 等对象。

设计要点：
    - 使用 SQLite 存储文档和切片元数据，支持增量更新。
    - 通过文件 MD5 哈希检测新增/修改的文件。
    - FAISS IndexFlatIP 存储向量（内积 = 余弦相似度，因 BGE-m3 输出已归一化）。
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import faiss
import numpy as np

from .parser import collect_and_chunk

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 500


def _init_sqlite(db_path: Path) -> sqlite3.Connection:
    """初始化 SQLite 数据库，创建表结构。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source      TEXT    NOT NULL UNIQUE,
            file_hash   TEXT    NOT NULL,
            file_ext    TEXT    NOT NULL,
            title       TEXT    NOT NULL,
            updated_at  TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id     INTEGER NOT NULL REFERENCES documents(id),
            chunk_index     INTEGER NOT NULL,
            content         TEXT    NOT NULL,
            content_length  INTEGER NOT NULL,
            page            INTEGER DEFAULT 1,
            vector_rowid    INTEGER
        );
        CREATE TABLE IF NOT EXISTS index_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    conn.commit()
    return conn


def _get_stored_file_hashes(conn: sqlite3.Connection) -> dict[str, str]:
    """从 SQLite 获取已入库文件的 source → file_hash 映射。"""
    rows = conn.execute("SELECT source, file_hash FROM documents").fetchall()
    return {row[0]: row[1] for row in rows}


def _clear_old_chunks_for_file(conn: sqlite3.Connection, doc_id: int):
    """删除指定文档的旧切片记录。"""
    conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
    conn.commit()


def _upsert_document(conn: sqlite3.Connection, source: str, file_hash: str,
                     file_ext: str, title: str) -> int:
    """插入或更新文档记录，返回 document_id。"""
    conn.execute(
        """INSERT INTO documents (source, file_hash, file_ext, title, updated_at)
           VALUES (?, ?, ?, ?, datetime('now'))
           ON CONFLICT(source) DO UPDATE SET
               file_hash = excluded.file_hash,
               file_ext  = excluded.file_ext,
               title     = excluded.title,
               updated_at = datetime('now')""",
        (source, file_hash, file_ext, title),
    )
    conn.commit()
    row = conn.execute("SELECT id FROM documents WHERE source = ?", (source,)).fetchone()
    return row[0] if row else 0


def _insert_chunks(conn: sqlite3.Connection, document_id: int,
                   chunks: list[dict], start_rowid: int):
    """批量插入切片记录。"""
    rows = [
        (document_id, c["chunk_index"], c["content"], c["content_length"],
         c.get("page", 1), start_rowid + i)
        for i, c in enumerate(chunks)
    ]
    conn.executemany(
        "INSERT INTO chunks (document_id, chunk_index, content, content_length, page, vector_rowid) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def _save_index_meta(conn: sqlite3.Connection, meta: dict):
    """将索引元信息写入 index_meta 表。"""
    for key, value in meta.items():
        conn.execute(
            "INSERT INTO index_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )
    conn.commit()


def _save_faiss_and_jsonl(index: faiss.IndexFlatIP, vectors: np.ndarray,
                          all_chunks: list[dict], db_dir: Path):
    """保存 FAISS 索引、向量矩阵 npy、JSONL 元数据。"""
    db_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = db_dir / "faiss.index"
    npy_path = db_dir / "vectors.npy"
    jsonl_path = db_dir / "metadata.jsonl"
    meta_path = db_dir / "index_meta.json"

    faiss.write_index(index, str(faiss_path))
    np.save(str(npy_path), vectors)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    meta = {
        "version": "1.0",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vector_dim": int(vectors.shape[1]),
        "total_chunks": len(all_chunks),
        "similarity": "inner_product (cosine equivalent)",
        "files": {
            "faiss_index": faiss_path.name,
            "vector_matrix": npy_path.name,
            "metadata": jsonl_path.name,
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def build_vectorstore(
    knowledge_dir: Path,
    db_dir: Path,
    model_dir: Path,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    force_rebuild: bool = False,
) -> dict:
    """构建/更新向量数据库。

    流程：
        1. 遍历 knowledge_dir 下所有 .pdf/.md 文件
        2. 通过 MD5 哈希对比 SQLite 中的记录，识别新增/修改的文件
        3. 仅解析和处理变更的文件（增量更新）
        4. 切片 → BGE-m3 编码 → FAISS 索引
        5. 元数据写入 SQLite + JSONL

    Args:
        knowledge_dir: 知识源文件目录（如 rag-tool-agent-course/knowledge/）。
        db_dir: 向量库输出目录（如 rag-tool-agent-course/kb_data/）。
        model_dir: BGE-m3 本地模型目录。
        chunk_size: 切片最大字符数，默认 2000。
        chunk_overlap: 切片重叠字符数，默认 500。
        force_rebuild: 是否强制全量重建（忽略增量检测）。

    Returns:
        dict: 包含构建统计信息。
            {
                "status": "ok" | "no_changes" | "error",
                "total_files": int,        # 扫描到的文件总数
                "new_files": int,          # 新增/修改的文件数
                "skipped_files": int,      # 跳过的文件数
                "total_chunks": int,       # 总切片数
                "vector_dim": int,         # 向量维度
                "build_time_ms": float,    # 构建耗时（毫秒）
            }
    """
    start_time = time.perf_counter()

    if not model_dir.is_dir():
        return {"status": "error", "message": f"未找到 BGE-m3 模型: {model_dir}"}

    db_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = db_dir / "metadata.db"
    conn = _init_sqlite(sqlite_path)

    # --- 1. 收集文档并切片 ---
    all_chunks = collect_and_chunk(knowledge_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not all_chunks:
        conn.close()
        return {"status": "no_files", "total_files": 0, "message": "knowledge 目录中没有找到 .pdf 或 .md 文件"}

    # --- 2. MD5 增量检测：按文件分组，对比哈希 ---
    stored_hashes = _get_stored_file_hashes(conn)
    chunks_by_file: dict[str, dict] = {}
    for chunk in all_chunks:
        src = chunk["source"]
        if src not in chunks_by_file:
            chunks_by_file[src] = {
                "file_hash": chunk["file_hash"],
                "file_ext": chunk.get("file_ext", ""),
                "title": chunk["title"],
                "chunks": [],
            }
        chunks_by_file[src]["chunks"].append(chunk)

    unchanged_files = 0
    total_files = len(chunks_by_file)
    has_changes = force_rebuild

    for source, info in chunks_by_file.items():
        current_hash = info["file_hash"]
        if source in stored_hashes and stored_hashes[source] == current_hash:
            unchanged_files += 1
        else:
            has_changes = True

    if not has_changes:
        conn.close()
        return {
            "status": "no_changes",
            "total_files": total_files,
            "message": "所有文件均未变更，跳过构建",
        }

    changed_files = total_files - unchanged_files

    # --- 3. 变更检测到，执行全量重建 ---
    # 清空 SQLite 旧数据
    conn.execute("DELETE FROM chunks")
    conn.execute("DELETE FROM documents")
    conn.execute("DELETE FROM index_meta")
    conn.commit()

    # --- 4. 加载 BGE-m3 模型并编码所有切片 ---
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(model_dir), device="cpu")

    texts = [c["content"] for c in all_chunks]
    vectors = model.encode(texts, batch_size=4, normalize_embeddings=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")
    dim = int(vectors.shape[1])

    # --- 5. 构建全新 FAISS 索引 ---
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # 为每个切片分配 vector_rowid
    for i, chunk in enumerate(all_chunks):
        chunk["vector_rowid"] = i

    # --- 6. 持久化 FAISS + JSONL ---
    _save_faiss_and_jsonl(index, vectors, all_chunks, db_dir)

    # --- 7. 写入 SQLite 元数据 ---
    for source, info in chunks_by_file.items():
        doc_id = _upsert_document(conn, source, info["file_hash"], info["file_ext"], info["title"])
        _insert_chunks(conn, doc_id, info["chunks"], 0)

    _save_index_meta(conn, {
        "version": "1.0",
        "vector_dim": dim,
        "total_chunks": len(all_chunks),
        "total_documents": total_files,
    })

    build_time_ms = (time.perf_counter() - start_time) * 1000
    conn.close()

    return {
        "status": "ok",
        "total_files": total_files,
        "new_files": changed_files,
        "skipped_files": unchanged_files,
        "total_chunks": len(all_chunks),
        "vector_dim": dim,
        "build_time_ms": build_time_ms,
    }


def load_vectorstore(db_dir: Path, model_dir: Path) -> dict | None:
    """加载已有的向量库。

    返回值包含：
        - index: FAISS 索引对象
        - metadata: JSONL 元数据列表
        - model: SentenceTransformer 模型对象
        - total_chunks: 切片总数

    Args:
        db_dir: 向量库目录（包含 faiss.index / metadata.jsonl）。
        model_dir: BGE-m3 本地模型目录。

    Returns:
        dict 或 None（如果向量库不存在）。
    """
    faiss_path = db_dir / "faiss.index"
    jsonl_path = db_dir / "metadata.jsonl"

    if not faiss_path.exists() or not jsonl_path.exists():
        return None

    from sentence_transformers import SentenceTransformer

    index = faiss.read_index(str(faiss_path))
    model = SentenceTransformer(str(model_dir), device="cpu")

    metadata: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metadata.append(json.loads(line))

    return {
        "index": index,
        "metadata": metadata,
        "model": model,
        "total_chunks": len(metadata),
    }


def get_kb_stats(db_dir: Path) -> dict:
    """获取知识库统计信息。

    Args:
        db_dir: 向量库目录。

    Returns:
        包含 files_count, chunks_count, last_build 等字段的字典。
    """
    sqlite_path = db_dir / "metadata.db"
    stats: dict = {
        "has_vectorstore": (db_dir / "faiss.index").exists(),
        "has_metadata": (db_dir / "metadata.jsonl").exists(),
        "file_count": 0,
        "chunk_count": 0,
        "last_build": None,
    }

    if sqlite_path.exists():
        conn = sqlite3.connect(str(sqlite_path))
        stats["file_count"] = conn.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]
        stats["chunk_count"] = conn.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]
        row = conn.execute(
            "SELECT value FROM index_meta WHERE key = 'version'"
        ).fetchone()
        stats["version"] = row[0] if row else None
        conn.close()

    return stats