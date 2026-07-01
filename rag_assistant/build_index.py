from __future__ import annotations

import argparse
import os
from pathlib import Path

from rag_core import DEFAULT_MODEL_PATH, build_and_save_index


def main() -> None:
    # 解决部分 macOS/conda 环境里 OpenMP 重复加载导致的崩溃。
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    project_dir = Path(__file__).resolve().parent
    default_source = project_dir.parent
    default_storage = project_dir / "storage"

    parser = argparse.ArgumentParser(description="为 RAG 问答机器人构建本地知识索引")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=default_source,
        help="知识库来源目录，默认读取整个 rag-tool-agent-course 课程目录。",
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=default_storage,
        help="索引保存目录。",
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv("BGE_M3_MODEL_PATH", DEFAULT_MODEL_PATH),
        help="BGE-m3 本地模型路径。",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=650,
        help="每个文本片段的大致字符数。",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=120,
        help="相邻片段重叠字符数。",
    )
    args = parser.parse_args()

    print("开始构建 RAG 索引")
    print(f"知识目录：{args.source_dir}")
    print(f"索引目录：{args.storage_dir}")
    print(f"BGE 模型：{args.model_path}")
    print(f"切片参数：chunk_size={args.chunk_size}, overlap={args.overlap}")
    print("-" * 70)

    build_and_save_index(
        source_dir=args.source_dir,
        storage_dir=args.storage_dir,
        model_path=args.model_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()