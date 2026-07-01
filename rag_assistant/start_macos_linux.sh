#!/usr/bin/env bash
set -euo pipefail

# macOS / Linux 启动脚本。
# 第一次运行会先构建索引，再启动 Streamlit 页面。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
if [[ -z "${BGE_M3_MODEL_PATH:-}" ]]; then
  if [[ -d "../open_models/bge-m3" ]]; then
    export BGE_M3_MODEL_PATH="../open_models/bge-m3"
  else
    export BGE_M3_MODEL_PATH="./models/bge-m3"
  fi
fi

if [[ ! -f "storage/chunks.json" || ! -f "storage/embeddings.npy" ]]; then
  echo "[RAG] 未检测到索引，开始构建。"
  python build_index.py
fi

echo "[RAG] 启动问答机器人：http://localhost:8501"
streamlit run app.py --server.address 0.0.0.0 --server.port 8501