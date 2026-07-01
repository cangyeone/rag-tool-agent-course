#!/bin/bash
# macOS / Linux 启动脚本 - 新版界面（多会话支持）
# 第一次运行会先构建索引，再启动 Streamlit 页面。

set -e

echo "========================================"
echo " RAG 问答机器人 - 新版界面"
echo "========================================"
echo ""

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "❌ 未找到 Python，请先安装 Python 3.8+"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖..."
python -c "import streamlit, numpy, faiss, sentence_transformers, requests, rich" 2>/dev/null || {
    echo "⚠️  缺少依赖，正在安装..."
    pip install streamlit numpy faiss-cpu sentence-transformers requests rich jieba
}

# 检查索引是否存在
if [ ! -f "storage/chunks.json" ] || [ ! -f "storage/embeddings.npy" ]; then
    echo ""
    echo "📚 检测到未构建索引，开始构建..."
    echo ""
    python build_index.py
    if [ $? -ne 0 ]; then
        echo "❌ 索引构建失败"
        exit 1
    fi
    echo ""
    echo "✅ 索引构建完成"
    echo ""
fi

# 启动 Streamlit 应用
echo "========================================"
echo "🌐 启动 Web 服务..."
echo "========================================"
echo ""
echo "📍 访问地址: http://localhost:8501"
echo "💡 按 Ctrl+C 停止服务"
echo ""

streamlit run app_new.py --server.address 0.0.0.0 --server.port 8501