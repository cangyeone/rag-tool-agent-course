# Windows PowerShell 启动脚本 - 新版界面（多会话支持）
# 第一次运行会先构建索引，再启动 Streamlit 页面。

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " RAG 问答机器人 - 新版界面" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 切换到脚本所在目录
Set-Location $PSScriptRoot

# 检查 Python 环境
try {
    python --version | Out-Null
} catch {
    Write-Host "❌ 未找到 Python，请先安装 Python 3.8+" -ForegroundColor Red
    exit 1
}

# 检查依赖
Write-Host "📦 检查依赖..." -ForegroundColor Yellow
try {
    python -c "import streamlit, numpy, faiss, sentence_transformers, requests, rich" 2>$null
} catch {
    Write-Host "️  缺少依赖，正在安装..." -ForegroundColor Yellow
    pip install streamlit numpy faiss-cpu sentence-transformers requests rich jieba
}

# 检查索引是否存在
if (-not (Test-Path "storage\chunks.json") -or -not (Test-Path "storage\embeddings.npy")) {
    Write-Host ""
    Write-Host "📚 检测到未构建索引，开始构建..." -ForegroundColor Yellow
    Write-Host ""
    python build_index.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ 索引构建失败" -ForegroundColor Red
        exit 1
    }
    Write-Host ""
    Write-Host "✅ 索引构建完成" -ForegroundColor Green
    Write-Host ""
}

# 启动 Streamlit 应用
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🌐 启动 Web 服务..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📍 访问地址: http://localhost:8501" -ForegroundColor Green
Write-Host "💡 按 Ctrl+C 停止服务" -ForegroundColor Gray
Write-Host ""

streamlit run .\app_new.py --server.address 0.0.0.0 --server.port 8501