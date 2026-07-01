$ErrorActionPreference = "Stop"

# Windows PowerShell 启动脚本。
# 第一次运行会先构建索引，再启动 Streamlit 页面。

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if (-not $env:KMP_DUPLICATE_LIB_OK) {
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
}

if (-not $env:OMP_NUM_THREADS) {
    $env:OMP_NUM_THREADS = "1"
}

if (-not $env:MKL_NUM_THREADS) {
    $env:MKL_NUM_THREADS = "1"
}

if (-not $env:BGE_M3_MODEL_PATH) {
    if (Test-Path "..\open_models\bge-m3") {
        $env:BGE_M3_MODEL_PATH = "..\open_models\bge-m3"
    } else {
        $env:BGE_M3_MODEL_PATH = ".\models\bge-m3"
    }
}

if ((-not (Test-Path ".\storage\chunks.json")) -or (-not (Test-Path ".\storage\embeddings.npy"))) {
    Write-Host "[RAG] 未检测到索引，开始构建。" -ForegroundColor Cyan
    python .\build_index.py
}

Write-Host "[RAG] 启动问答机器人：http://localhost:8501" -ForegroundColor Cyan
streamlit run .\app.py --server.address 0.0.0.0 --server.port 8501