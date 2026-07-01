# start windows：代码说明

> 对应代码：`RAG_问答机器人/start_windows.ps1`

## 1. 这段代码对应的行业需求

业务知识库问答：围绕订单服务规则、安全制度、应急预案、巡检记录和培训材料做可追溯检索。

## 2. 课堂目标

Windows PowerShell 启动脚本

## 3. 讲解顺序

- 先用业务问题开场：业务现场为什么需要这段能力。
- 再讲输入是什么：用户问题、规则资料、图片/记录、工具参数或环境信息。
- 然后讲处理过程：每一步生成什么中间结果。
- 最后讲输出怎么看：哪些结果可信，哪些地方要人工确认或回到官方系统。

## 4. 代码结构

1. 先写参数区，例如 `param([string]$Action = "Status")`。
2. 写当前脚本目录变量，避免使用固定盘符或个人电脑路径。
3. 把安装、启动、停止、状态、日志等动作拆成函数。
4. 每一步都用 `Write-Host` 打印中文说明。
5. 最后用 `switch ($Action)` 分发动作。

## 5. 主要函数或代码块

- 这个脚本没有明显函数拆分，可以按代码段讲解。

## 6. 可修改点

- 把示例问题换成在线客服、安全检查、设备运维或培训答疑中的真实表达。
- 改一条模拟规则或模拟工具返回值，观察最终输出怎么变化。
- 故意制造一个缺字段、空结果或接口不可用的情况，说明兜底逻辑。
- 补充一条测试用例，检查输出是否仍然符合业务边界。

## 7. 运行方式

在脚本所在目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\start_windows.ps1 -Action Status
```

如果是安装脚本，可以把 `Status` 换成 `Install`、`Start`、`Stop`、`Logs` 等动作。

涉及 DeepSeek 时，只使用环境变量 `DEEPSEEK_API_KEY`，不要把 Key 写入文件。

## 8. 完整参考代码

```powershell
$ErrorActionPreference = "Stop"

# Windows PowerShell 启动脚本。
# 第一次运行会先构建索引，再启动 Streamlit 页面。

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if (-not $env:KMP_DUPLICATE_LIB_OK) {
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
}

if (-not $env:BGE_M3_MODEL_PATH) {
    $env:BGE_M3_MODEL_PATH = ".\models\bge-m3"
}

if ((-not (Test-Path ".\storage\chunks.json")) -or (-not (Test-Path ".\storage\embeddings.npy"))) {
    Write-Host "[RAG] 未检测到索引，开始构建。" -ForegroundColor Cyan
    python .\build_index.py
}

Write-Host "[RAG] 启动问答机器人：http://localhost:8501" -ForegroundColor Cyan
streamlit run .\app.py --server.address 0.0.0.0 --server.port 8501
```