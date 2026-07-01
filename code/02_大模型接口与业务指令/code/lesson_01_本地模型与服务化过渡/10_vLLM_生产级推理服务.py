"""
学习目标：理解 vLLM 在生产级模型推理中的作用，掌握部署和调用方法。
vLLM 是当前最主流的高性能 LLM 推理引擎，支持 OpenAI 兼容 API 和 PagedAttention 显存优化。
"""


import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
import os
import subprocess
import sys
import time


print("10 vLLM——生产级模型推理服务")
print("=" * 72)
print()

# ═══════════════════════════════════════════════════
# 一、vLLM 是什么，为什么需要它
# ═══════════════════════════════════════════════════
print("── 一、vLLM 是什么？ ──")
print()
print("vLLM（Very Large Language Model）是 UC Berkeley 开源的高性能推理引擎。")
print()
print("核心能力：")
vllm_features = [
    ("PagedAttention", "像操作系统管理虚拟内存一样管理 KV Cache，显存利用率提升 2-4 倍"),
    ("连续批处理", "多个请求动态合并成一个 batch，GPU 利用率接近 100%"),
    ("OpenAI 兼容 API", "一行命令启动服务，用标准 /v1/chat/completions 接口调用"),
    ("量化支持", "支持 AWQ、GPTQ、FP8 等量化格式，小显存跑大模型"),
    ("张量并行", "自动把模型拆分到多张 GPU，无需手动处理分布式"),
    ("前缀缓存", "相同 system prompt 只计算一次，多轮对话场景大幅提速"),
]
for name, desc in vllm_features:
    print(f"   ■ {name}：{desc}")
print()

# ═══════════════════════════════════════════════════
# 二、vLLM vs 其他后端的定位
# ═══════════════════════════════════════════════════
print("── 二、vLLM 在五类后端中的定位 ──")
print()
print("回顾当前讲过的四类后端：")
print()
print("   1. transformers      → 学习原理，本地调试")
print("   2. Ollama            → 本地服务化，开箱即用")
print("   3. DeepSeek API      → 云端调用，零运维")
print("   4. OpenAI API        → 最强能力，最成熟生态")
print()
print("   5. vLLM              → 自建高性能推理服务，兼顾性能与隐私  ← 本节新增")
print()

# 对比表：在所有后端中加入 vLLM
print("   ── 五类后端全景对比 ──")
print()
vllm_comparison = {
    "定位":        "生产级自建推理服务",
    "部署方式":     "pip install vllm，一行命令启动",
    "模型格式":     "HuggingFace 模型直接加载，支持 AWQ/GPTQ 量化",
    "推理速度":     "极快（PagedAttention + 连续批处理，吞吐量比 transformers 高 10-30x）",
    "API 兼容性":   "完全兼容 OpenAI /v1/chat/completions 格式",
    "并发能力":     "高（原生支持多请求并发，GPU 持续满载）",
    "显存效率":     "极高（PagedAttention 近乎零浪费）",
    "隐私安全":     "最高（数据不出服务器，完全自控）",
    "运维成本":     "中（需 GPU 服务器，需监控和更新模型）",
    "适用场景":     "企业私有化部署、示例业务系统 内部系统、高并发生产环境",
    "不适合":       "无 GPU 环境、快速原型验证、个人学习",
}
for k, v in vllm_comparison.items():
    print(f"     {k}：{v}")
print()

# ═══════════════════════════════════════════════════
# 三、vLLM 部署实战（模拟 + 可运行命令）
# ═══════════════════════════════════════════════════
print("── 三、vLLM 部署步骤 ──")
print()

# 3.1 安装
print("【第 1 步：安装 vLLM】")
print()
install_cmds = [
    "# 基础安装（自动检测 CUDA 版本）",
    "pip install vllm",
    "",
    "# 指定 CUDA 版本（如果自动检测失败）",
    "pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124",
    "",
    "# 验证安装",
    "python -c 'import vllm; print(vllm.__version__)'",
]
for cmd in install_cmds:
    print(f"   {cmd}")
print()

# 3.2 启动服务
print("【第 2 步：启动 vLLM OpenAI 兼容服务】")
print()
server_cmds = [
    "# 最简启动（单 GPU，默认端口 8000）",
    "python -m vllm.entrypoints.openai.api_server \\",
    "    --model Qwen/Qwen2.5-7B-Instruct \\",
    "    --served-model-name qwen-7b",
    "",
    "# 多 GPU 张量并行",
    "python -m vllm.entrypoints.openai.api_server \\",
    "    --model Qwen/Qwen2.5-7B-Instruct \\",
    "    --tensor-parallel-size 2 \\",
    "    --gpu-memory-utilization 0.90",
    "",
    "# 量化模型（节省显存）",
    "python -m vllm.entrypoints.openai.api_server \\",
    "    --model Qwen/Qwen2.5-7B-Instruct-AWQ \\",
    "    --quantization awq",
]
for cmd in server_cmds:
    print(f"   {cmd}")
print()
print("   服务启动后，可以通过 http://localhost:8000/v1/chat/completions 调用。")
print("   接口格式与 DeepSeek 完全一致，换后端只需改 base_url 和 api_key。")
print()

# 3.3 检查服务是否就绪
print("【第 3 步：检查服务健康状态（课堂可用 curl 验证）】")
print()
print("   curl http://localhost:8000/health")
print("   # 返回空即表示就绪")
print()
print("   curl http://localhost:8000/v1/models")
print('   # 返回 {"object":"list","data":[{"id":"qwen-7b",...}]}')
print()

# ═══════════════════════════════════════════════════
# 四、Python 调用 vLLM（OpenAI 兼容接口）
# ═══════════════════════════════════════════════════
print("── 四、Python 调用 vLLM ──")
print()
print("   vLLM 的 API 与 DeepSeek 完全兼容，代码几乎无需修改：")
print()

# 调用示例代码
vllm_code = """
import requests

# vLLM 服务地址（本地或内网）
VLLM_URL = "http://localhost:8000/v1/chat/completions"

# 与 DeepSeek 调用几乎一样，只需改 URL 和 model 名称
response = requests.post(VLLM_URL, json={
    "model": "qwen-7b",
    "messages": [
        {"role": "system", "content": "你是 通用客服助手，回答要简洁，最终以官方页面为准。"},
        {"role": "user", "content": "候补申请能保证成功吗？"}
    ],
    "temperature": 0.2,
    "max_tokens": 200
})

data = response.json()
answer = data["choices"][0]["message"]["content"]
print("vLLM 回答：", answer)
"""
print(vllm_code.strip())

# ═══════════════════════════════════════════════════
# 五、vLLM + DeepSeek 混合架构（示例业务系统 场景）
# ═══════════════════════════════════════════════════
print()
print("── 五、示例业务系统 场景：vLLM + DeepSeek 混合部署 ──")
print()
print("   生产建议架构：")
print()
architecture = [
    ["入口层",     "Nginx / API Gateway",              "统一鉴权、限流、路由"],
    ["路由层",     "Router（参考 Ch05）",               "按问题复杂度分流"],
    ["推理层A",    "vLLM（本地 GPU 集群）",              "高并发常规问答，数据不出内网"],
    ["推理层B",    "DeepSeek API（云端）",               "复杂推理、长文本分析"],
    ["知识层",     "RAG 向量库（参考 Ch03）",            "示例业务系统 规章、FAQ 检索"],
    ["工具层",     "Function Calling（参考 Ch04）",      "服务点查询、订单编号查询、工单系统"],
]
for layer, tech, purpose in architecture:
    print(f"   {layer:<10} {tech:<35} {purpose}")
print()
print("   路由策略示例：")
print("   - 简单 FAQ / 退变更规则          → vLLM 本地处理（快、零费用）")
print("   - 多步骤推理 / 长文本分析        → DeepSeek API（模型更强）")
print("   - 敏感工单 / 内部运维            → vLLM 处理（数据不出内网）")
print("   - 高峰期弹性扩容                 → DeepSeek API 兜底（无需加 GPU）")
print()

# ═══════════════════════════════════════════════════
# 六、vLLM 与 Ollama 的区别（常见混淆点）
# ═══════════════════════════════════════════════════
print("── 六、vLLM vs Ollama：不要搞混 ──")
print()
print("   vLLM 和 Ollama 都能在本地起一个 OpenAI 兼容 API，但有本质区别：")
print()
vllm_vs_ollama = [
    ("目标用户",     "Ollama：个人开发者，开箱即用",          "vLLM：企业生产环境，高性能"),
    ("模型格式",     "Ollama：GGUF 量化格式",                 "vLLM：HF 原始格式 + AWQ/GPTQ"),
    ("推理性能",     "Ollama：适合单用户，吞吐量一般",         "vLLM：连续批处理，吞吐量极高"),
    ("显存管理",     "Ollama：基本显存分配",                  "vLLM：PagedAttention，近乎零浪费"),
    ("并发能力",     "Ollama：排队处理",                       "vLLM：动态 batching，GPU 满载"),
    ("部署复杂度",   "Ollama：brew install / 一键安装包",      "vLLM：需要 CUDA 环境，pip install"),
    ("API 兼容",     "Ollama：自有格式 + OpenAI 兼容",        "vLLM：原生 OpenAI 兼容"),
    ("示例业务系统 选谁",   "Ollama：课堂演示、本地开发调试",         "vLLM：生产环境高并发推理"),
]
for dim, ollama_side, vllm_side in vllm_vs_ollama:
    print(f"   {dim:<12} {ollama_side:<38} {vllm_side}")
print()

# ═══════════════════════════════════════════════════
# 七、环境检测（课堂实用）
# ═══════════════════════════════════════════════════
print("── 七、课堂环境检测 ──")
print()

# 检测 GPU
gpu_available = False
try:
    import torch
    if torch.cuda.is_available():
        gpu_available = True
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"   ✓ GPU 可用：{gpu_name}（{gpu_mem:.0f} GB）")
    else:
        print("   ✗ 未检测到 GPU（vLLM 需要 NVIDIA GPU）")
except ImportError:
    print("   ✗ PyTorch 未安装")
    print("   （课堂可先理解概念，课后在有 GPU 的机器上实操）")

# 检测 vLLM
vllm_installed = False
try:
    import vllm
    vllm_installed = True
    print(f"   ✓ vLLM 已安装（版本 {vllm.__version__}）")
except ImportError:
    print("   ✗ vLLM 未安装")
    print("   （课堂可先理解概念，课后安装：pip install vllm）")
print()

# ═══════════════════════════════════════════════════
# 八、课后实操指引
# ═══════════════════════════════════════════════════
print("── 八、课后实操指引 ──")
print()
print("   如果你有一张 NVIDIA GPU（>=8GB 显存），可以试试：")
print()
print("   # 1. 安装 vLLM")
print("   pip install vllm")
print()
print("   # 2. 下载一个小模型（约 1.5GB）")
print("   huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir ~/models/qwen-1.5b")
print()
print("   # 3. 启动服务")
print("   python -m vllm.entrypoints.openai.api_server \\")
print("       --model ~/models/qwen-1.5b \\")
print("       --served-model-name qwen-1.5b \\")
print("       --max-model-len 4096")
print()
print("   # 4. 用 Python 调用（前面第四节的代码直接用）")
print()
print("   # 5. 对比 vLLM 和 transformers 的速度差异：")
print("   #    同样的问题，同样的小模型，vLLM 的吞吐量通常是 transformers 的 10-30 倍")

# 课堂小结
print()
print("── 本节小结 ──")
print()
print("   vLLM 是连接「本地学习」和「生产部署」的桥梁：")
print("   1. 学原理       → transformers（Ch01）")
print("   2. 本地服务     → Ollama（本节之前）")
print("   3. 云端 API     → DeepSeek / OpenAI（本 lesson_02）")
print("   4. 自建高性能   → vLLM（本节新增）★")
print()
print("   五类后端覆盖了从本地到云端、从学习到生产的全部场景。")