"""
学习目标：总结本地模型到云端 API 的过渡路线图，
形成完整的心智模型，知道什么场景用什么后端。
"""

import os
import json
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("09 从本地到云端 —— Ch02 lesson_01 总结")
print("=" * 72)

# ── 路线图 ──
print("── 从本地到云端的成长路径 ──")
print()
stages = [
    ("阶段 1：本地探索", "Ch01", "transformers", "理解 tokenize / generate / decode，建立大模型运作直觉"),
    ("阶段 2：服务化认知", "Ch02 L1", "HTTP 协议", "理解 API 调用方式、请求/响应结构、密钥管理"),
    ("阶段 3：云端实战", "Ch02 L2", "DeepSeek API", "真实调用、消息结构、错误处理、结构化输出"),
    ("阶段 4：RAG 构建", "Ch03+", "DeepSeek + 向量库", "知识库检索增强生成，解决大模型幻觉问题"),
    ("阶段 5：Agent 工具", "Ch04+", "Function Calling", "模型调用外部工具，从\u201c说\u201d到\u201c做\u201d"),
    ("阶段 6：生产部署", "Ch05+", "可视化工具 / LangChain", "工作流编排、评测、监控、灰度发布"),
]

for stage, chapter, tech, desc in stages:
    print(f"   {stage}")
    print(f"      对应章节：{chapter}")
    print(f"      核心技术：{tech}")
    print(f"      目标：{desc}")
    print()

# ── 关键概念回顾 ──
print("── 本 lesson 关键概念回顾 ──")
print()
concepts = [
    ("本地模型三步骤", "tokenize → generate → decode，全在本机运行"),
    ("本地模型五大局限", "资源占用、无 API、难扩展、规模受限、运维成本高"),
    ("模型服务化", "把模型包装成 HTTP API，支持多租户和负载均衡"),
    ("Chat Completions API", "POST /v1/chat/completions，统一的云端调用协议"),
    ("API 密钥管理", "环境变量 → .env → .gitignore，绝不硬编码"),
    ("五类后端", "transformers / Ollama / DeepSeek / OpenAI / vLLM，按场景选择"),
]
for concept, desc in concepts:
    print(f"   ■ {concept}")
    print(f"     {desc}")
    print()

# ── 代码速查 ──
print("── 代码速查：四种后端的差异 ──")
print()
print("# transformers")
print("model.generate(**inputs, max_new_tokens=100)")
print()
print("# Ollama")
print("POST http://localhost:11434/api/generate")
print('{"model": "qwen2.5:0.5b", "prompt": "...", "stream": false}')
print()
print("# DeepSeek & OpenAI (同一格式)")
print("base_url = https://api.deepseek.com")
print("POST /chat/completions")
print("headers: Authorization: Bearer $DEEPSEEK_API_KEY")
print('body: {"model": "deepseek-v4-flash", "messages": [...], "thinking": {"type": "disabled"}}')
print("# vLLM（OpenAI 兼容）")
print("python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct")
print("POST http://localhost:8000/v1/chat/completions")
print('body: {"model": "qwen-7b", "messages": [...]}')
print()

# ── 决策流程图（ASCII art）──
print("── 什么时候用什么？决策流程图 ──")
print()
print("""
                    开始选择模型后端
                          │
              ┌───────────┼───────────┐
              │           │           │
         需要离线？    需要 HTTP？   啥都不需要
              │           │           │
           ┌──┴──┐    ┌──┴──┐    直接上云端
           │     │    │     │        │
      数据敏感？ 学习？ │  小型服务？  ┌──┴──┐
           │     │    │     │        │     │
        是│   否│    │     │      预算低？ 要最强？
          │     │    │     │        │     │
   transformers  Ollama  Ollama  transformers  DeepSeek  OpenAI
   (有GPU)   (无GPU)                  (中文首选)  (GPT-4o)

   项目起步 → DeepSeek API
   本地调试 → transformers 或 Ollama
   灰度发布 → DeepSeek API + DeepSeek-R1
   最终上线 → 以 DeepSeek API 为主，Ollama 做兜底
""")

# ── 下一步 ──
print("── 下一步：lesson_02 DeepSeek 调用与回答边界 ──")
print()
print("   lesson_01 学会了\u201c怎么调用\u201d，lesson_02 学\u201c怎么用好\u201d。")
print("   具体内容：")
print("   1. 最小 DeepSeek 请求体 —— 发送第一个真实 API 请求")
print("   2. 系统消息控制边界 —— 通过 system prompt 约束模型行为")
print("   3. 结构化 JSON 输出 —— 让模型返回机器可解析的数据")
print("   4. 多案例批量测试 —— 测试模型在同类问题上的一致性")
print("   5. 错误与兜底 —— 网络超时、限流、重试策略")
print("   6. AI Native 应用画布 —— 通用智能应用的完整架构")

# ── 课堂检查点 ──
print()
print("── 课堂检查点：你掌握了这些吗？ ──")
checks = [
    ("能说出本地模型的三个核心步骤", False),
    ("能列出本地模型的两个以上局限", False),
    ("知道 Chat Completions API 的请求 URL", False),
    ("理解 messages 中 system/user/assistant 三种角色", False),
    ("知道如何安全存储 API Key", False),
    ("能对比四种模型后端的使用场景", False),
    ("能把一个 transformers 调用改写为 API 调用", False),
]
for desc, _ in checks:
    print(f"   [ ] {desc}")
print()
print("   （课堂结束后自己打勾，确认学习成果）")

# ── 环境变量提醒 ──
api_key = os.getenv("DEEPSEEK_API_KEY", "")
if not api_key:
    print("\n⚠ 提醒：你还没设置 DEEPSEEK_API_KEY。")
    print("   lesson_02 需要发送真实 API 请求，请提前设置：")
    print("   export DEEPSEEK_API_KEY=your_api_key_here")
    print("   获取地址：https://platform.deepseek.com/api_keys")
else:
    print(f"\n✓ DEEPSEEK_API_KEY 已设置（{api_key[:8]}...），可以进入 lesson_02。")