"""04 压缩 Memory：使用 LLM 把长历史压缩成短摘要。

课堂目标：
1. 长对话不能无限塞进上下文。
2. 压缩 memory 不是简单截断，而是让 LLM 读完整历史后提炼要点。
3. 下一轮辩论只带 summary + 最近几条发言。
"""

import os
import json
import requests
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")


api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    env_file = str(COURSE_ROOT / "code/11_专题_辩论与文章润色/code/.env")
    if os.path.exists(env_file):
        for line in open(env_file, "r", encoding="utf-8"):
            if line.strip().startswith("DEEPSEEK_API_KEY="):
                api_key = line.strip().split("=", 1)[1]
if not api_key:
    raise SystemExit("请设置 DEEPSEEK_API_KEY，或复制仓库根目录 .env.example 为 .env 后填写。")

base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
url = base_url + "/chat/completions"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

public_memory = [
    "正方：RAG 可以直接接入企业文档，更新快。",
    "反方：微调可以让模型更稳定地完成固定任务。",
    "正方：RAG 的引用来源更清楚，便于审计。",
    "反方：如果文档质量差，RAG 也会检索到错误资料。",
    "正方：RAG 更适合频繁变化的制度和知识库。",
    "反方：微调适合沉淀长期稳定的业务表达方式。",
]

messages = [
    {
        "role": "system",
        "content": (
            "你是辩论记录员，负责做 memory compression。"
            "请阅读完整 public_memory，把它压缩成下一轮辩论可用的短记忆。"
            "不要逐句复述，要保留双方核心论点、冲突点和未解决问题。"
            "必须输出 JSON，字段为 summary、pro_points、con_points、open_questions。"
        ),
    },
    {
        "role": "user",
        "content": json.dumps({"public_memory": public_memory}, ensure_ascii=False),
    },
]

payload = {
    "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    "messages": messages,
    "temperature": 0.2,
    "max_tokens": 600,
    "response_format": {"type": "json_object"},
}

print("一、压缩前的 public_memory")
print("条数：", len(public_memory))
print(json.dumps(public_memory, ensure_ascii=False, indent=2))
print("-" * 72)

print("二、把完整 memory 交给 LLM 做压缩")
print("模型：", payload["model"])
print("输出格式：JSON")
print("压缩要求：保留双方核心论点、冲突点和未解决问题")
print("-" * 72)

response = requests.post(url, headers=headers, json=payload, timeout=60)
response.raise_for_status()
compressed = json.loads(response.json()["choices"][0]["message"]["content"])

print("三、LLM 压缩后的 memory")
print(json.dumps(compressed, ensure_ascii=False, indent=2))

# 实际系统里常见做法：保留压缩摘要 + 最近几条原文。
recent = public_memory[-2:]
next_context = {
    "compressed_memory": compressed,
    "recent_messages": recent,
}

print("\n四、下一轮只带这些上下文")
print(json.dumps(next_context, ensure_ascii=False, indent=2))