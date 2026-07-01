"""01 生成 Plan：让模型先拆任务。

课堂目标：
1. Coding Agent 不应该一上来就改代码。
2. 它先读任务，再生成 plan。
3. plan 用 JSON 输出，程序才能继续读取。
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
    env_file = str(COURSE_ROOT / "code/10_专题_极简CodingAgent/code/.env")
    if os.path.exists(env_file):
        for line in open(env_file, "r", encoding="utf-8"):
            if line.strip().startswith("DEEPSEEK_API_KEY="):
                api_key = line.strip().split("=", 1)[1]
if not api_key:
    raise SystemExit("请设置 DEEPSEEK_API_KEY，或复制仓库根目录 .env.example 为 .env 后填写。")

base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
url = base_url + "/chat/completions"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
user_id = os.getenv("DEEPSEEK_USER_ID", os.getenv("USER", "classroom_user"))

goal = "修复 demo_project/calculator.py 里的 add 函数，让测试通过。"

messages = [
    {
        "role": "system",
        "content": (
            "你是一个极简 coding agent。先不要写代码，先给出计划。"
            "必须输出 JSON，字段为 goal、steps、need_tools、done_check。"
        ),
    },
    {"role": "user", "content": goal},
]

payload = {
    "model": model,
    "messages": messages,
    "thinking": {"type": "disabled"},
    "temperature": 0.2,
    "max_tokens": 600,
    "response_format": {"type": "json_object"},
    "user_id": user_id,
}

print("任务：", goal)
print("模型：", model)
print("user_id：", user_id)
print("-" * 72)

response = requests.post(url, headers=headers, json=payload, timeout=60)
response.raise_for_status()
content = response.json()["choices"][0]["message"]["content"]

print("模型输出的 plan JSON：")
print(content)

plan = json.loads(content)
print("\n程序读取 plan：")
for i, step in enumerate(plan.get("steps", []), 1):
    print(f"{i}. {step}")