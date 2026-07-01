"""02 工具调用：让模型选择 read_file。

课堂目标：
1. 模型不直接接触文件系统。
2. 模型发起 tool call。
3. 程序执行工具，把结果交给模型。
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
    env_file = str(COURSE_ROOT / "code/10_专题_极简代码智能体/code/.env")
    if os.path.exists(env_file):
        for line in open(env_file, "r", encoding="utf-8"):
            if line.strip().startswith("DEEPSEEK_API_KEY="):
                api_key = line.strip().split("=", 1)[1]
if not api_key:
    raise SystemExit("请设置 DEEPSEEK_API_KEY，或复制仓库根目录 .env.example 为 .env 后填写。")

base_dir = COURSE_ROOT / "code/10_专题_极简代码智能体/code"
project_dir = base_dir / "demo_project"

base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
url = base_url + "/chat/completions"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
user_id = os.getenv("DEEPSEEK_USER_ID", os.getenv("USER", "classroom_user"))

messages = [
    {"role": "system", "content": "你是 coding agent。需要看文件时，请调用 read_file 工具。"},
    {"role": "user", "content": "请读取 demo_project/calculator.py，看看 add 函数可能有什么问题。"},
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取 demo_project 里的文件内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对路径，例如 calculator.py"}
                },
                "required": ["path"],
            },
        },
    }
]

payload = {
    "model": model,
    "messages": messages,
    "tools": tools,
    "tool_choice": "auto",
    "thinking": {"type": "disabled"},
    "temperature": 0.1,
    "max_tokens": 600,
    "user_id": user_id,
}

print("模型：", model)
print("user_id：", user_id)
print("-" * 72)

response = requests.post(url, headers=headers, json=payload, timeout=60)
response.raise_for_status()
assistant_message = response.json()["choices"][0]["message"]
messages.append(assistant_message)

print("模型返回：")
print(json.dumps(assistant_message, ensure_ascii=False, indent=2))
print("-" * 72)

tool_calls = assistant_message.get("tool_calls") or []

for tool_call in tool_calls:
    tool_name = tool_call["function"]["name"]
    tool_args = json.loads(tool_call["function"]["arguments"])

    if tool_name == "read_file":
        safe_path = Path(tool_args["path"]).name
        file_path = project_dir / safe_path
        tool_result = file_path.read_text(encoding="utf-8")
    else:
        tool_result = f"未知工具：{tool_name}"

    print("程序执行工具：", tool_name)
    print("工具参数：", tool_args)
    print("工具结果：")
    print(tool_result)

    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": tool_result,
        }
    )

payload = {
    "model": model,
    "messages": messages,
    "thinking": {"type": "disabled"},
    "temperature": 0.1,
    "max_tokens": 600,
    "user_id": user_id,
}

response = requests.post(url, headers=headers, json=payload, timeout=60)
response.raise_for_status()

print("-" * 72)
print("模型基于文件内容的分析：")
print(response.json()["choices"][0]["message"]["content"])