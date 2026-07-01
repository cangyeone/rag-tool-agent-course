"""03 生成代码并写入文件。

课堂目标：
1. 把原始代码发给模型。
2. 要求模型输出完整文件内容。
3. 程序写入新文件 calculator_fixed.py。
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
    env_file = str(COURSE_ROOT / "code/06_专题_极简代码智能体/code/.env")
    if os.path.exists(env_file):
        for line in open(env_file, "r", encoding="utf-8"):
            if line.strip().startswith("DEEPSEEK_API_KEY="):
                api_key = line.strip().split("=", 1)[1]
if not api_key:
    raise SystemExit("请设置 DEEPSEEK_API_KEY，或复制仓库根目录 .env.example 为 .env 后填写。")

base_dir = COURSE_ROOT / "code/06_专题_极简代码智能体/code"
project_dir = base_dir / "demo_project"
source_path = project_dir / "calculator.py"
target_path = project_dir / "calculator_fixed.py"

source_code = source_path.read_text(encoding="utf-8")

base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
url = base_url + "/chat/completions"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
user_id = os.getenv("DEEPSEEK_USER_ID", os.getenv("USER", "classroom_user"))

messages = [
    {
        "role": "system",
            "content": (
                "你是 coding agent。请修复代码。"
                "必须输出 JSON，字段为 explanation、new_code。new_code 必须是完整 Python 文件。"
                "new_code 里只放纯代码，不要放 ```python 或 ```。"
            ),
        },
    {
        "role": "user",
        "content": json.dumps(
            {
                "任务": "修复 add 函数，让加法正确。",
                "文件名": "calculator.py",
                "当前代码": source_code,
            },
            ensure_ascii=False,
        ),
    },
]

payload = {
    "model": model,
    "messages": messages,
    "thinking": {"type": "disabled"},
    "temperature": 0.1,
    "max_tokens": 800,
    "response_format": {"type": "json_object"},
    "user_id": user_id,
}

print("原始代码：")
print(source_code)
print("-" * 72)

response = requests.post(url, headers=headers, json=payload, timeout=60)
response.raise_for_status()
data = json.loads(response.json()["choices"][0]["message"]["content"])
new_code = data["new_code"].strip()

print("模型解释：")
print(data["explanation"])
print("\n模型生成的新代码：")
print("```python")
print(new_code)
print("```")

target_path.write_text(new_code + "\n", encoding="utf-8")
print("\n已写入：", target_path)