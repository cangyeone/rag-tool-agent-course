"""05 运行测试并自动修复。

课堂目标：
1. 先运行测试，拿到错误。
2. 把错误和代码一起发给模型。
3. 写回修复后的 calculator.py。
4. 再运行测试，观察是否通过。
"""

import os
import json
import subprocess
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
code_path = project_dir / "calculator.py"
test_path = project_dir / "test_calculator.py"

# 每次演示前先恢复一个明显 bug，保证课堂输出稳定。
code_path.write_text(
    "def add(a, b):\n"
    "    return a - b\n\n\n"
    "def multiply(a, b):\n"
    "    return a * b\n",
    encoding="utf-8",
)

print("第一次运行测试：")
first = subprocess.run(
    ["python", str(test_path.name)],
    cwd=project_dir,
    text=True,
    capture_output=True,
)
print("returncode:", first.returncode)
print(first.stdout)
print(first.stderr)

source_code = code_path.read_text(encoding="utf-8")
test_code = test_path.read_text(encoding="utf-8")
error_text = first.stdout + "\n" + first.stderr

base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
url = base_url + "/chat/completions"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
user_id = os.getenv("DEEPSEEK_USER_ID", os.getenv("USER", "classroom_user"))

messages = [
    {
        "role": "system",
            "content": (
                "你是 coding agent。请根据测试错误修复代码。"
                "必须输出 JSON，字段为 bug_reason、new_code。new_code 是完整 calculator.py。"
                "new_code 里只放纯 Python 代码，不要放 Markdown 代码块标记。"
            ),
        },
    {
        "role": "user",
        "content": json.dumps(
            {
                "calculator.py": source_code,
                "test_calculator.py": test_code,
                "测试错误": error_text,
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
    "max_tokens": 900,
    "response_format": {"type": "json_object"},
    "user_id": user_id,
}

response = requests.post(url, headers=headers, json=payload, timeout=60)
response.raise_for_status()
data = json.loads(response.json()["choices"][0]["message"]["content"])
new_code = data["new_code"].strip()

print("-" * 72)
print("模型判断 bug 原因：")
print(data["bug_reason"])

print("\n模型生成的新代码：")
print("```python")
print(new_code)
print("```")

code_path.write_text(new_code + "\n", encoding="utf-8")
print("\n已写回 calculator.py")

print("\n第二次运行测试：")
second = subprocess.run(
    ["python", str(test_path.name)],
    cwd=project_dir,
    text=True,
    capture_output=True,
)
print("returncode:", second.returncode)
print(second.stdout)
print(second.stderr)