"""05 最小 Coding Agent 循环。

课堂目标：
1. plan：模型先给计划。
2. act：程序按计划执行工具。
3. observe：工具返回观察结果。
4. memory：把观察结果保存下来。
5. repeat：根据测试结果继续修复。

为了初学者能看清楚，工具不封装成复杂类，只用 if/elif。
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
    env_file = str(COURSE_ROOT / "code/10_专题_极简CodingAgent/code/.env")
    if os.path.exists(env_file):
        for line in open(env_file, "r", encoding="utf-8"):
            if line.strip().startswith("DEEPSEEK_API_KEY="):
                api_key = line.strip().split("=", 1)[1]
if not api_key:
    raise SystemExit("请设置 DEEPSEEK_API_KEY，或复制仓库根目录 .env.example 为 .env 后填写。")

base_dir = COURSE_ROOT / "code/10_专题_极简CodingAgent/code"
project_dir = base_dir / "demo_project"
code_path = project_dir / "calculator.py"
test_path = project_dir / "test_calculator.py"

# 先重置 bug，保证每次演示都能看到 agent 修复过程。
code_path.write_text(
    "def add(a, b):\n"
    "    return a - b\n\n\n"
    "def multiply(a, b):\n"
    "    return a * b\n",
    encoding="utf-8",
)

base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
url = base_url + "/chat/completions"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
user_id = os.getenv("DEEPSEEK_USER_ID", os.getenv("USER", "classroom_user"))

goal = "修复 demo_project/calculator.py，让 test_calculator.py 通过。"
memory = []

for round_no in range(1, 5):
    print(f"\n第 {round_no} 轮 Agent 循环")
    print("=" * 72)

    # 一个很小但很重要的 agent harness：
    # 如果上一轮刚写过文件，下一轮不再问模型“要做什么”，程序直接跑测试。
    # 这样可以避免模型反复写代码，却忘记验证结果。
    if memory and memory[-1]["action"] == "write_file":
        print("程序策略：上一轮写过文件，本轮直接运行测试。")
        result = subprocess.run(
            ["python", str(test_path.name)],
            cwd=project_dir,
            text=True,
            capture_output=True,
        )
        observation = (
            f"测试 returncode={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        print("\n工具 observation：")
        print(observation)
        memory.append({"round": round_no, "action": "run_tests", "observation": observation})
        if result.returncode == 0:
            print("测试通过，Agent 任务完成。")
            break
        continue

    messages = [
        {
            "role": "system",
            "content": (
                "你是极简 coding agent。你只能选择一个 action。"
                "必须输出 JSON，字段为 thought、action、path、new_code、reason。"
                "action 只能是 read_file、write_file、run_tests、finish。"
                "如果要写文件，new_code 必须是完整文件内容。"
                "new_code 只放纯 Python 代码，不要放 ```python 或 ```。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "goal": goal,
                    "memory": memory,
                    "可用文件": ["calculator.py", "test_calculator.py"],
                    "规则": "先读文件，再运行测试，再修复，再复测。测试通过后 finish。",
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
        "max_tokens": 1000,
        "response_format": {"type": "json_object"},
        "user_id": user_id,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    step = json.loads(response.json()["choices"][0]["message"]["content"])

    print("模型 thought：", step.get("thought"))
    print("模型 action：", step.get("action"))
    print("模型 reason：", step.get("reason"))

    action = step.get("action")
    path = Path(str(step.get("path") or "")).name

    if action == "read_file":
        file_path = project_dir / path
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            observation = f"读取 {path}：\n{content}"
        else:
            observation = f"文件不存在：{path}"

    elif action == "write_file":
        file_path = project_dir / path
        if path not in ["calculator.py", "test_calculator.py"]:
            observation = f"拒绝写入非演示文件：{path}"
        else:
            new_code = (step.get("new_code") or "").strip()
            file_path.write_text(new_code + "\n", encoding="utf-8")
            observation = f"已写入 {path}"

    elif action == "run_tests":
        result = subprocess.run(
            ["python", str(test_path.name)],
            cwd=project_dir,
            text=True,
            capture_output=True,
        )
        observation = (
            f"测试 returncode={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    elif action == "finish":
        print("Agent 认为任务完成。")
        break

    else:
        observation = f"未知 action：{action}"

    print("\n工具 observation：")
    print(observation)
    memory.append({"round": round_no, "action": action, "observation": observation})

print("\n最终 calculator.py：")
print(code_path.read_text(encoding="utf-8"))