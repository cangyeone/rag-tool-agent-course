"""04 加记忆系统：公共记忆 + 私有记忆。

课堂目标：
1. public_memory：所有玩家都能看到。
2. private_memory：每个玩家自己看到。
3. memory 会影响下一轮发言。

本脚本只演示两名玩家的两轮发言，便于看清 memory 的作用。
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
    env_file = str(COURSE_ROOT / "code/05_专题_多角色推理游戏/code/.env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("DEEPSEEK_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
if not api_key:
    raise SystemExit("请先在同目录 .env 中写入 DEEPSEEK_API_KEY，或设置系统环境变量。")

base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
url = base_url + "/chat/completions"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

players = ["A", "B"]
roles = {"A": "狼人", "B": "预言家"}

public_memory = [
    "第 1 天：B 怀疑 A 发言太保守。",
]

private_memory = {
    "A": ["我是狼人，不能暴露身份。B 对我有怀疑，需要转移注意。"],
    "B": ["我是预言家，昨晚查验 A，结果显示 A 是狼人。"],
}

for round_no in [1, 2]:
    print(f"\n第 {round_no} 轮发言")
    print("=" * 72)

    for player in players:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是狼人杀玩家。你会收到公共记忆和你的私有记忆。"
                    "请输出 JSON，字段为 speech、new_memory。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "玩家": player,
                        "身份": roles[player],
                        "公共记忆": public_memory,
                        "你的私有记忆": private_memory[player],
                        "任务": "根据记忆发言，并写一条新的私有记忆。",
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        payload = {
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 500,
            "response_format": {"type": "json_object"},
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = json.loads(response.json()["choices"][0]["message"]["content"])

        speech = data.get("speech", "")
        new_memory = data.get("new_memory", "")

        print(f"{player} 发言：{speech}")
        public_memory.append(f"第 {round_no} 轮：{player} 说：{speech}")
        if new_memory:
            private_memory[player].append(new_memory)

print("\n最终公共记忆：")
print(json.dumps(public_memory, ensure_ascii=False, indent=2))

print("\n最终私有记忆：")
print(json.dumps(private_memory, ensure_ascii=False, indent=2))
