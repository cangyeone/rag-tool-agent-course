"""02 单角色狼人：角色扮演 + JSON 输出。

课堂目标：
1. 展示 Role Playing：同一个模型可以扮演一个具体角色。
2. 展示 JSON：让模型按固定字段输出，方便程序读取。
3. 展示博弈：狼人需要隐藏身份，同时推动投票。
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

public_state = {
    "day": 1,
    "alive_players": ["A", "B", "C", "D", "E"],
    "public_history": [
        "主持人：天亮了，现在开始第一轮发言。",
        "B：我觉得 A 昨晚太安静了，有点可疑。",
        "C：现在信息太少，不要太快下结论。",
    ],
}

private_role = {
    "player": "A",
    "role": "狼人",
    "goal": "隐藏身份，误导好人阵营，把怀疑引向其他玩家。",
}

messages = [
    {
        "role": "system",
        "content": (
            "你正在扮演狼人杀玩家。你必须严格输出 JSON，不能输出多余文字。"
            "JSON 字段为 speech、target、reason、risk。"
        ),
    },
    {
        "role": "user",
        "content": json.dumps(
            {
                "公开局面": public_state,
                "你的私有身份": private_role,
                "任务": "请作为 A 发言，隐藏狼人身份，并给出你想推动怀疑的目标。",
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

print("公开局面：")
print(json.dumps(public_state, ensure_ascii=False, indent=2))
print("\nA 的私有身份只给模型，不给其他玩家：")
print(json.dumps(private_role, ensure_ascii=False, indent=2))
print("-" * 72)

response = requests.post(url, headers=headers, json=payload, timeout=60)
response.raise_for_status()

content = response.json()["choices"][0]["message"]["content"]
print("模型原始输出：")
print(content)

print("\n程序解析 JSON：")
data = json.loads(content)
print("发言：", data.get("speech"))
print("怀疑目标：", data.get("target"))
print("理由：", data.get("reason"))
print("风险：", data.get("risk"))
