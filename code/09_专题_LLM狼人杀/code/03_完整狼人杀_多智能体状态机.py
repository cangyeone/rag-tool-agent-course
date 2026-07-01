"""03 完整狼人杀：多智能体 + 游戏状态机。

课堂目标：
1. 每个玩家都是一个 Agent。
2. game_state 保存当前局面。
3. 每个 Agent 只看到公开信息和自己的私有身份。
4. 用 JSON 让每个 Agent 输出发言和投票。

为了课堂节奏，本脚本只跑一轮白天发言和投票。
"""

import os
import json
import requests
from collections import Counter
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")


api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    env_file = str(COURSE_ROOT / "code/09_专题_LLM狼人杀/code/.env")
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

# 状态机：用 stage 表示游戏推进到哪一步。
game_state = {
    "day": 1,
    "stage": "day_speech",
    "alive_players": ["A", "B", "C", "D", "E"],
    "public_history": ["主持人：第 1 天白天开始，请依次发言。"],
    "votes": {},
}

roles = {
    "A": "狼人",
    "B": "预言家",
    "C": "女巫",
    "D": "村民",
    "E": "村民",
}

print("初始 game_state：")
print(json.dumps(game_state, ensure_ascii=False, indent=2))
print("-" * 72)

# 第一阶段：每个玩家发言。
for player in game_state["alive_players"]:
    messages = [
        {
            "role": "system",
            "content": (
                "你是狼人杀玩家。请根据公开信息和自己的身份发言。"
                "必须输出 JSON，字段为 player、speech、suspect、reason。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "你是谁": player,
                    "你的身份": roles[player],
                    "公开游戏状态": game_state,
                    "任务": "轮到你发言。不要泄露不该公开的信息。",
                },
                ensure_ascii=False,
            ),
        },
    ]

    payload = {
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    data = json.loads(content)

    line = f"{player}：{data.get('speech', '')}"
    game_state["public_history"].append(line)
    print(line)

print("-" * 72)

# 第二阶段：投票。为了简单，每个 Agent 从存活玩家里选择一个目标。
game_state["stage"] = "day_vote"
for player in game_state["alive_players"]:
    candidates = [p for p in game_state["alive_players"] if p != player]

    messages = [
        {
            "role": "system",
            "content": (
                "你是狼人杀玩家，现在进入投票阶段。"
                "必须输出 JSON，字段为 voter、vote、reason。vote 必须是候选人之一。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "你是谁": player,
                    "你的身份": roles[player],
                    "可投票对象": candidates,
                    "公开游戏状态": game_state,
                    "任务": "请选择一个投票对象。",
                },
                ensure_ascii=False,
            ),
        },
    ]

    payload = {
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 400,
        "response_format": {"type": "json_object"},
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = json.loads(response.json()["choices"][0]["message"]["content"])

    vote = data.get("vote")
    if vote not in candidates:
        vote = candidates[0]
    game_state["votes"][player] = vote
    print(f"{player} 投票给 {vote}，理由：{data.get('reason')}")

vote_count = Counter(game_state["votes"].values())
out_player = vote_count.most_common(1)[0][0]
game_state["stage"] = "day_result"
game_state["out_player"] = out_player

print("-" * 72)
print("投票统计：", dict(vote_count))
print("出局玩家：", out_player)
print("\n最终 game_state：")
print(json.dumps(game_state, ensure_ascii=False, indent=2))
