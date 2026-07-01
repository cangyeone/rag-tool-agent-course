"""06 加裁判 Agent：检查状态并裁决胜负。

课堂目标：
1. 裁判 Agent 不参与发言，只负责检查规则。
2. 输入是 game_state，输出是结构化 JSON。
3. 裁判 Agent 可以发现异常投票、更新阶段、判断阵营胜负。
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
    env_file = str(COURSE_ROOT / "code/09_专题_多角色推理游戏/code/.env")
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

game_state = {
    "day": 1,
    "stage": "day_vote_finished",
    "alive_players": ["A", "B", "C", "D", "E"],
    "roles": {
        "A": "狼人",
        "B": "预言家",
        "C": "女巫",
        "D": "村民",
        "E": "村民",
    },
    "votes": {
        "A": "B",
        "B": "A",
        "C": "A",
        "D": "A",
        "E": "B",
    },
}

# 程序先做确定性统计。裁判 Agent 再基于统计做解释和状态裁决。
vote_count = Counter(game_state["votes"].values())
out_player = vote_count.most_common(1)[0][0]
alive_after_vote = [p for p in game_state["alive_players"] if p != out_player]
wolf_alive = [p for p in alive_after_vote if game_state["roles"][p] == "狼人"]
good_alive = [p for p in alive_after_vote if game_state["roles"][p] != "狼人"]

rule_result = {
    "vote_count": dict(vote_count),
    "out_player": out_player,
    "alive_after_vote": alive_after_vote,
    "wolf_alive": wolf_alive,
    "good_alive": good_alive,
}

messages = [
    {
        "role": "system",
        "content": (
            "你是狼人杀裁判 Agent。你不参与玩家发言，只做规则裁决。"
            "必须输出 JSON，字段为 valid、next_stage、winner、judge_explain、warnings。"
        ),
    },
    {
        "role": "user",
        "content": json.dumps(
            {
                "game_state": game_state,
                "确定性统计结果": rule_result,
                "胜负规则": "狼人全部出局则好人胜利；狼人数量大于等于好人数量则狼人胜利；否则进入下一夜。",
            },
            ensure_ascii=False,
        ),
    },
]

payload = {
    "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    "messages": messages,
    "temperature": 0.1,
    "max_tokens": 600,
    "response_format": {"type": "json_object"},
}

print("程序统计结果：")
print(json.dumps(rule_result, ensure_ascii=False, indent=2))
print("-" * 72)

response = requests.post(url, headers=headers, json=payload, timeout=60)
response.raise_for_status()

content = response.json()["choices"][0]["message"]["content"]
judge = json.loads(content)

print("裁判 Agent 裁决：")
print(json.dumps(judge, ensure_ascii=False, indent=2))
