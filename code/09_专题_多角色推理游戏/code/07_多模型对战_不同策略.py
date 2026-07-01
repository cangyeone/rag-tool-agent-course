"""07 多模型对战：不同模型配置、不同策略。

课堂目标：
1. 同一个游戏状态，可以交给不同模型或不同参数。
2. 低 temperature 更稳，高 temperature 更有变化。
3. 可以把模型输出放在一起比较，形成评测思路。

说明：
- 默认都用 deepseek-v4-flash，靠 temperature 区分策略。
- 如果账号支持其他模型，可用环境变量 DEEPSEEK_MODEL_A / DEEPSEEK_MODEL_B 替换。
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

model_a = os.getenv("DEEPSEEK_MODEL_A", "deepseek-v4-flash")
model_b = os.getenv("DEEPSEEK_MODEL_B", "deepseek-v4-pro")

game_state = {
    "day": 1,
    "alive_players": ["A", "B", "C", "D", "E"],
    "public_history": [
        "B：A 的发言比较谨慎，我觉得有点可疑。",
        "C：目前还没有硬信息，先听后面的人怎么说。",
    ],
}

agents = [
    {
        "name": "模型A_稳健玩家",
        "model": model_a,
        "temperature": 0.2,
        "role": "预言家",
        "player": "B",
        "style": "稳健、少下结论、尽量给证据。",
    },
    {
        "name": "模型B_进攻玩家",
        "model": model_b,
        "temperature": 0.9,
        "role": "狼人",
        "player": "A",
        "style": "主动转移怀疑，发言更有攻击性。",
    },
]

results = []

for agent in agents:
    messages = [
        {
            "role": "system",
            "content": (
                "你是狼人杀玩家。请根据角色、风格和公开局面发言。"
                "必须输出 JSON，字段为 player、speech、vote_target、reason。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "玩家": agent["player"],
                    "身份": agent["role"],
                    "发言风格": agent["style"],
                    "公开局面": game_state,
                    "可投票对象": [p for p in game_state["alive_players"] if p != agent["player"]],
                },
                ensure_ascii=False,
            ),
        },
    ]

    payload = {
        "model": agent["model"],
        "messages": messages,
        "temperature": agent["temperature"],
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
    }

    print(f"\n调用 {agent['name']} | model={agent['model']} | temperature={agent['temperature']}")
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = json.loads(response.json()["choices"][0]["message"]["content"])
    results.append({"agent": agent, "result": data})

    print(json.dumps(data, ensure_ascii=False, indent=2))

print("\n对战结果对比")
print("=" * 72)
for item in results:
    agent = item["agent"]
    data = item["result"]
    print(f"{agent['name']}：投票给 {data.get('vote_target')} | 理由：{data.get('reason')}")
