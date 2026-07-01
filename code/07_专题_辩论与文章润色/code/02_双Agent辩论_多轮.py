"""02 双 Agent 辩论：正反方多轮交互。

课堂目标：
1. 正方 Agent 和反方 Agent 轮流发言。
2. 每个 Agent 都能看到公开历史。
3. public_history 就是最简单的共享记忆。
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
    env_file = str(COURSE_ROOT / "code/07_专题_辩论与文章润色/code/.env")
    if os.path.exists(env_file):
        for line in open(env_file, "r", encoding="utf-8"):
            if line.strip().startswith("DEEPSEEK_API_KEY="):
                api_key = line.strip().split("=", 1)[1]
if not api_key:
    raise SystemExit("请设置 DEEPSEEK_API_KEY，或复制仓库根目录 .env.example 为 .env 后填写。")

base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
url = base_url + "/chat/completions"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

topic = "企业内部知识问答系统应该优先使用 RAG，而不是微调。"
public_history = []

agents = [
    {"name": "正方", "position": "支持优先使用 RAG"},
    {"name": "反方", "position": "反对优先使用 RAG，认为微调也很重要"},
]

for round_no in range(1, 4):
    print(f"\n第 {round_no} 轮")
    print("=" * 72)

    for agent in agents:
        messages = [
            {
                "role": "system",
                "content": (
                    f"你是辩论赛{agent['name']}。你的立场：{agent['position']}。"
                    "请针对当前公开历史发言，控制在 120 字以内。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "辩题": topic,
                        "公开历史": public_history,
                        "任务": "给出本轮发言。可以回应对手，也可以补充自己的新论点。",
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        payload = {
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": 400,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        speech = response.json()["choices"][0]["message"]["content"]

        line = f"{agent['name']}：{speech}"
        public_history.append(line)
        print(line)

print("\n最终公开历史：")
print(json.dumps(public_history, ensure_ascii=False, indent=2))
