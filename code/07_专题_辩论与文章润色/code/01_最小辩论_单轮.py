"""01 最小辩论：单轮正反方发言。

课堂目标：
1. 用 system message 做角色扮演。
2. 正方和反方看到同一个辩题，但立场不同。
3. 输出不做复杂封装，方便初学者逐行看。
"""

import os
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

for side in ["正方", "反方"]:
    messages = [
        {
            "role": "system",
            "content": f"你是辩论赛的{side}一辩。发言要清楚、有论点、有例子，控制在 150 字以内。",
        },
        {
            "role": "user",
            "content": f"辩题：{topic}\n请给出你的开篇陈词。",
        },
    ]

    payload = {
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 400,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"]

    print("=" * 72)
    print(side)
    print(answer)
