"""03 加入 Memory：公开记忆 + 私有记忆。

课堂目标：
1. public_memory：双方都能看到的辩论历史。
2. private_memory：每个辩手自己的策略记录。
3. 每轮发言后，模型同时输出 speech 和 new_private_memory。
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
    env_file = str(COURSE_ROOT / "code/11_专题_辩论与文章润色/code/.env")
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

public_memory = ["辩论开始。"]
private_memory = {
    "正方": ["核心策略：强调 RAG 接入知识快、可追溯、维护成本低。"],
    "反方": ["核心策略：强调微调能形成稳定风格和任务能力，RAG 不能解决所有问题。"],
}

for round_no in range(1, 3):
    print(f"\n第 {round_no} 轮")
    print("=" * 72)

    for side in ["正方", "反方"]:
        messages = [
            {
                "role": "system",
                "content": (
                    f"你是{side}辩手。必须输出 JSON，字段为 speech、new_private_memory。"
                    "speech 是本轮发言，new_private_memory 是你自己的策略记录。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "辩题": topic,
                        "公开记忆": public_memory,
                        "你的私有记忆": private_memory[side],
                        "任务": "结合记忆完成本轮发言。",
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        payload = {
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": 500,
            "response_format": {"type": "json_object"},
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = json.loads(response.json()["choices"][0]["message"]["content"])

        speech = data.get("speech", "")
        new_memory = data.get("new_private_memory", "")

        print(f"{side}：{speech}")
        public_memory.append(f"{side}：{speech}")
        if new_memory:
            private_memory[side].append(new_memory)

print("\n公开记忆：")
print(json.dumps(public_memory, ensure_ascii=False, indent=2))

print("\n私有记忆：")
print(json.dumps(private_memory, ensure_ascii=False, indent=2))
