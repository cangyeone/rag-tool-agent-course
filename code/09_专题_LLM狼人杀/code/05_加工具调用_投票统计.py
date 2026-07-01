"""05 加工具调用：让模型选择工具，程序执行工具。

课堂目标：
1. 模型不直接算票，而是发起 tool call。
2. 程序拿到 tool_calls 后执行真实工具。
3. 工具结果再回传给模型，模型生成最终解释。
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

votes = {
    "A": "B",
    "B": "A",
    "C": "A",
    "D": "A",
    "E": "B",
}

messages = [
    {
        "role": "system",
        "content": "你是狼人杀主持人。需要统计投票时，请调用工具 count_votes。",
    },
    {
        "role": "user",
        "content": json.dumps(
            {
                "任务": "请统计投票并宣布出局玩家。",
                "votes": votes,
            },
            ensure_ascii=False,
        ),
    },
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "count_votes",
            "description": "统计狼人杀投票，返回每名玩家得票数和最高票玩家。",
            "parameters": {
                "type": "object",
                "properties": {
                    "votes": {
                        "type": "object",
                        "description": "投票字典，例如 {'A':'B','B':'A'}，key 是投票人，value 是被投票人。",
                    }
                },
                "required": ["votes"],
            },
        },
    }
]

payload = {
    "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    "messages": messages,
    "tools": tools,
    "tool_choice": "auto",
    "temperature": 0.2,
    "max_tokens": 600,
}

print("投票数据：")
print(json.dumps(votes, ensure_ascii=False, indent=2))
print("-" * 72)

response = requests.post(url, headers=headers, json=payload, timeout=60)
response.raise_for_status()

assistant_message = response.json()["choices"][0]["message"]
print("模型返回的 assistant message：")
print(json.dumps(assistant_message, ensure_ascii=False, indent=2))

messages.append(assistant_message)

tool_calls = assistant_message.get("tool_calls") or []
if not tool_calls:
    print("\n模型没有发起 tool call。课堂上可以重新运行，或把 tool_choice 改成 required。")
    raise SystemExit

for tool_call in tool_calls:
    tool_name = tool_call["function"]["name"]
    tool_args = json.loads(tool_call["function"]["arguments"])

    if tool_name == "count_votes":
        vote_data = tool_args["votes"]
        counter = Counter(vote_data.values())
        top_player = counter.most_common(1)[0][0]
        tool_result = {
            "vote_count": dict(counter),
            "out_player": top_player,
            "raw_votes": vote_data,
        }
    else:
        tool_result = {"error": f"未知工具：{tool_name}"}

    print("\n程序执行工具：", tool_name)
    print(json.dumps(tool_result, ensure_ascii=False, indent=2))

    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": json.dumps(tool_result, ensure_ascii=False),
        }
    )

payload = {
    "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    "messages": messages,
    "temperature": 0.2,
    "max_tokens": 500,
}

response = requests.post(url, headers=headers, json=payload, timeout=60)
response.raise_for_status()

print("\n模型基于工具结果生成最终宣布：")
print(response.json()["choices"][0]["message"]["content"])
