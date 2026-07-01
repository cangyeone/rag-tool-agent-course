"""01 调 API：最小 chat 示例。

课堂目标：
1. 看懂 chat messages 的基本形状。
2. 看懂 system / user / assistant 分别放什么。
3. 确认 DeepSeek API 能正常调用。

运行前设置：
本目录已经提供本地演示用 .env。脚本会先读取系统环境变量，
如果没有设置环境变量，再读取同目录 .env。
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

# system：设定模型回答边界。
# user：当前用户问题。
messages = [
    {"role": "system", "content": "你是一个课堂演示助手，回答要简洁，适合初学者理解。"},
    {"role": "user", "content": "用三句话解释：为什么狼人杀适合演示 LLM Agent？"},
]

payload = {
    "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    "messages": messages,
    "temperature": 0.3,
    "max_tokens": 400,
}

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

print("发送给模型的 messages：")
print(json.dumps(messages, ensure_ascii=False, indent=2))
print("-" * 72)

response = requests.post(url, headers=headers, json=payload, timeout=60)
print("HTTP 状态码：", response.status_code)
response.raise_for_status()

data = response.json()
answer = data["choices"][0]["message"]["content"]

print("\n模型回答：")
print(answer)