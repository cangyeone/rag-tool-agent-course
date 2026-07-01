"""
05 DeepSeek API 最简调用。

这一节只做一件事：用最少的代码把 DeepSeek 调起来。

运行方式：
    先进入 rag-tool-agent-course 课程根目录，再运行：
    python code/02_模型接口与指令设计/code/lesson_01_本地模型与服务化过渡/05_从transformers到API.py

密钥设置：
    macOS / Linux:
        export DEEPSEEK_API_KEY=your_api_key_here

    Windows PowerShell:
        $env:DEEPSEEK_API_KEY="your_api_key_here"
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("05 DeepSeek API 最简调用")
print("=" * 72)

# DeepSeek 当前推荐的 OpenAI 兼容接口：
#   base_url: https://api.deepseek.com
#   path    : /chat/completions
#
# 模型名推荐使用：
#   deepseek-v4-flash：默认课堂演示，速度快
#   deepseek-v4-pro  ：更强，适合复杂任务
#
# deepseek-v4-flash / deepseek-v4-pro 是兼容旧名，后续会弃用。

API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
URL = BASE_URL + "/chat/completions"

print("接口地址：", URL)
print("模型名称：", MODEL)
print("密钥状态：", "已设置（不打印明文）" if API_KEY else "未设置")

if not API_KEY:
    print("\n未设置 DEEPSEEK_API_KEY，下面只打印请求体，不发送真实请求。")
    print("macOS / Linux：export DEEPSEEK_API_KEY=your_api_key_here")
    print('Windows PowerShell：$env:DEEPSEEK_API_KEY="your_api_key_here"')

question = "候补申请为什么不能保证成功？请用三句话说明。"

# 最简请求体只有三个关键字段：
# 1. model：调用哪个模型
# 2. messages：对话内容
# 3. stream：是否流式输出
payload = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": question},
    ],
    "thinking": {"type": "disabled"},
    "stream": False,
}

print("\n请求体：")
print(json.dumps(payload, ensure_ascii=False, indent=2))

if API_KEY:
    response = requests.post(
        URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    print("\nHTTP 状态码：", response.status_code)

    if response.status_code != 200:
        print("请求失败，返回内容：")
        print(response.text[:1200])
        raise SystemExit(1)

    data = response.json()
    message = data["choices"][0]["message"]
    answer = message.get("content", "")

    print("\n模型回答：")
    print(answer)

    print("\n用量信息：")
    print(json.dumps(data.get("usage", {}), ensure_ascii=False, indent=2))

print("\n本节要点：")
print("1. 云端模型调用的本质是一次 HTTP POST 请求。")
print("2. messages 是对话内容，最少可以只有一条 user 消息。")
print("3. thinking 设为 disabled，表示普通回答模式。")