"""
02 阿里百炼参数控制。

本节在最简调用基础上加入常用参数：
1. system：设定回答边界
2. temperature：控制随机性
3. top_p：控制候选范围
4. max_tokens：控制最大输出长度
5. stream：是否流式输出
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("02 阿里百炼参数控制")
print("=" * 72)

API_BASE = os.getenv("BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("BAILIAN_MODEL", "qwen-plus")
API_KEY = os.getenv("BAILIAN_API_KEY", "").strip()
URL = API_BASE.rstrip("/") + "/chat/completions"

messages = [
    {
        "role": "system",
        "content": (
            "你是企业内训课堂的模型接口助教。"
            "回答要具体、简洁，尽量使用短句。"
        ),
    },
    {
        "role": "user",
        "content": "请解释 temperature 和 max_tokens 对模型回答有什么影响。",
    },
]

payloads = [
    (
        "稳定回答：temperature 较低",
        {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 350,
            "stream": False,
        },
    ),
    (
        "表达更发散：temperature 较高",
        {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.9,
            "top_p": 0.95,
            "max_tokens": 350,
            "stream": False,
        },
    ),
    (
        "强制短回答：max_tokens 较小",
        {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 80,
            "stream": False,
        },
    ),
]

print("接口地址：", URL)
print("模型名称：", MODEL)
print("密钥状态：", "已填写（不打印明文）" if API_KEY else "未填写")

if not API_KEY:
    print("\nAPI_KEY 为空，本次只展示三组请求体。")

for title, payload in payloads:
    print("\n" + "-" * 60)
    print(title)
    print("请求体：")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if not API_KEY:
        continue

    start = time.time()
    response = requests.post(
        URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=90,
    )
    elapsed = time.time() - start

    print("HTTP 状态码：", response.status_code, f"耗时：{elapsed:.2f}s")

    if response.status_code != 200:
        print("请求失败：")
        print(response.text[:1600])
        continue

    data = response.json()
    answer = data["choices"][0]["message"].get("content", "")
    usage = data.get("usage", {})

    print("\n模型回答：")
    print(answer)
    print("\nToken 用量：")
    print(json.dumps(usage, ensure_ascii=False, indent=2))

print("\n参数说明：")
print("1. system 写得越清楚，模型越容易稳定在指定边界内。")
print("2. temperature 越低，回答越稳；越高，表达越灵活。")
print("3. max_tokens 是输出上限，不是目标长度。")
print("4. stream=False 适合脚本演示；stream=True 适合网页逐字显示。")