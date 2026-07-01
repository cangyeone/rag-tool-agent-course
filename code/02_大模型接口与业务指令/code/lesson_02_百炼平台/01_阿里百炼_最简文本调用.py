"""
01 阿里百炼最简文本调用。

只保留最少字段，先把接口调通：
1. API_BASE：阿里百炼 OpenAI 兼容地址
2. MODEL：本节使用 qwen3.7-plus
3. messages：用户问题
4. stream=False：一次性返回完整结果

运行方式：
    cd rag-tool-agent-course
    python code/02_大模型接口与业务指令/code/lesson_02_百炼平台/01_阿里百炼_最简文本调用.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("01 阿里百炼最简文本调用")
print("=" * 72)

API_BASE = os.getenv("BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("BAILIAN_MODEL", "qwen-plus")

# API Key 从环境变量读取，公开仓库不要写入真实 Key。
API_KEY = os.getenv("BAILIAN_API_KEY", "").strip()

URL = API_BASE.rstrip("/") + "/chat/completions"

question = "请用三句话说明：大模型接口和本地 transformers 调用有什么区别？"

payload = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": question},
    ],
    "stream": False,
}

print("接口地址：", URL)
print("模型名称：", MODEL)
print("密钥状态：", "已填写（不打印明文）" if API_KEY else "未填写")
print("\n请求体：")
print(json.dumps(payload, ensure_ascii=False, indent=2))

if not API_KEY:
    print("\nAPI_KEY 为空，本次只展示请求体，不发送真实请求。")
    print("课堂演示时，把百炼 API Key 填入脚本顶部的 API_KEY 变量即可。")
    raise SystemExit(0)

response = requests.post(
    URL,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json=payload,
    timeout=90,
)

print("\nHTTP 状态码：", response.status_code)

if response.status_code != 200:
    print("请求失败：")
    print(response.text[:1600])
    raise SystemExit(1)

data = response.json()
answer = data["choices"][0]["message"].get("content", "")

print("\n模型回答：")
print(answer)

print("\nToken 用量：")
print(json.dumps(data.get("usage", {}), ensure_ascii=False, indent=2))

print("\n本节要点：")
print("1. OpenAI 兼容接口本质上就是 HTTP POST。")
print("2. 文本问答最核心的是 model 和 messages。")
print("3. 返回内容一般在 choices[0].message.content 中。")