"""
06 DeepSeek API 参数控制。

这一节在 05 的最简调用上继续往前走：
1. 加入 system / user 的对话格式
2. 加入 temperature、top_p、max_tokens
3. 对比不同参数对输出风格的影响
4. 继续保持密钥从环境变量读取
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

print("06 DeepSeek API 参数控制")
print("=" * 72)

API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
URL = BASE_URL + "/chat/completions"

print("接口地址：", URL)
print("模型名称：", MODEL)
print("密钥状态：", "已设置（不打印明文）" if API_KEY else "未设置")

question = "用户问：退款费是怎么计算的？请给出客服辅助回答。"

messages = [
    {
        "role": "system",
        "content": (
            "你是客服辅助助手。回答要稳妥、简洁；"
            "涉及规则、费用和时效时，提醒以官方页面实际显示为准。"
        ),
    },
    {
        "role": "user",
        "content": question,
    },
]

print("\n同一个问题：")
print(question)

print("\nmessages：")
print(json.dumps(messages, ensure_ascii=False, indent=2))

# 参数说明：
# temperature 越低，回答越稳定；越高，表达越发散。
# top_p 控制候选词范围，通常和 temperature 只重点调一个。
# max_tokens 控制最大输出长度，避免回答过长或费用不可控。
# stream=False 表示一次性返回完整答案。
payload_stable = {
    "model": MODEL,
    "messages": messages,
    "thinking": {"type": "disabled"},
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 350,
    "stream": False,
}

payload_open = {
    "model": MODEL,
    "messages": messages,
    "thinking": {"type": "disabled"},
    "temperature": 0.9,
    "top_p": 0.95,
    "max_tokens": 350,
    "stream": False,
}

payload_short = {
    "model": MODEL,
    "messages": messages,
    "thinking": {"type": "disabled"},
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 80,
    "stream": False,
}

payloads = [
    ("低 temperature：更稳", payload_stable),
    ("高 temperature：表达更发散", payload_open),
    ("小 max_tokens：强制短回答", payload_short),
]

if not API_KEY:
    print("\n未设置 DEEPSEEK_API_KEY，只打印三组请求体。")
    for title, payload in payloads:
        print("\n" + "-" * 60)
        print(title)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
else:
    for title, payload in payloads:
        print("\n" + "-" * 60)
        print(title)
        print("请求参数：")
        print(json.dumps(
            {
                "temperature": payload["temperature"],
                "top_p": payload["top_p"],
                "max_tokens": payload["max_tokens"],
                "thinking": payload["thinking"],
            },
            ensure_ascii=False,
            indent=2,
        ))

        start = time.time()
        response = requests.post(
            URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        elapsed = time.time() - start

        print("HTTP 状态码：", response.status_code, f"耗时：{elapsed:.2f}s")

        if response.status_code != 200:
            print("请求失败：")
            print(response.text[:1200])
            continue

        data = response.json()
        answer = data["choices"][0]["message"].get("content", "")
        usage = data.get("usage", {})

        print("\n模型回答：")
        print(answer)

        print("\nToken 用量：")
        print(
            "prompt={prompt}, completion={completion}, total={total}".format(
                prompt=usage.get("prompt_tokens"),
                completion=usage.get("completion_tokens"),
                total=usage.get("total_tokens"),
            )
        )

print("\n本节要点：")
print("1. system 用来设定回答边界，user 是真实问题。")
print("2. temperature 影响稳定性，max_tokens 影响输出长度。")
print("3. 普通业务问答通常使用 thinking disabled，先把输出做稳。")