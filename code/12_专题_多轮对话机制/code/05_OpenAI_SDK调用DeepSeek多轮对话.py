"""05 使用 OpenAI SDK 调用 DeepSeek 多轮对话。

DeepSeek 的 /chat/completions 接口兼容 OpenAI SDK。
写法上主要改两处：
1. api_key 使用 DEEPSEEK_API_KEY。
2. base_url 改成 https://api.deepseek.com。

运行方式：
    cd rag-tool-agent-course
    export DEEPSEEK_API_KEY=your_api_key_here
    python code/12_专题_多轮对话机制/code/05_OpenAI_SDK调用DeepSeek多轮对话.py
"""

from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("05 OpenAI SDK 调用 DeepSeek 多轮对话")
print("=" * 72)

api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
if not api_key:
    raise SystemExit(
        "未设置 DEEPSEEK_API_KEY。\n"
        "macOS/Linux: export DEEPSEEK_API_KEY=your_api_key_here\n"
        "Windows PowerShell: $env:DEEPSEEK_API_KEY=\"sk-xxx\""
    )

client = OpenAI(
    api_key=api_key,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

messages = [
    {
        "role": "system",
        "content": "你是一个课堂助手，回答要短、清楚、适合代码演示。",
    }
]

questions = [
    "请解释 API 为什么是无状态的。",
    "那无状态 API 怎么实现多轮对话？",
]

for round_index, question in enumerate(questions, start=1):
    print(f"\n{'=' * 24} 第 {round_index} 轮 {'=' * 24}")
    print("用户：", question)

    messages.append({"role": "user", "content": question})

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=700,
        extra_body={"thinking": {"type": "disabled"}},
    )

    assistant_message = completion.choices[0].message
    content = assistant_message.content or ""

    print("\n模型回答：")
    print(content)

    # OpenAI SDK 返回的是对象，不是普通 dict。
    # 这里手动追加 dict，课堂上更容易看懂 messages 的形状。
    messages.append({"role": "assistant", "content": content})

    print("\n下一轮会继续传入的消息数量：", len(messages))

print("\n结论")
print("DeepSeek 可以用 OpenAI SDK 调。")
print("多轮对话仍然靠 messages，SDK 不会自动替你保存历史。")
