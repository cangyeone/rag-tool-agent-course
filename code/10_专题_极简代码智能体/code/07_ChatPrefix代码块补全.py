"""07 Chat Prefix：用代码块前缀约束模型输出代码。

这个接口适合演示一个细节：
如果希望模型只输出 Python 代码，可以先给 assistant 一个前缀：

    ```python

然后让模型继续补全。再用 stop=["```"] 让模型遇到代码块结束标记就停。

这种方式不是“真正执行代码”，只是让输出格式更稳定。
它适合课堂演示、代码生成器、网页端代码片段生成。
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
    env_file = COURSE_ROOT / "code/10_专题_极简代码智能体/code/.env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("DEEPSEEK_API_KEY="):
                api_key = line.strip().split("=", 1)[1]

if not api_key:
    raise SystemExit("请设置 DEEPSEEK_API_KEY，或复制仓库根目录 .env.example 为 .env 后填写。")

beta_base_url = os.getenv("DEEPSEEK_BETA_BASE_URL", "https://api.deepseek.com/beta").rstrip("/")
url = beta_base_url + "/chat/completions"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
model = os.getenv("DEEPSEEK_CODE_MODEL", "deepseek-v4-pro")
user_id = os.getenv("DEEPSEEK_USER_ID", os.getenv("USER", "classroom_user"))

messages = [
    {
        "role": "user",
        "content": (
            "写一个 Python 函数 classify_order_question(text)，"
            "根据文本判断问题属于 refund、change、seat、other 四类之一。"
            "只写函数和 4 行简单测试。"
        ),
    },
    {
        "role": "assistant",
        "content": "```python\n",
        "prefix": True,
    },
]

payload = {
    "model": model,
    "messages": messages,
    "thinking": {"type": "disabled"},
    "temperature": 0.1,
    "max_tokens": 800,
    "stop": ["```"],
    "user_id": user_id,
}

print("Chat Prefix 请求体：")
print(json.dumps(payload, ensure_ascii=False, indent=2))
print("-" * 72)

response = requests.post(url, headers=headers, json=payload, timeout=60)
print("HTTP 状态码：", response.status_code)
if response.status_code != 200:
    print(response.text)
    raise SystemExit("Chat Prefix 请求失败。")

data = response.json()
code = data["choices"][0]["message"]["content"]

print("模型续写的代码：")
print("```python")
print(code.rstrip())
print("```")

print("\n说明：")
print("1. assistant 的 prefix=True 表示这条 assistant 消息只是开头，不是完整回答。")
print("2. content='```python\\n' 会把输出引向 Python 代码块。")
print("3. stop=['```'] 可以避免模型继续输出解释文字。")