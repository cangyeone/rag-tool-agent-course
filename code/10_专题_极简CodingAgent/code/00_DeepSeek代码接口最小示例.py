"""00 DeepSeek 代码接口最小示例。

这个脚本放在 Coding Agent 之前。
先看清楚一件事：模型本身不是编辑器，也不是 Python 解释器。
它通过 API 收到一段文本，再返回一段文本。

代码场景里常见两种接口：
1. Chat Completions：让模型生成完整代码、解释代码、修复代码。
2. FIM Completion：给模型代码前缀和后缀，让它补中间缺失的代码。
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
    env_file = COURSE_ROOT / "code/10_专题_极简CodingAgent/code/.env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("DEEPSEEK_API_KEY="):
                api_key = line.strip().split("=", 1)[1]

if not api_key:
    raise SystemExit("请设置 DEEPSEEK_API_KEY，或复制仓库根目录 .env.example 为 .env 后填写。")

user_id = os.getenv("DEEPSEEK_USER_ID", os.getenv("USER", "classroom_user"))
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

print("00 DeepSeek 代码接口最小示例")
print("=" * 72)
print("user_id:", user_id)

# ------------------------------------------------------------
# 一、Chat Completions：生成完整代码
# ------------------------------------------------------------

base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
chat_url = base_url + "/chat/completions"

chat_payload = {
    "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    "messages": [
        {
            "role": "system",
            "content": (
                "你是一个代码助手。只输出可直接运行的 Python 代码。"
                "不要输出 Markdown 代码块，不要输出解释文字。"
            ),
        },
        {
            "role": "user",
            "content": (
                "写一个函数 count_words(text)，统计字符串中每个英文单词出现次数。"
                "要求忽略大小写，去掉常见标点，并在文件底部写 2 行测试打印。"
            ),
        },
    ],
    "thinking": {"type": "disabled"},
    "temperature": 0.1,
    "max_tokens": 800,
    "user_id": user_id,
}

print("\n一、Chat Completions：生成完整代码")
print("-" * 72)
print("请求地址:", chat_url)
print("请求体:")
print(json.dumps(chat_payload, ensure_ascii=False, indent=2))

chat_response = requests.post(chat_url, headers=headers, json=chat_payload, timeout=60)
print("\nHTTP 状态码:", chat_response.status_code)
if chat_response.status_code != 200:
    print(chat_response.text)
    raise SystemExit("Chat Completions 请求失败。")

chat_data = chat_response.json()
chat_code = chat_data["choices"][0]["message"]["content"].strip()

print("\n模型返回的代码:")
print("```python")
print(chat_code)
print("```")

# ------------------------------------------------------------
# 二、FIM Completion：补全代码中间部分
# ------------------------------------------------------------

beta_base_url = os.getenv("DEEPSEEK_BETA_BASE_URL", "https://api.deepseek.com/beta").rstrip("/")
fim_url = beta_base_url + "/completions"

prefix = """def safe_divide(a, b):
    \"\"\"安全除法：b 为 0 时返回 None。\"\"\"
"""

suffix = """

print(safe_divide(10, 2))
print(safe_divide(10, 0))
"""

fim_payload = {
    "model": os.getenv("DEEPSEEK_CODE_MODEL", "deepseek-v4-pro"),
    "prompt": prefix,
    "suffix": suffix,
    "temperature": 0.1,
    "max_tokens": 200,
}

print("\n二、FIM Completion：补全代码中间部分")
print("-" * 72)
print("请求地址:", fim_url)
print("prefix 是光标前面的代码，suffix 是光标后面的代码。")
print("请求体:")
print(json.dumps(fim_payload, ensure_ascii=False, indent=2))

fim_response = requests.post(fim_url, headers=headers, json=fim_payload, timeout=60)
print("\nHTTP 状态码:", fim_response.status_code)
if fim_response.status_code != 200:
    print(fim_response.text)
    raise SystemExit("FIM Completion 请求失败。")

fim_data = fim_response.json()
middle_code = fim_data["choices"][0]["text"]
full_code = prefix + middle_code + suffix

print("\n模型补出的中间代码:")
print("```python")
print(middle_code.rstrip())
print("```")

print("\n拼接后的完整代码:")
print("```python")
print(full_code.rstrip())
print("```")

print("\n课堂观察点:")
print("1. Chat Completions 更适合生成完整文件、解释代码、修复 bug。")
print("2. FIM Completion 更像 IDE 里的补全：已有前后文，让模型补光标位置。")
print("3. 写入 .py 文件时保存纯代码；展示给人看时再加 ```python 标记。")