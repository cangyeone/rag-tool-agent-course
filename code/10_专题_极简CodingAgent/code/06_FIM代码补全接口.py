"""06 DeepSeek FIM 代码补全接口。

FIM 是 Fill In the Middle，意思是“补中间”。
普通 Chat API 更适合做：解释代码、生成完整文件、修复 bug、写测试。
FIM 更适合做：已有前缀和后缀，让模型补中间缺失的一段代码。

DeepSeek 的 FIM 仍是 Beta 接口：
1. base_url 使用 https://api.deepseek.com/beta
2. endpoint 是 /completions
3. model 使用 deepseek-v4-pro
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

beta_base_url = os.getenv("DEEPSEEK_BETA_BASE_URL", "https://api.deepseek.com/beta").rstrip("/")
url = beta_base_url + "/completions"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

# prefix 是光标前面的代码。
# suffix 是光标后面的代码。
# 模型要做的事情：补出 prefix 和 suffix 中间缺失的部分。
prefix = """def normalize_score(score):
    \"\"\"把 0-100 的分数转换到 0-1 区间。\"\"\"
"""

suffix = """

print(normalize_score(85))
print(normalize_score(-10))
print(normalize_score(120))
"""

payload = {
    "model": os.getenv("DEEPSEEK_CODE_MODEL", "deepseek-v4-pro"),
    "prompt": prefix,
    "suffix": suffix,
    "max_tokens": 256,
    "temperature": 0.1,
    "stop": ["\nprint("],
}

print("FIM 请求体：")
print(json.dumps({**payload, "prompt": prefix, "suffix": suffix}, ensure_ascii=False, indent=2))
print("-" * 72)

response = requests.post(url, headers=headers, json=payload, timeout=60)
print("HTTP 状态码：", response.status_code)
if response.status_code != 200:
    print(response.text)
    raise SystemExit("FIM 请求失败。")

data = response.json()
middle_code = data["choices"][0]["text"]

print("模型补出的中间代码：")
print("```python")
print(middle_code.rstrip())
print("```")

full_code = prefix + middle_code + suffix

print("\n拼接后的完整代码：")
print("```python")
print(full_code.rstrip())
print("```")

print("\n说明：")
print("普通 Chat API 是让模型回答一段话；FIM 是把光标前后代码交给模型，让它补光标位置。")
print("IDE 里的代码补全、函数体补全、缺失分支补全，都更接近这个接口。")