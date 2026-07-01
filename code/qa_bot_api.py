"""示例业务系统 问答机器人 API 版 —— DeepSeek 在线 API + rich TUI 界面。

运行方式：
  python qa_bot_api.py
退出方式：输入 quit / exit / q

前置 —— 设置 DEEPSEEK_API_KEY（三选一）：

  1. macOS / Linux 终端：
     export DEEPSEEK_API_KEY=your_api_key_here

  2. Windows PowerShell（当前会话）：
     $env:DEEPSEEK_API_KEY="YOUR_API_KEY_HERE"

     Windows PowerShell（永久生效）：
     [Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "sk-xxx", "User")

  3. 在 qa_bot_api.py 同目录创建 .env 文件，写入：
     DEEPSEEK_API_KEY=your_api_key_here
     （无需设置系统环境变量，脚本自动读取）

获取 Key：https://platform.deepseek.com/api_keys

可选环境变量：
  QA_USER_ID       用户标识，透传到 API 的 user 字段（用于用量追踪/审计）
  DEEPSEEK_MODEL   模型选择（默认 deepseek-chat，可选 deepseek-reasoner）
  QA_MAX_TOKENS    最大输出 token 数（默认 1024）
  QA_TEMPERATURE   采样温度 0~2（默认 0.3）
"""

import os
import json
import requests
from pathlib import Path

# 尝试从 .env 文件读取 API Key
_ENV_FILE = Path(__file__).resolve().parent / ".env"
if _ENV_FILE.exists():
    with open(_ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from rich.console import Console
from rich.panel import Panel

console = Console()

# ── 配置 ──
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
#API_KEY = os.getenv("BAILIAN_API_KEY", "").strip()
CHAT_URL = BASE_URL + "/v1/chat/completions"
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
MAX_TOKENS = int(os.getenv("QA_MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("QA_TEMPERATURE", "0.3"))
USER_ID = os.getenv("QA_USER_ID", "")
SYSTEM_PROMPT = "你是 通用客服助手，回答简洁专业、条理清晰。如涉及业务规则，请注明以官方页面为准。"

if not API_KEY:
    console.print(Panel.fit(
        "[bold red]未设置 DEEPSEEK_API_KEY[/bold red]\n\n"
        "方式一：export DEEPSEEK_API_KEY=your_api_key_here\n"
        "方式二：在 qa_bot_api.py 同目录创建 .env 文件，写入 DEEPSEEK_API_KEY=your_api_key_here\n"
        "获取地址：https://platform.deepseek.com/api_keys",
        title="错误",
        border_style="red",
    ))
    raise SystemExit(1)

# ── API 调用 ──

def chat_once(messages, stream=True):
    """调用 DeepSeek API，返回生成器（逐 chunk 产出文本）。"""
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "top_p": 0.9,
        "stream": stream,
    }
    if USER_ID:
        payload["user"] = USER_ID
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    if not stream:
        resp = requests.post(CHAT_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        yield data["choices"][0]["message"]["content"].strip()
        return

    resp = requests.post(CHAT_URL, headers=headers, json=payload, timeout=300, stream=True)
    resp.raise_for_status()

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            if chunk.get("choices"):
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("content"):
                    yield delta["content"]
        except json.JSONDecodeError:
            pass

# ── TUI 主循环 ──

def main():
    global MODEL, TEMPERATURE

    user_info = f"  user_id: {USER_ID}\n" if USER_ID else ""
    console.print(Panel.fit(
        f"[bold]示例业务系统 问答机器人[/bold]\n"
        f"后端: DeepSeek  |  模型: {MODEL}\n"
        f"{user_info}"
        f"temperature={TEMPERATURE}  max_tokens={MAX_TOKENS}",
        border_style="cyan",
    ))
    console.print("[dim]输入 quit/exit/q 退出  |  clear 清空历史  |  /model 切换模型  |  /temp 调温度[/dim]\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("🧑 你：").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]再见！[/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]再见！[/dim]")
            break

        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            console.print("[green]✓ 对话历史已清空[/green]\n")
            continue

        # / 命令
        if user_input.startswith("/model"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                new_model = parts[1].strip()
                if new_model in ("deepseek-chat", "deepseek-reasoner"):
                    MODEL = new_model
                    console.print(f"[green]✓ 已切换模型: {MODEL}[/green]\n")
                else:
                    console.print(f"[red]未知模型: {new_model}，可选: deepseek-chat / deepseek-reasoner[/red]\n")
            else:
                console.print(f"当前模型: {MODEL}\n")
            continue

        if user_input.startswith("/temp"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                try:
                    t = float(parts[1].strip())
                    if 0 <= t <= 2:
                        TEMPERATURE = t
                        console.print(f"[green]✓ temperature = {TEMPERATURE}[/green]\n")
                    else:
                        console.print("[red]temperature 范围 0~2[/red]\n")
                except ValueError:
                    console.print("[red]temperature 必须是数字[/red]\n")
            else:
                console.print(f"当前 temperature: {TEMPERATURE}\n")
            continue

        # ── 发送请求 ──
        messages.append({"role": "user", "content": user_input})

        try:
            full_answer = ""
            console.print("🤖 客服：", end="")
            for piece in chat_once(messages, stream=True):
                full_answer += piece
                console.print(piece, end="")
            console.print()
            console.print()

            if full_answer:
                messages.append({"role": "assistant", "content": full_answer})

        except requests.Timeout:
            console.print("\n[red]请求超时，请重试[/red]\n")
        except requests.HTTPError as e:
            console.print(f"\n[red]API 错误: {e.response.status_code} {e.response.text[:200]}[/red]\n")
        except Exception as e:
            console.print(f"\n[red]异常: {type(e).__name__}: {e}[/red]\n")

if __name__ == "__main__":
    main()