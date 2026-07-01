"""最简版 示例业务系统 问答机器人 —— 基于本地 Qwen3.5-9B，支持 Markdown 渲染。

运行方式：python qa_bot.py
退出方式：输入 quit / exit / q
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    console = Console()
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

COURSE_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = COURSE_ROOT / "open_models" / "Qwen3.5-9B"
if not MODEL_PATH.is_dir():
    raise SystemExit(f"未找到模型：{MODEL_PATH}")

# ── 加载模型 ──
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if device != "cpu" else torch.float32
print(f"加载 Qwen3.5-9B（device={device}, dtype={dtype}）...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), torch_dtype=dtype).to(device).eval()
print(f"加载完成，参数量约 {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
print()

SYSTEM_PROMPT = "你是 通用客服助手，回答简洁专业。/no_think"

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

MAX_TOKENS = int(os.getenv("QA_MAX_TOKENS", "2048"))
TEMPERATURE = float(os.getenv("QA_TEMPERATURE", "0.3"))

MARKDOWN_MODE = os.getenv("QA_MARKDOWN", "1") == "1"

if _HAS_RICH:
    console.print(Panel.fit(
        f"[bold]示例业务系统 问答机器人[/bold]  Qwen3.5-9B  |  max_tokens={MAX_TOKENS}  temperature={TEMPERATURE}",
        border_style="cyan",
    ))
    console.print("[dim]输入 quit/exit/q 退出  |  clear 清空历史  |  QA_MARKDOWN=0 关闭渲染[/dim]\n")
else:
    print("=" * 60)
    print("  示例业务系统 问答机器人 (Qwen3.5-9B)")
    print(f"  max_tokens={MAX_TOKENS}  temperature={TEMPERATURE}")
    print("  输入 quit / exit / q 退出")
    print("  输入 clear 清空对话历史")
    print("=" * 60)
    print()

while True:
    try:
        user_input = input("🧑 你：").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n再见！")
        break

    if not user_input:
        continue

    if user_input.lower() in ("quit", "exit", "q"):
        print("再见！")
        break

    if user_input.lower() == "clear":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if _HAS_RICH:
            console.print("[green]✓ 对话历史已清空[/green]\n")
        else:
            print("✓ 对话历史已清空\n")
        continue

    messages.append({"role": "user", "content": user_input})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            do_sample=TEMPERATURE > 0,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    messages.append({"role": "assistant", "content": answer})

    if _HAS_RICH and MARKDOWN_MODE:
        console.print(Markdown(answer))
    else:
        print(f"🤖 客服：{answer}")

    print()