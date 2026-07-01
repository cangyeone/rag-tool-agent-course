"""03 原始文本续写 —— 体验最原始的 Qwen。

学习目标：跳脱 Chat 模式的封装，用最原始的方式使用大模型——给它一段文本，看它预测下一个 token。理解大模型本质上是一个"文本续写引擎"，并观察逐字符输入时 token 如何动态变化。

运行方式：python 03_原始文本续写.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")
from transformers import AutoModelForCausalLM, AutoTokenizer

print("03 原始文本续写")
print("=" * 72)


def _find_qwen_path():
    candidate = COURSE_ROOT / "open_models" / "Qwen3.5-9B"
    if candidate.is_dir():
        return str(candidate)
    raise FileNotFoundError("未找到 open_models/Qwen3.5-9B。请确认已经从 rag-tool-agent-course 根目录运行，并且模型已下载。")


QWEN_PATH = _find_qwen_path()
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device != "cpu" else torch.float32
print(f"加载 Qwen3.5-0.8B（device={device}）...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH)
model = AutoModelForCausalLM.from_pretrained(QWEN_PATH, torch_dtype=dtype).to(device).eval()

# ========== 一、模型的最原始形态 —— 文本续写引擎 ==========
print("\n" + "=" * 60)
print("一、模型的最原始形态：一个'接话'引擎")
print("=" * 60)

print("  大模型本质上做一件事：给定前面的文本，预测下一个 token。")
print("  Chat 模式是对这个原始能力的封装。先看看去掉封装后的样子。")
print()

# 最简单的原始续写：给一句不完整的话，看模型怎么接
prompts = [
    f"<|im_start|>system\n你是一个翻译助手，需要把用户输入的所有信息都翻译成英文，不要解释，直接给结果。<|im_end|><|im_start|>user\n你是谁？<|im_end|><|im_start|>assistant\n <think>\n\n</think>", # 给定的提示词。
]


for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print(f"  输入：{inputs}")
    #break 
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=4096, temperature=0.0, do_sample=False,
            #top_p=0.8,
            #top_k=50, 
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][prompt_len:]
    continuation = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"  前半句：{prompt}")
    print(f"  模型续：{continuation}")
    print(f"  生成 token 数：{len(generated)}")
    print()