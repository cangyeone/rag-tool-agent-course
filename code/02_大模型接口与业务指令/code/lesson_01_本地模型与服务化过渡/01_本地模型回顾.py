"""
学习目标：快速回顾本地模型的三个核心步骤——分词、生成、解码。
不是 Ch01 的重复，而是从"服务化过渡"的视角重新审视本地模型做了什么。
"""


import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("01 本地模型回顾 —— 三个核心动作")
print("=" * 72)


def _find_qwen_path():
    candidate = COURSE_ROOT / "open_models" / "Qwen3.5-0.8B"
    if candidate.is_dir():
        return str(candidate)
    raise FileNotFoundError("未找到 open_models/Qwen3.5-0.8B。请确认已经从 rag-tool-agent-course 根目录运行，并且模型已下载。")


QWEN_PATH = _find_qwen_path()
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device != "cpu" else torch.float32
print(f"加载 Qwen3.5-0.8B（device={device}）...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH)
model = AutoModelForCausalLM.from_pretrained(QWEN_PATH, torch_dtype=dtype).to(device).eval()

# ── 步骤 1：分词（Tokenize）──
text = "北京南到上海虹桥的候补申请怎么解释？"
tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text, add_special_tokens=False)

print("\n── 步骤 1：分词（Tokenize）──")
print("输入文本：", text)
print("分词结果（tokens）：", tokens)
print("映射 ID：", ids)
print(f"词表大小：{tokenizer.vocab_size}")
print("→ 本地模型的第一步就是把文字切成 token ID，这是所有模型推理的入口。")

# ── 步骤 2：生成（Generate）──
print("\n── 步骤 2：逐 token 生成 ──")
messages = [{"role": "user", "content": text}]
rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(rendered, return_tensors="pt").to(device)
prompt_len = inputs["input_ids"].shape[1]
print(f"Prompt token 数：{prompt_len}")

with torch.no_grad():
    output_ids = model.generate(
        **inputs, max_new_tokens=50, temperature=0.2, do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True, output_scores=True,
    )

generated_ids = output_ids.sequences[0][prompt_len:]
# 展示每一步选中的 token
print("逐步生成过程：")
for i, (gid, scores) in enumerate(zip(generated_ids, output_ids.scores)):
    token_str = tokenizer.decode([gid])
    top3_indices = scores.topk(3).indices.tolist()
    top3_tokens = [tokenizer.decode([t]) for t in top3_indices]
    print(f"  step {i + 1}: 选中 {token_str!r} (id={gid.item()}), top3候选: {top3_tokens}")

# ── 步骤 3：解码（Decode）──
print("\n── 步骤 3：解码 —— 把 token ID 变回文字 ──")
decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
print(f"生成的 token ID 序列：{generated_ids.tolist()}")
print(f"解码结果：{decoded}")

print("\n── 小结 ──")
print("本地模型 = tokenize → generate → decode")
print("这三步全在你自己的机器上运行，不需要网络，不依赖外部服务。")
print("以上输出由本地 Qwen3.5-0.8B 真实生成。")
print("下一节：为什么本地模型在生产环境中不够用？")