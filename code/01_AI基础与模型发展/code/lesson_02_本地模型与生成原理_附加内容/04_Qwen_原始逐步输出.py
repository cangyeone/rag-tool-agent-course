"""04 Qwen 原始逐步输出。

学习目标：不用 chat template 格式，直接给普通输入文本让模型生成，观察模型如何逐步（token 级）输出。理解 TextIteratorStreamer 的工作原理。

运行方式：python 04_Qwen_原始逐步输出.py
"""


import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

print("04 Qwen 原始逐步输出")
print("=" * 72)

model_dir = COURSE_ROOT / "open_models" / "Qwen3.5-0.8B"
if not model_dir.exists():
    model_dir = None
if model_dir is None:
    raise SystemExit("未找到 open_models/Qwen3.5-0.8B")

print("一、加载 tokenizer 和 model")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32
print(f"  设备：{device}")
print(f"  精度：{dtype}")
print("  正在加载模型权重...")
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype).to(device).eval()
param_count = sum(p.numel() for p in model.parameters()) / 1e9
print(f"  模型参数：{param_count:.2f} B")

input_text = "请用一句话解释：为什么 通用咨询助手不能承诺候补申请一定成功？"
print("\n二、原始输入文本")
print(f"  {input_text}")

print("\n三、非 chat template 模式 vs chat template 模式")
# 不加 chat template，模型看到的就是纯文本
inputs_raw = tokenizer(input_text, return_tensors="pt").to(device)
print(f"  原始模式 token 数：{inputs_raw['input_ids'].shape[1]}")

# 加 chat template，模型会看到带角色标记的格式化文本
messages = [{"role": "user", "content": input_text}]
rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs_chat = tokenizer(rendered, return_tensors="pt").to(device)
print(f"  Chat 模式 token 数：{inputs_chat['input_ids'].shape[1]}")
print(f"  Chat 模式增加 token 数：{inputs_chat['input_ids'].shape[1] - inputs_raw['input_ids'].shape[1]}（用于角色标记）")

print("\n四、原始模式逐步输出（非 chat template）")
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

gen_kwargs = {**inputs_raw, "streamer": streamer, "max_new_tokens": 80, "do_sample": False}
thread = Thread(target=model.generate, kwargs=gen_kwargs)
thread.start()
output_parts = []
print("  ", end="", flush=True)
for piece in streamer:
    print(piece, end="", flush=True)
    output_parts.append(piece)
thread.join()
print()

print("\n五、对比：Chat template 模式输出")
streamer2 = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
gen_kwargs2 = {**inputs_chat, "streamer": streamer2, "max_new_tokens": 80, "do_sample": False}
thread2 = Thread(target=model.generate, kwargs=gen_kwargs2)
thread2.start()
print("  ", end="", flush=True)
for piece in streamer2:
    print(piece, end="", flush=True)
thread2.join()
print()

print("\n要点：这里没有 messages，模型只看到一段普通输入文本。")
print("不加 chat template 时，模型靠文本中的自然语言暗示来理解任务；加了 chat template 后，模型通过特殊 token 明确知道自己的角色。")