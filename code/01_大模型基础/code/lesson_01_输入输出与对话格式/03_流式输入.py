"""03 基于Qwen模型的流式输入。

学习目标：理解大模型的流式（streaming）输出机制，观察模型如何逐 token 生成文本，体验流式输出相比一次性返回的差异。

运行方式：python 03_流式输入.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import time
import torch
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

print("03 基于Qwen模型的流式输入")
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

# ========== 一、一次性输出（非流式）==========
print("\n" + "=" * 60)
print("一、一次性输出 —— 模型生成完再统一返回")
print("=" * 60)

question = "候补申请的规则是什么？用一句话回答。"
messages = [
    {"role": "system", "content": "你是 通用客服助手，回答要简洁专业。"},
    {"role": "user", "content": question},
]
rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(rendered, return_tensors="pt").to(device)
prompt_len = inputs["input_ids"].shape[1]

print(f"用户问题：{question}")
print(f"Prompt token 数：{prompt_len}")
print(f"\n一次性输出结果：")

start = time.time()
with torch.no_grad():
    output_ids = model.generate(
        **inputs, max_new_tokens=80, temperature=0.3, do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
generated = output_ids[0][prompt_len:]
answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
elapsed = time.time() - start
print(f"  {answer}")
print(f"  生成 token 数：{len(generated)}，耗时：{elapsed:.1f}s")
print(f"  特点：用户需要等待全部 token 生成完毕才能看到结果")

# ========== 二、TextStreamer 流式输出 ==========
print("\n" + "=" * 60)
print("二、TextStreamer 流式输出 —— 使用 HuggingFace 内置流式器")
print("=" * 60)

question2 = "ORD-1001 没票了，还能怎么办？"
messages2 = [
    {"role": "system", "content": "你是 通用客服助手，回答要简洁专业。"},
    {"role": "user", "content": question2},
]
rendered2 = tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
inputs2 = tokenizer(rendered2, return_tensors="pt").to(device)

print(f"用户问题：{question2}")
print(f"\n流式输出效果（token 逐个打印）：")

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
start = time.time()
with torch.no_grad():
    model.generate(
        **inputs2, max_new_tokens=80, temperature=0.3, do_sample=True,
        pad_token_id=tokenizer.eos_token_id, streamer=streamer,
    )
elapsed_stream = time.time() - start
print(f"\n  流式耗时：{elapsed_stream:.1f}s")
print(f"  特点：token 逐个输出，用户能立刻看到模型在'打字'")

# ========== 三、手动逐 token 生成（展示内部机制）==========
print("\n" + "=" * 60)
print("三、手动逐 token 生成 —— 看清每次迭代选了哪个 token")
print("=" * 60)

question3 = "退款费用怎么算？用一句话回答。"
messages3 = [
    {"role": "system", "content": "你是 通用客服助手，回答要简洁专业。"},
    {"role": "user", "content": question3},
]
rendered3 = tokenizer.apply_chat_template(messages3, tokenize=False, add_generation_prompt=True)
inputs3 = tokenizer(rendered3, return_tensors="pt").to(device)
prompt_len3 = inputs3["input_ids"].shape[1]

print(f"用户问题：{question3}")
print(f"Prompt token 数：{prompt_len3}")
print(f"\n逐步生成过程（每次展示选中的 token 及其 top3 候选）：")

with torch.no_grad():
    output = model.generate(
        **inputs3, max_new_tokens=30, temperature=0.3, do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True, output_scores=True,
    )

generated_ids = output.sequences[0][prompt_len3:]
printed_text = ""
for i, (gid, scores) in enumerate(zip(generated_ids, output.scores)):
    token_str = tokenizer.decode([gid])
    printed_text += token_str
    top3_indices = scores[0].topk(3).indices.tolist()
    top3_tokens = [repr(tokenizer.decode([t])) for t in top3_indices]
    print(f"  step {i+1:2d}: 选中 {token_str!r:12s}  top3候选: {', '.join(top3_tokens)}")
    if gid.item() == tokenizer.eos_token_id:
        print(f"         → 生成结束（EOS token）")
        break

print(f"\n完整输出：{printed_text}")
print(f"生成 token 数：{len(generated_ids)}")

# ========== 四、流式对用户体验的意义 ==========
print("\n" + "=" * 60)
print("四、流式输出的实际意义")
print("=" * 60)
print("  • 用户感知延迟大幅降低：看到第一个 token 的时间远小于完整响应时间")
print("  • 打字机效果提升交互感：用户能感知模型在'思考'，不会怀疑卡住了")
print("  • 支持中途打断：在流式过程中可以检测不安全内容并提前截断")
print("  • 示例业务系统 客服场景：用户输入问题后立刻看到回复逐字出现，体验接近人工客服")
print()
print("  核心方法：")
print("    TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)")
print("    model.generate(..., streamer=streamer)")
print()
print("  本脚本使用本地 Qwen3.5-0.8B 实际推理，所有输出均为真实模型生成。")
print("  可以修改：更换 question 内容；调整 temperature 观察输出变化；对比流式与非流式的耗时差异。")