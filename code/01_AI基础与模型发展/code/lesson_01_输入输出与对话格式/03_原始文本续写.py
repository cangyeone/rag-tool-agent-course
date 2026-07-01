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

# ========== 一、模型的最原始形态 —— 文本续写引擎 ==========
print("\n" + "=" * 60)
print("一、模型的最原始形态：一个'接话'引擎")
print("=" * 60)

print("  大模型本质上做一件事：给定前面的文本，预测下一个 token。")
print("  Chat 模式是对这个原始能力的封装。先看看去掉封装后的样子。")
print()

# 最简单的原始续写：给一句不完整的话，看模型怎么接
prompts = [
    "你", 
    "北京南到上海虹桥的",
    "候补申请是指当订单售罄时，",
    "退款手续费的计算规则是：服务开始前",
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=30, temperature=0.5, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][prompt_len:]
    continuation = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"  前半句：{prompt}")
    print(f"  模型续：{continuation}")
    print(f"  生成 token 数：{len(generated)}")
    print()

# ========== 二、逐字符输入观察 ==========
print("=" * 60)
print("二、逐字符输入 —— 每加一个字符，模型'看到'了什么")
print("=" * 60)

question = "G107没票了还能怎么办？"
print(f"\n完整问题：{question}")
print(f"共 {len(question)} 个字符\n")

accumulated = ""
prev_token_count = 0
for i, ch in enumerate(question, 1):
    accumulated += ch
    tokens = tokenizer.tokenize(accumulated)
    ids = tokenizer.encode(accumulated, add_special_tokens=False)
    new_tokens = tokens[prev_token_count:]
    new_ids = ids[prev_token_count:]

    new_token_str = " | ".join(f"{t!r}" for t in new_tokens) if new_tokens else "(无新增 token)"
    print(f"  第{i:02d}个字符 {ch!r} 加入后:")
    print(f"    累计文本：{accumulated}")
    print(f"    token 总数：{len(tokens):2d}，新增 token：{new_token_str}")
    print(f"    token ids：{ids}")
    prev_token_count = len(tokens)

print(f"\n  要点：前几个字符可能共用同一个 token（BPE 合并），")
print(f"       当汉字序列形成常见词时才会切出新的 token。")

# ========== 三、逐字符输入时的模型续写变化 ==========
print("\n" + "=" * 60)
print("三、逐字符输入时，模型'接话'内容的变化")
print("=" * 60)

question3 = "候补申请为什么"
print(f"\n逐步输入'{question3}'，观察模型续写如何逐渐收敛：")

accumulated3 = ""
for i, ch in enumerate(question3, 1):
    accumulated3 += ch
    inputs = tokenizer(accumulated3, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=8, temperature=0.3, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][prompt_len:]
    continuation = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"  输入 [{accumulated3}] → 模型续: {continuation}")

print('\n  要点：输入越完整，模型续写越准确。这是“自回归”的本质——')
print(f"       每一步的预测都依赖前面所有 token。")

# ========== 四、原始续写 vs Chat 格式化 ==========
print("\n" + "=" * 60)
print("四、原始续写 vs Chat 格式 —— 同一问题两种问法")
print("=" * 60)

question4 = "候补申请能保证成功吗？"

# 方式 A：原始文本续写（无格式）
print(f"\n【方式 A】原始文本续写：")
raw_text = f"问题：{question4}\n回答："
inputs_raw = tokenizer(raw_text, return_tensors="pt").to(device)
prompt_raw = inputs_raw["input_ids"].shape[1]
with torch.no_grad():
    out_raw = model.generate(
        **inputs_raw, max_new_tokens=60, temperature=0.3, do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
answer_raw = tokenizer.decode(out_raw[0][prompt_raw:], skip_special_tokens=True).strip()
print(f"  输入：{repr(raw_text)}")
print(f"  输出：{answer_raw}")
print(f"  输入 token 数：{prompt_raw}")

# 方式 B：Chat 格式
print(f"\n【方式 B】Chat 格式：")
messages = [
    {"role": "system", "content": "你是 示例业务系统 在线客服，回答要专业准确。"},
    {"role": "user", "content": question4},
]
rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs_chat = tokenizer(rendered, return_tensors="pt").to(device)
prompt_chat = inputs_chat["input_ids"].shape[1]
with torch.no_grad():
    out_chat = model.generate(
        **inputs_chat, max_new_tokens=60, temperature=0.3, do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
answer_chat = tokenizer.decode(out_chat[0][prompt_chat:], skip_special_tokens=True).strip()
print(f"  输入：{repr(rendered[:80])}...")
print(f"  输出：{answer_chat}")
print(f"  输入 token 数：{prompt_chat}（比原始多 {prompt_chat - prompt_raw} 个 role 标记）")

print(f"\n  对比：Chat 格式多了 role 标记 token，让模型明确知道自己的角色和任务。")
print(f"       原始续写模式下模型只是尽力'接话'，回答可能不够受控。")

# ========== 五、总结 ==========
print("\n" + "=" * 60)
print("五、原始文本续写要点")
print("=" * 60)
print("  1. 大模型本质是一个 token 续写引擎，chat 是对它的封装")
print("  2. 输入越完整、越接近训练数据格式，续写质量越好")
print("  3. 逐字符输入时，BPE 分词器会动态合并常见词，不是每加一个字就多一个 token")
print("  4. 原始续写灵活但不可控，Chat 格式通过 role 标记约束模型行为")
print("  5. 理解原始续写后才能理解为什么需要 Chat 模板（接下来几节的内容）")
print()
print("  本脚本使用本地 Qwen3.5-0.8B 实际推理，所有输出均为真实模型生成。")
print("  可以修改：更换 prompt 文本；调整 temperature 观察续写多样性；对比中英文续写差异。")