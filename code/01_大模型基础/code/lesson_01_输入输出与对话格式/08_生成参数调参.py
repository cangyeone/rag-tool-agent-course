"""08 生成参数调参体验。

学习目标：理解 temperature、do_sample、top_k、top_p 等生成参数的含义，通过同一问题不同参数的对比实验，观察输出如何变化，掌握在 示例业务系统 客服场景中的调参策略。

运行方式：python 08_生成参数调参.py
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

print("08 生成参数调参体验")
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


def qwen_generate(messages, max_new_tokens=50, temperature=0.7, do_sample=True, top_k=50, top_p=0.9):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ========== 一、do_sample：采样 vs 贪心 ==========
print("\n" + "=" * 60)
print("一、do_sample：到底要不要'掷骰子'")
print("=" * 60)

print("  do_sample=False（贪心解码）：每步只选概率最高的 token，输出确定且可复现")
print("  do_sample=True（随机采样）：每步按概率分布随机挑选，每次输出可能不同")
print()

question = "候补申请的规则是什么？"
messages = [
    {"role": "system", "content": "你是 通用客服助手，回答要简洁专业。"},
    {"role": "user", "content": question},
]

# 贪心解码（确定性）
print(f"【do_sample=False 贪心解码】同一问题运行 3 次：")
for i in range(3):
    answer = qwen_generate(messages, max_new_tokens=40, do_sample=False, temperature=1.0)
    print(f"  第{i+1}次: {answer[:60]}...")
print("  → 每次结果完全一致，适合需要确定性回答的场景（如规章查询）")

# 随机采样
print(f"\n【do_sample=True 随机采样】同一问题运行 3 次：")
for i in range(3):
    answer = qwen_generate(messages, max_new_tokens=40, do_sample=True, temperature=0.7)
    print(f"  第{i+1}次: {answer[:60]}...")
print("  → 每次结果可能有差异，适合需要多样性的场景（如话术润色）")

# ========== 二、temperature：控制随机程度 ==========
print("\n" + "=" * 60)
print("二、temperature：概率分布的'温度计'")
print("=" * 60)

print("  temperature 控制概率分布的平滑程度：")
print("    < 1.0 → 拉大高低概率的差距，输出更确定、更保守")
print("    = 1.0 → 保持原始概率分布")
print("    > 1.0 → 缩小概率差距，输出更随机、更发散")
print()

question2 = "请用一句话介绍 示例业务系统 候补申请功能。"
messages2 = [
    {"role": "system", "content": "你是 通用客服助手。"},
    {"role": "user", "content": question2},
]

temps = [0.1, 0.5, 0.7, 1.0, 1.5]
for t in temps:
    answer = qwen_generate(messages2, max_new_tokens=50, temperature=t, do_sample=True, top_k=50, top_p=0.9)
    stability = "很确定" if t <= 0.3 else ("适中" if t <= 0.8 else "较发散")
    print(f"  temperature={t:.1f} ({stability}): {answer[:80]}...")

print(f"\n  temperature 选择建议：")
print(f"    0.1-0.3：事实问答、规章查询、代码生成 → 需要准确一致")
print(f"    0.5-0.7：客服对话、内容摘要 → 需要一定灵活性")
print(f"    0.8-1.0：创意写作、话术多样化 → 需要多样性")

# ========== 三、top_k：限制候选池大小 ==========
print("\n" + "=" * 60)
print("三、top_k：只在前 K 个候选中选")
print("=" * 60)

print("  top_k 每步只保留概率最高的 K 个 token，从它们中采样，其它直接丢弃。")
print("  top_k 越小，输出越保守；越大越自由。")
print()

question3 = "退款费用怎么算？"
messages3 = [
    {"role": "system", "content": "你是 通用客服助手，回答要准确。"},
    {"role": "user", "content": question3},
]

top_k_values = [1, 5, 20, 50, 100]
for k in top_k_values:
    answer = qwen_generate(messages3, max_new_tokens=50, temperature=0.7, do_sample=True, top_k=k, top_p=1.0)
    label = "(贪心)" if k == 1 else ("(保守)" if k <= 5 else ("(适中)" if k <= 20 else "(自由)"))
    print(f"  top_k={k:3d} {label}: {answer[:80]}...")

print(f"\n  top_k=1 等价于贪心解码（do_sample 形同虚设）。")
print(f"  推荐：top_k=40-50 配合 temperature=0.7，在质量和多样性间平衡。")

# ========== 四、top_p（nucleus sampling）：动态候选池 ==========
print("\n" + "=" * 60)
print("四、top_p（核采样）：按累积概率动态截断")
print("=" * 60)

print("  top_p 不是固定数量，而是从高到低累加概率，累加到 ≥ top_p 时截断。")
print("  优点：能根据概率分布动态调整候选池大小，比固定 top_k 更智能。")
print()

question4 = "ORD-1001 没票了，还能怎么办？"
messages4 = [
    {"role": "system", "content": "你是 通用客服助手，回答要简洁。"},
    {"role": "user", "content": question4},
]

top_p_values = [0.5, 0.7, 0.85, 0.95, 1.0]
for p in top_p_values:
    answer = qwen_generate(messages4, max_new_tokens=50, temperature=0.7, do_sample=True, top_k=0, top_p=p)
    label = "(很保守)" if p <= 0.5 else ("(保守)" if p <= 0.7 else ("(适中)" if p <= 0.85 else "(自由)"))
    print(f"  top_p={p:.2f} {label}: {answer[:80]}...")

print(f"\n  top_p 和 top_k 通常不同时使用。用其中一个即可。")
print(f"  推荐：top_p=0.85-0.95 配合 temperature=0.7，是常见搭配。")

# ========== 五、各参数组合对比 ==========
print("\n" + "=" * 60)
print("五、典型参数组合在 示例业务系统 场景的对比")
print("=" * 60)

question5 = "候补申请能保证成功吗？"
messages5 = [
    {"role": "system", "content": "你是 通用客服助手，回答要专业准确。"},
    {"role": "user", "content": question5},
]

configs = [
    ("严谨模式",    {"temperature": 0.1, "do_sample": True, "top_p": 0.5,  "top_k": 20},  "事实问答，如规章查询"),
    ("日常客服",    {"temperature": 0.5, "do_sample": True, "top_p": 0.85, "top_k": 40},  "客服对话，需要自然"),
    ("话术润色",    {"temperature": 0.8, "do_sample": True, "top_p": 0.95, "top_k": 50},  "同义改写，需要多样性"),
    ("贪心对比",    {"temperature": 1.0, "do_sample": False,"top_p": 1.0,  "top_k": 50},  "纯贪心，作为对比基线"),
]

for name, params, desc in configs:
    answer = qwen_generate(messages5, max_new_tokens=60, **params)
    print(f"\n  [{name}] {desc}")
    print(f"    参数: temp={params['temperature']}, do_sample={params['do_sample']}, top_p={params['top_p']}, top_k={params['top_k']}")
    print(f"    输出: {answer[:100]}...")

# ========== 六、repetition_penalty：防止重复 ==========
print("\n" + "=" * 60)
print("六、repetition_penalty：防止模型'复读'")
print("=" * 60)

print("  repetition_penalty 对已出现的 token 降权，防止模型陷入循环重复。")
print("  > 1.0 惩罚重复，< 1.0 鼓励重复。推荐 1.05-1.2。")
print()

# 用一个容易让模型重复的 prompt
repeat_prompt = [
    {"role": "user", "content": "请重复说'你好'五次："},
]

for rp in [1.0, 1.1, 1.3]:
    text = tokenizer.apply_chat_template(repeat_prompt, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=40, temperature=0.5, do_sample=True,
            top_p=0.9, repetition_penalty=rp,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    print(f"  repetition_penalty={rp}: {answer[:80]}")

# ========== 七、max_new_tokens：控制生成长度 ==========
print("\n" + "=" * 60)
print("七、max_new_tokens：控制回答长度")
print("=" * 60)

messages_len = [
    {"role": "system", "content": "你是 通用客服助手。"},
    {"role": "user", "content": "候补申请的规则是什么？"},
]

for max_tok in [20, 50, 100]:
    answer = qwen_generate(messages_len, max_new_tokens=max_tok, temperature=0.5, do_sample=True, top_p=0.85)
    truncated = len(answer) >= 80 and max_tok < 100
    print(f"  max_new_tokens={max_tok:3d}: {answer[:80]}{'...[截断]' if truncated else ''}")

# ========== 八、参数速查表 ==========
print("\n" + "=" * 60)
print("八、参数速查表")
print("=" * 60)

print(f"  {'参数':<22} {'含义':<30} {'推荐值':<18} {'影响'}")
print(f"  {'-'*22} {'-'*30} {'-'*18} {'-'*30}")
print(f"  {'temperature':<22} {'概率分布的平滑程度':<30} {'0.5-0.7':<18} {'越高越随机，越低越确定'}")
print(f"  {'do_sample':<22} {'是否随机采样':<30} {'True':<18} {'False=贪心，每次一样'}")
print(f"  {'top_k':<22} {'只从概率前K个中采样':<30} {'40-50':<18} {'越小越保守，越大越自由'}")
print(f"  {'top_p':<22} {'累加到概率p后截断':<30} {'0.85-0.95':<18} {'越小越保守，动态候选池'}")
print(f"  {'repetition_penalty':<22} {'惩罚已出现的token':<30} {'1.05-1.1':<18} {'>1减少重复，=1不惩罚'}")
print(f"  {'max_new_tokens':<22} {'最多生成多少个token':<30} {'100-300':<18} {'越大回答越长'}")
print()
print(f"  本脚本使用本地 Qwen3.5-0.8B 实际推理，所有输出均为真实模型生成。")
print(f"  可以修改：组合不同参数观察输出变化；为不同业务场景（规章查询/客服对话/话术生成）设置不同参数。")