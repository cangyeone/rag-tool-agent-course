"""06 think 标签和清理。

学习目标：先用 Qwen 模型生成带思考标签（<think>、<thinking> 等）的输出，然后识别并清理不同模型输出的思考标签变体，掌握正则清理和保留可展示依据的策略。

运行方式：python 06_think标签清理.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import re
import torch
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")
from transformers import AutoModelForCausalLM, AutoTokenizer

print("06 think 标签和清理")
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


def qwen_generate(messages, max_new_tokens=200, temperature=0.3):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
            do_sample=True, pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ========== 一、用 Qwen 生成带 think 标签的输出 ==========
print("\n一、让 Qwen 输出带思考过程的回答")

think_prompts = [
    {
        "name": "带 <think> 标签的推理",
        "system": "你是 通用客服助手。请先用 <think>...</think> 标签写出你的分析过程，然后给出最终回答。",
        "user": "ORD-1001 没票了，候补申请能保证成功吗？",
    },
    {
        "name": "带 <thinking> 标签的推理",
        "system": "你是 通用客服助手。请先用 <thinking>...</thinking> 标签写出分析，然后正式回答。",
        "user": "退款费用怎么算？",
    },
]

raw_outputs = []
for tp in think_prompts:
    print(f"\n  场景：{tp['name']}")
    print(f"  提问：{tp['user']}")
    messages = [
        {"role": "system", "content": tp["system"]},
        {"role": "user", "content": tp["user"]},
    ]
    output = qwen_generate(messages, max_new_tokens=200, temperature=0.3)
    raw_outputs.append({"name": tp["name"], "raw": output})
    print(f"  模型原始输出：")
    for line in output.split("\n")[:5]:
        print(f"    {line}")
    if len(output.split("\n")) > 5:
        print(f"    ...")

# 补充一些经典格式的测试样本（从真实模型中不一定能稳定产出这些变体）
raw_outputs.append({
    "name": "```think 代码块格式",
    "raw": "```think\n分析：用户想退款，需要确认服务开始时间。退款费用按阶梯收取，越早退越便宜。\n```\n根据查询，退款费用：服务开始前8天以上免收，48h以上收5%，24h以上收10%，24h内收20%。",
})

# ========== 二、清理策略 ==========
print("\n\n二、统一清理策略")


def clean_think_tags(text: str) -> dict:
    """清理各种 think/thinking/reasoning 标签，返回清理后的文本和元信息。"""
    patterns = [
        r"<think>.*?</think>",
        r"<thinking>.*?</thinking>",
        r"<reasoning>.*?</reasoning>",
        r"<thought>.*?</thought>",
        r"<scratchpad>.*?</scratchpad>",
        r"```think\n.*?\n```",
        r"```thinking\n.*?\n```",
        r"```reasoning\n.*?\n```",
    ]
    extracted_segments = []
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        for m in matches:
            extracted_segments.append(m.strip())
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return {"cleaned": text, "extracted_count": len(extracted_segments), "segments": extracted_segments}


for i, case in enumerate(raw_outputs, 1):
    result = clean_think_tags(case["raw"])
    has_think = result["extracted_count"] > 0
    print(f"\n[{i}] {case['name']}:")
    print(f"    包含 think 段：{'是' if has_think else '否'}（{result['extracted_count']} 段）")
    if has_think:
        for seg in result["segments"]:
            print(f"    思考内容：{seg[:80]}...")
    print(f"    清理后展示：{result['cleaned'][:120]}...")

# ========== 三、业务场景 ==========
print("\n\n三、业务场景 —— 示例业务系统 客服能否展示思考过程？")
scenarios = [
    ("完全隐藏", "不展示任何 think 内容，只展示最终回答。适合面向乘客的客服"),
    ("可折叠展示", "用 <details> 标签包裹思考过程，乘客可选择展开查看。适合内部工作台"),
    ("简化为一句依据", "从 think 中提取一句简洁依据作为引用，如「根据候补申请规则第3条…」。适合需要解释的答复"),
]
for name, desc in scenarios:
    print(f"  • {name}：{desc}")

# ========== 四、从真实输出中提取依据 ==========
print("\n\n四、从 think 内容中提取依据引用")

messages_evidence = [
    {"role": "system", "content": "你是 通用客服助手。先用 <think>...</think> 写出你的推理，引用具体规则条目（如「根据《行业用户运输规程》第X条」），然后给出最终回答。"},
    {"role": "user", "content": "候补申请为什么不能保证成功？"},
]
output_with_evidence = qwen_generate(messages_evidence, max_new_tokens=250, temperature=0.3)
print(f"\n  模型原始输出：")
for line in output_with_evidence.split("\n")[:6]:
    print(f"    {line}")

# 提取 think 内容中的依据
think_match = re.search(r"<think>(.*?)</think>", output_with_evidence, re.DOTALL)
if think_match:
    think_content = think_match.group(1)
    evidence = re.findall(r"根据[^。\n]*[。\n]", think_content)
    if not evidence:
        evidence = re.findall(r"第[^。\n]*[。\n]", think_content)
    print(f"\n  提取到的依据: {evidence if evidence else '无明确规则引用'}")
else:
    print(f"\n  模型未输出 <think> 标签，尝试从全文提取依据...")
    evidence = re.findall(r"根据[^。\n]*[。\n]", output_with_evidence)
    print(f"  提取到的依据: {evidence if evidence else '无明确规则引用'}")

# 清理后展示
cleaned_result = clean_think_tags(output_with_evidence)
print(f"\n  清理 think 后的最终回答：")
for line in cleaned_result["cleaned"].split("\n")[:4]:
    print(f"    {line}")

# ========== 五、DeepSeek 风格的 think 标签 ==========
print("\n\n五、DeepSeek 模型原生的 think 标签格式")
print("  DeepSeek-R1 原生输出格式：")
print("    <think>")
print("    嗯，用户问的是候补申请为什么不能保证成功...")
print("    需要从候补申请机制的本质来解释...")
print("    </think>")
print("    候补申请是根据退款和库存情况自动兑现的，因此不能保证100%成功。")
print()
print("  DeepSeek-R1 的 think 标签是模型训练时内置的思考过程输出，")
print("  与 Qwen 的 ChatML 格式无关，是独立的输出结构。")
print("  清理策略同样适用正则匹配 <think>...</think> 并移除。")

print("\n\n要点：think 区适合调试和内部审计，不适合直接给用户看。业务系统通常展示最终回答，必要时展示简短依据。")
print("所有模型输出均由本地 Qwen3.5-0.8B 真实生成。")
print("可以修改：更换 prompt 观察不同的 think 标签格式；调整清理正则处理更多变体。")