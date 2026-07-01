"""05 Chat模式 —— 原始字符串拼接。

学习目标：理解 Chat 模式的底层原理——不依赖任何模板函数，手动使用 Qwen 的 ChatML 特殊标记（<|im_start|>/<|im_end|>）拼接出模型能理解的完整对话格式，掌握 messages 到 token 序列的原始转换过程。

运行方式：python 05_chat原始拼接.py
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

print("05 Chat模式 —— 原始拼接")
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

# ========== 一、认识 Qwen 的 ChatML 特殊标记 ==========
print("\n" + "=" * 60)
print("一、Qwen 使用的 ChatML 特殊标记（对话格式的'骨架'）")
print("=" * 60)

special_tokens = [
    ("<|im_start|>", "对话轮次开始标记，后跟角色名"),
    ("<|im_end|>", "对话轮次结束标记"),
    ("<|endoftext|>", "文档结束 / 填充标记"),
]
for tok, desc in special_tokens:
    try:
        tid = tokenizer.convert_tokens_to_ids(tok)
        print(f"  {tok}  →  token id = {tid}  ({desc})")
    except Exception:
        print(f"  {tok}  →  词表中不存在")

print(f"\n  Qwen ChatML 完整格式：")
print(f"  <|im_start|>system")
print(f"  {{系统提示词}}<|im_end|>")
print(f"  <|im_start|>user")
print(f"  {{用户问题}}<|im_end|>")
print(f"  <|im_start|>assistant")
print(f"  {{模型从这里开始生成回答}}")

# ========== 二、手动拼接 ChatML 字符串 ==========
print("\n" + "=" * 60)
print("二、手动拼接 —— 不依赖任何模板函数")
print("=" * 60)

system_prompt = "你是 通用客服助手，回答要简洁专业。"
user_question = "ORD-1001 没票了，还能怎么办？"

# 手动拼接 ChatML 格式
chatml_manual = (
    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    f"<|im_start|>user\n{user_question}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)

print(f"手动拼接后的完整字符串：")
print(f"  {repr(chatml_manual)}")
print(f"  总字符数：{len(chatml_manual)}")

# 查看 tokenize 结果
manual_tokens = tokenizer.tokenize(chatml_manual)
manual_ids = tokenizer.encode(chatml_manual, add_special_tokens=False)
print(f"\n  token 数量（不含 BOS）：{len(manual_tokens)}")
print(f"\n  逐 token 解析：")
print(f"  {'序号':<6} {'token':<20} {'id':<8} {'角色'}")
print(f"  " + "-" * 55)
current_role = "system"
for i, (tok, tid) in enumerate(zip(manual_tokens, manual_ids), 1):
    tok_display = tok.replace("Ġ", " ").replace("Ċ", "\\n")
    if tok in ("<|im_start|>", "<|im_end|>"):
        role_label = "标记"
    elif tok in ("system", "user", "assistant"):
        current_role = tok
        role_label = "角色声明"
    else:
        role_label = current_role
    print(f"  {i:<6} {tok_display:<20} {tid:<8} {role_label}")

# ========== 三、用手动拼接的字符串做推理 ==========
print("\n" + "=" * 60)
print("三、用手动拼接的字符串直接推理")
print("=" * 60)

inputs_manual = tokenizer(chatml_manual, return_tensors="pt").to(device)
prompt_len = inputs_manual["input_ids"].shape[1]

print(f"  输入 token 数：{prompt_len}")
print(f"\n  模型回答：")

with torch.no_grad():
    output_ids = model.generate(
        **inputs_manual, max_new_tokens=100, temperature=0.3, do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
generated = output_ids[0][prompt_len:]
answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
print(f"    {answer}")

# ========== 四、对比：手动拼接 vs messages 结构 ==========
print("\n" + "=" * 60)
print("四、对比：手动拼接 vs messages 列表（接口层）")
print("=" * 60)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_question},
]

print(f"  messages 结构（给人看的）：")
print(f"    [")
print(f'      {{"role": "system", "content": "{system_prompt}"}},')
print(f'      {{"role": "user", "content": "{user_question}"}}')
print(f"    ]")

rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"\n  apply_chat_template 渲染结果：")
print(f"    {repr(rendered)}")
print(f"\n  手动拼接 vs 模板渲染是否一致：{chatml_manual.strip() == rendered.strip()}")

# ========== 五、多轮对话的手动拼接 ==========
print("\n" + "=" * 60)
print("五、多轮对话的手动拼接")
print("=" * 60)

multi_messages = [
    {"role": "system", "content": "你是 通用客服助手。"},
    {"role": "user", "content": "候补申请是什么？"},
    {"role": "assistant", "content": "候补申请是指当订单无库存时，您可以提交候补申请订单，系统按排队顺序在有退款或新增库存时自动兑现。"},
    {"role": "user", "content": "那候补申请截止时间呢？"},
]

# 手动拼接多轮
chatml_multi = ""
for m in multi_messages:
    chatml_multi += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
chatml_multi += "<|im_start|>assistant\n"

multi_rendered = tokenizer.apply_chat_template(multi_messages, tokenize=False, add_generation_prompt=True)
print(f"  多轮对话手动拼接：")
for line in chatml_multi.strip().split("\n"):
    print(f"    {line}")
print(f"\n  手动拼接与模板渲染一致：{chatml_multi.strip() == multi_rendered.strip()}")
print(f"  手动拼接 token 数：{len(tokenizer.encode(chatml_multi, add_special_tokens=False))}")

# ========== 六、总结 ==========
print("\n" + "=" * 60)
print("六、原始拼接要点")
print("=" * 60)
print("  1. ChatML 是 Qwen 模型训练时使用的对话格式，手动拼接必须严格遵循")
print("  2. 核心标记：<|im_start|>角色名\\n内容<|im_end|>")
print("  3. assistant 轮次末尾不加 <|im_end|>（这是生成起点）")
print("  4. 手动拼接 = apply_chat_template 的底层实现，理解它有助于调试格式问题")
print("  5. 不同模型有不同的 chat 标记（如 Llama 用 <|start_header_id|>），不能混用")
print()
print("  本脚本使用本地 Qwen3.5-0.8B 实际推理，所有输出均为真实模型生成。")
print("  可以修改：替换 system prompt 和 user question 观察手动拼接的返回变化；")
print("           尝试在多轮对话中插入新的 user/assistant 轮次。")