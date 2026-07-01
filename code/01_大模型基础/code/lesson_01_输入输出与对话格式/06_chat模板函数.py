"""06 Chat模式 —— 使用 chat 模板函数。

学习目标：掌握 transformers 的 `apply_chat_template()` 方法，理解它是如何自动将 messages 列表转换为模型能理解的完整文本格式，并与上一节的手动拼接做对比，体会模板函数的便利性和正确性保障。

运行方式：python 06_chat模板函数.py
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

print("06 Chat模式 —— 使用模板函数")
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

# ========== 一、apply_chat_template 基本用法 ==========
print("\n" + "=" * 60)
print("一、apply_chat_template 基本用法")
print("=" * 60)

print("\n  核心方法签名：")
print("    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)")
print()
print("  参数说明：")
print("    messages  →  list[dict]，每项含 role 和 content")
print("    tokenize  →  False 返回文本字符串；True 返回 token id 列表")
print("    add_generation_prompt → True 时自动追加生成提示（如 <|im_start|>assistant\\n）")

# 构造 messages
system_prompt = "你是 通用客服助手，回答要简洁专业。"
user_question = "ORD-1001 没票了，还能怎么办？"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_question},
]

print(f"\n  输入 messages：")
for m in messages:
    print(f"    role={m['role']}, content=\"{m['content']}\"")

# 方式1：tokenize=False，返回文本
rendered_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"\n  tokenize=False 返回文本（可供人阅读）：")
print(f"    {repr(rendered_text)}")

# 方式2：tokenize=True，返回 token id 列表
rendered_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
print(f"\n  tokenize=True 返回 token ids（直接送模型）：")
print(f"    token 数量：{len(rendered_ids)}")
print(f"    前 10 个 id：{rendered_ids[:10]}")
print(f"    后 10 个 id：{rendered_ids[-10:]}")

# ========== 二、完整推理流程（从 messages 到回答）==========
print("\n" + "=" * 60)
print("二、完整推理流程 —— messages → template → tokenize → generate")
print("=" * 60)

# 第1步：messages 结构
print("\n  第1步：构造 messages")
print(f"    [{messages[0]['role']}] {messages[0]['content']}")
print(f"    [{messages[1]['role']}] {messages[1]['content']}")

# 第2步：渲染模板
rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"\n  第2步：apply_chat_template 渲染")
print(f"    渲染后字符数：{len(rendered)}")

# 第3步：tokenize
inputs = tokenizer(rendered, return_tensors="pt").to(device)
prompt_len = inputs["input_ids"].shape[1]
print(f"\n  第3步：tokenize → {prompt_len} 个 token")

# 第4步：推理
print(f"\n  第4步：模型推理 →")
with torch.no_grad():
    output_ids = model.generate(
        **inputs, max_new_tokens=100, temperature=0.3, do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
generated = output_ids[0][prompt_len:]
answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
print(f"    {answer}")

# ========== 三、与手动拼接的对比 ==========
print("\n" + "=" * 60)
print("三、模板函数 vs 手动拼接（回顾上节内容）")
print("=" * 60)

# 手动拼接
chatml_manual = (
    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    f"<|im_start|>user\n{user_question}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)

print(f"  手动拼接（上次的方法）：")
print(f"    {repr(chatml_manual)}")
print(f"  模板函数（本次的方法）：")
print(f"    {repr(rendered_text)}")
print(f"  两者一致：{chatml_manual.strip() == rendered_text.strip()}")

print(f"\n  对比总结：")
print(f"    手动拼接：需要记住 ChatML 格式、手动处理换行和标记，容易出错")
print(f"    模板函数：自动根据模型训练时的格式渲染，跨模型切换时无需改动代码")
print(f"    推荐：始终使用 apply_chat_template，只在调试格式问题时了解手动拼接")

# ========== 四、多轮对话的模板渲染 ==========
print("\n" + "=" * 60)
print("四、多轮对话的模板渲染")
print("=" * 60)

multi_messages = [
    {"role": "system", "content": "你是 通用客服助手。"},
    {"role": "user", "content": "候补申请是什么？"},
    {"role": "assistant", "content": "候补申请是指当订单无库存时，您可以提交候补申请订单，系统按排队顺序在有退款或新增库存时自动兑现。"},
    {"role": "user", "content": "那截止时间呢？"},
]

# 逐轮观察 token 累积
print(f"\n  逐轮 token 累积：")
for i in range(1, len(multi_messages)):
    partial = multi_messages[:i + 1]
    is_last_user = partial[-1]["role"] == "user"
    partial_rendered = tokenizer.apply_chat_template(partial, tokenize=False, add_generation_prompt=is_last_user)
    partial_len = len(tokenizer.encode(partial_rendered, add_special_tokens=False))
    print(f"    前 {i + 1} 条消息: {partial_len} tokens (角色={partial[-1]['role']})")

full_rendered = tokenizer.apply_chat_template(multi_messages, tokenize=False, add_generation_prompt=True)
full_len = len(tokenizer.encode(full_rendered, add_special_tokens=False))
print(f"\n  完整对话 token 数：{full_len}")
print(f"  Qwen3.5-0.8B 上下文窗口：约 32768 tokens，当前占用 {full_len / 32768 * 100:.1f}%")

# 用多轮对话推理
print(f"\n  多轮对话推理结果：")
inputs_multi = tokenizer(full_rendered, return_tensors="pt").to(device)
prompt_multi_len = inputs_multi["input_ids"].shape[1]
with torch.no_grad():
    output_ids = model.generate(
        **inputs_multi, max_new_tokens=80, temperature=0.3, do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
answer_multi = tokenizer.decode(output_ids[0][prompt_multi_len:], skip_special_tokens=True).strip()
print(f"    {answer_multi}")

# ========== 五、add_generation_prompt 参数的作用 ==========
print("\n" + "=" * 60)
print("五、add_generation_prompt 参数的细节")
print("=" * 60)

without_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
with_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
diff = with_prompt[len(without_prompt):]

print(f"  add_generation_prompt=False：{repr(without_prompt[-40:])}")
print(f"  add_generation_prompt=True： {repr(with_prompt[-40:])}")
print(f"  追加的内容：{repr(diff)}")
print(f"  说明：add_generation_prompt=True 自动追加 '<|im_start|>assistant\\n'，")
print(f"       让模型知道该它说话了，这是生成回答的起点。")

# ========== 六、总结 ==========
print("\n" + "=" * 60)
print("六、模板函数要点")
print("=" * 60)
print("  1. 始终使用 apply_chat_template() 渲染 messages，不要手动拼接")
print("  2. tokenize=False 用于查看文本；tokenize=True 直接得到 token ids")
print("  3. add_generation_prompt=True 是生成回答的关键，缺失时模型可能不输出")
print("  4. 多轮对话只需把所有历史消息传入 messages，模板函数自动处理格式")
print("  5. 更换模型时只需更换 tokenizer，无需改动 messages 拼接逻辑")
print()
print("  本脚本使用本地 Qwen3.5-0.8B 实际推理，所有输出均为真实模型生成。")
print("  可以修改：更换 system prompt；增加多轮对话轮次；对比不同 temperature 的输出。")