"""07 Qwen 本地模型最原始多轮对话。

这个脚本不讲工具调用，也不讲 JSON。
只展示本地 Qwen 多轮对话最基本的结构：

1. 程序维护 messages。
2. tokenizer.apply_chat_template 把 messages 渲染成 prompt。
3. model.generate 生成回答。
4. 把 assistant 回答追加回 messages。
5. 下一轮继续带着历史 messages 生成。

运行方式：
    cd rag-tool-agent-course
    python code/08_专题_多轮对话机制/code/07_Qwen本地模型最原始多轮对话.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("07 Qwen 本地模型最原始多轮对话")
print("=" * 72)

MODEL_DIR = COURSE_ROOT / "open_models" / "Qwen3.5-0.8B"
if not MODEL_DIR.is_dir():
    raise SystemExit("未找到 open_models/Qwen3.5-0.8B，请先确认本地模型已经下载。")

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device != "cpu" else torch.float32

print(f"加载本地模型：{MODEL_DIR}")
print(f"运行设备：{device}")

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), torch_dtype=dtype).to(device).eval()

messages = [
    {
        "role": "system",
        "content": "你是一个课堂助手，回答要简洁、清楚，适合初学者理解。",
    }
]

questions = [
    "请用两句话解释什么是多轮对话。 /no_think",
    "那它和单轮问答最大的区别是什么？ /no_think",
    "请用一个客服助手的例子说明。 /no_think",
]

for round_index, question in enumerate(questions, start=1):
    print(f"\n{'=' * 24} 第 {round_index} 轮 {'=' * 24}")
    print("用户：", question)

    messages.append({"role": "user", "content": question})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("\n本轮实际输入给模型的 prompt 片段：")
    print(prompt[:1000])
    if len(prompt) > 1000:
        print("...（后面省略）")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=True,
            temperature=0.3,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][prompt_length:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print("\n模型回答：")
    print(answer)

    messages.append({"role": "assistant", "content": answer})

    print("\n当前 messages：")
    for item in messages:
        print(f"- {item['role']}: {item['content'][:80]}")

print("\n结论")
print("本地模型最原始的多轮对话，就是程序不断维护 messages。")
print("模型并不会自动记住历史，是历史消息被重新渲染进了下一轮 prompt。")
