"""02 最小 next-token 循环。

学习目标：理解大模型生成文本的核心原理——不断预测下一个 token，并通过 temperature 参数控制生成结果的随机性。

运行方式：python 02_最小_next_token_循环.py
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

print("02 最小 next-token 循环")
print("=" * 72)


def _find_qwen_path():
    candidate = COURSE_ROOT / "open_models" / "Qwen3.5-0.8B"
    if candidate.is_dir():
        return str(candidate)
    raise FileNotFoundError("未找到 open_models/Qwen3.5-0.8B。请确认已经从 rag-tool-agent-course 根目录运行，并且模型已下载。")


QWEN_PATH = _find_qwen_path()
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device != "cpu" else torch.float32
print(f"加载 Qwen3.5-0.8B 模型（device={device}）...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH)
model = AutoModelForCausalLM.from_pretrained(QWEN_PATH, torch_dtype=dtype).to(device).eval()

print("\n一、Temperature（温度）参数类比")
print("  温度 → 控制选中低频 token 的概率")
print("  低温 (T=0)：贪婪解码，每次选最高概率 token → 输出稳定、可预测")
print("  中温 (T=0.7)：按原始概率采样 → 输出有变化但总体合理")
print("  高温 (T=2.0)：概率分布拉平 → 输出随机、有创造性但可能跑偏")

prompt = "候补申请的规则是"
print(f"\n  提示词：{prompt!r}")


def generate_with_temp(prompt_text, temperature):
    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


print("\n二、Temperature = 0（贪婪解码，每次选最高概率）")
output_greedy = generate_with_temp(prompt, 0)
print(f"  生成结果：{output_greedy}")

print("\n三、Temperature = 0.7（正常采样）")
torch.manual_seed(42)
output_normal = generate_with_temp(prompt, 0.7)
print(f"  生成结果：{output_normal}")

print("\n四、Temperature = 2.0（高温，结果更随机）")
torch.manual_seed(123)
output_high = generate_with_temp(prompt, 2.0)
print(f"  生成结果：{output_high}")

print("\n\n五、多次运行对比（T=0.7，不同 seed）")
print("  同样的提示词和 temperature，不同随机种子产生不同结果：")
for seed in [7, 42, 99]:
    torch.manual_seed(seed)
    result = generate_with_temp(prompt, 0.7)
    print(f"  seed={seed}: {result[:80]}...")

print("\n要点：大模型生成长回答，本质上也是不断重复预测下一个 token。")
print("Temperature 是控制生成多样性的核心参数——示例业务系统 客服场景建议低温（0~0.3），避免随机回答。")
print("以上输出均由本地 Qwen3.5-0.8B 真实生成。")