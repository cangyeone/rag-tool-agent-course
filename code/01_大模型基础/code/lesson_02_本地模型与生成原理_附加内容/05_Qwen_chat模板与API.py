"""05 Qwen chat 模板与 API。

学习目标：理解 messages（接口层）如何被 tokenizer 渲染成模型真正看到的输入文本（模型层），看清 chat template 在背后的转换逻辑，并对比本地模型和云端 API 的请求格式差异。

运行方式：python 05_Qwen_chat模板与API.py
"""


import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")
import json

from transformers import AutoTokenizer

print("05 Qwen chat 模板与 API")
print("=" * 72)

model_dir = COURSE_ROOT / "open_models" / "Qwen3.5-0.8B"
if not model_dir.exists():
    model_dir = None
if model_dir is None:
    raise SystemExit("未找到 open_models/Qwen3.5-0.8B")

tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 单轮对话
messages_single = [
    {"role": "system", "content": "你是 课程助教，回答要简洁。"},
    {"role": "user", "content": "候补申请为什么不能保证成功？"},
]

print("一、接口层看到的 messages（给人看的结构）")
print(json.dumps(messages_single, ensure_ascii=False, indent=2))

print("\n二、tokenizer 内部渲染（给模型看的结构）")
print("  步骤1：apply_chat_template 将 messages 转为文本模板")
rendered = tokenizer.apply_chat_template(messages_single, tokenize=False, add_generation_prompt=True)
print(f"  渲染后文本：\n{rendered}")

print("\n  步骤2：将渲染文本 tokenize")
rendered_ids = tokenizer.encode(rendered, add_special_tokens=False)
print(f"  token ids ({len(rendered_ids)} 个)：{rendered_ids[:30]}...")

# 展示特殊 token 的位置
print("\n三、渲染文本中的结构标记")
# 找到特殊 token 的位置
for token_name in ["<|im_start|>", "<|im_end|>"]:
    try:
        tid = tokenizer.convert_tokens_to_ids(token_name)
        print(f"  {token_name}: id={tid}")
    except Exception:
        print(f"  {token_name}: 未找到")

print("\n四、多轮对话的 chat template 渲染")
messages_multi = [
    {"role": "system", "content": "你是 课程助教，回答要简洁。"},
    {"role": "user", "content": "候补申请为什么不能保证成功？"},
    {"role": "assistant", "content": "候补申请按排队顺序兑现，库存取决于其他乘客的退款与变更情况，所以不能保证一定成功。"},
    {"role": "user", "content": "那候补申请截止时间是什么时候？"},
]
rendered_multi = tokenizer.apply_chat_template(messages_multi, tokenize=False, add_generation_prompt=True)
print("  多轮渲染后文本：")
print(f"  {rendered_multi[:200]}...")
print(f"  多轮 token 总数：{len(tokenizer.encode(rendered_multi, add_special_tokens=False))}")

print("\n五、本地模型 vs 云端 API 请求格式对比")
print("  本地模型（transformers）调用方式：")
print("    tokenizer.apply_chat_template(messages) → tokenize → model.generate(input_ids)")

print("\n  云端 API（OpenAI 兼容）请求体：")
api_body = {
    "model": "deepseek-v4-flash",
    "messages": messages_single,
    "temperature": 0.2,
    "max_tokens": 200,
    "stream": True,
}
print(json.dumps(api_body, ensure_ascii=False, indent=2))

print("\n  区别总结：")
differences = [
    ("messages 格式", "相同（都是 OpenAI 兼容格式）", "相同"),
    ("角色标记", "由 tokenizer chat_template 自动添加", "API 服务端内部处理，用户不可见"),
    ("tokenize", "用户需要手动调用 tokenizer", "API 服务端自动完成"),
    ("流式输出", "用 TextIteratorStreamer 迭代", "HTTP SSE 协议推送（data: {...}）"),
    ("参数控制", "通过 generate() 参数控制", "通过 JSON 请求体字段控制"),
]
print(f"  {'对比项':<18} {'本地模型':<38} {'云端 API'}")
for item, local, cloud in differences:
    print(f"  {item:<18} {local:<38} {cloud}")

print("\n要点：messages 是给人和接口看的结构，chat template 是给本地模型看的文本格式。API 调用时这两步合并了。")