"""01 字符进入模型前长什么样。

学习目标：理解文本从原始汉字到 token id 的完整转换过程，掌握 BPE（字节对编码）的核心思想，知道为什么模型"不认识字"而只认识 token id。

运行方式：python 01_字符进入模型前长什么样.py
"""


import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

import torch
from transformers import AutoTokenizer

print("01 字符进入模型前长什么样")
print("=" * 72)


def _find_qwen_tokenizer():
    candidate = COURSE_ROOT / "open_models" / "Qwen3.5-0.8B"
    if candidate.is_dir():
        return AutoTokenizer.from_pretrained(str(candidate))
    raise FileNotFoundError("未找到 open_models/Qwen3.5-0.8B。请确认已经从 rag-tool-agent-course 根目录运行，并且模型已下载。")


tokenizer = _find_qwen_tokenizer()

question = "服务点A到服务点B的候补申请怎么解释？"
print("一、原始输入")
print(question)

print("\n二、逐字符观察（Unicode 码点）")
for i, ch in enumerate(question, start=1):
    print(f"  {i:02d}. {repr(ch):>6}  U+{ord(ch):04X}  类型: {'中文' if ord(ch) > 127 else '英文/符号'}")

print("\n三、真实 Qwen 分词器输出")
# word -> ID -> token
encoded = tokenizer.encode(question, add_special_tokens=False)
# token ID -> word 
tokens = tokenizer.convert_ids_to_tokens(encoded)

print(f"  token 总数（不含特殊 token）：{len(tokens)}")
print(f"  token id 序列：{encoded}")
print(f"  词表大小：{tokenizer.vocab_size}")
print()

print("四、逐 token 对照（真实 BPE 分词结果）")
print(f"  {'序号':<6} {'token':<16} {'id':<8} {'说明'}")
print("  " + "-" * 50)
for i, (tok, tid) in enumerate(zip(tokens, encoded), start=1):
    desc = "中文词" if any('\u4e00' <= ch <= '\u9fff' for ch in tok.replace('Ġ', '')) else "英文/标点"
    print(f"  {i:<6} {tok!r:<16} {tid:<8} {desc}")

print("\n五、还原验证：id 序列 → token → 文本")
decoded = tokenizer.decode(encoded, skip_special_tokens=True)
print(f"  id 序列: {encoded}")
print(f"  还原文本: {decoded}")
print(f"  与原文一致: {decoded == question}")

print("\n要点：模型不直接看中文句子，而是看 token id 序列。")
print('BPE 让模型在"词级语义"和"字级灵活性"之间取得平衡——常见词整体编码，生僻字拆成片段。')
print(f"本次使用的是 Qwen3.5-0.8B 的真实分词器（词表大小 {tokenizer.vocab_size}）。")