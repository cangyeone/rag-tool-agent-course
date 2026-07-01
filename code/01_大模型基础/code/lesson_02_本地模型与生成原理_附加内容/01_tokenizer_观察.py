"""01 tokenizer 观察。

学习目标：亲眼观察 tokenizer 如何把中文文本拆成 token 和 token id，理解同一段文字能被不同 tokenizer 拆成不同的 token 序列。

运行方式：python 01_tokenizer_观察.py
"""


import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("01 tokenizer 观察")
print("=" * 72)

text = "服务点A到服务点B的候补申请怎么解释？"
model_dir = COURSE_ROOT / "open_models" / "Qwen3.5-0.8B"
if not model_dir.exists():
    model_dir = None

print("输入文本：", text)
print("模型目录：", model_dir if model_dir else "未找到 open_models/Qwen3.5-0.8B")

try:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text, add_special_tokens=False)
    print("\n一、真实 Qwen tokenizer 输出")
    print("词表大小：", tokenizer.vocab_size)
    print("tokens:", tokens)
    print("ids   :", ids)

    print("\n二、逐 token 对照")
    for i, (tok, tid) in enumerate(zip(tokens, ids)):
        print(f"  {i:02d}. token={tok!r:<12} id={tid}")

    # 额外观察：特殊 token
    print("\n三、特殊 token 观察")
    special_tokens = {
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token,
        "unk_token": tokenizer.unk_token,
    }
    for name, val in special_tokens.items():
        if val is not None:
            print(f"  {name}: {repr(val)} → id={tokenizer.convert_tokens_to_ids(val)}")

except Exception as exc:
    print("\n无法加载真实 tokenizer：", exc)
    raise SystemExit("请先确保 open_models/Qwen3.5-0.8B 已下载。")

print("\n要点：tokenizer 负责把文字变成模型能计算的数字。同一个意思，不同模型的 tokenizer 切分方式不同。")
print('中文 tokenizer 的挑战：既要切出常见词（如"北京"），又要处理未登录词（如生僻地名）。')