"""03 Qwen 本地模型加载检查。

学习目标：学会检查本地模型文件的完整性，理解加载模型所需的三个核心文件（config.json、tokenizer.json、model.safetensors），并为后续的本地推理做好准备。

运行方式：python 03_Qwen_本地模型加载检查.py
"""


import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("03 Qwen 本地模型加载检查")
print("=" * 72)

model_dir = COURSE_ROOT / "open_models" / "Qwen3.5-0.8B"
if not model_dir.exists():
    model_dir = None

if model_dir is None:
    print("未找到模型目录：open_models/Qwen3.5-0.8B")
    raise SystemExit("请先下载 Qwen3.5-0.8B。")

print("一、模型目录")
print(f"  {model_dir}")

# 检查关键文件
print("\n二、关键文件检查")
required_files = [
    ("config.json", "模型架构配置（层数、维度、注意力头数）"),
    ("tokenizer.json", "词表文件（token 到 id 的映射表）"),
    ("tokenizer_config.json", "分词器配置（特殊 token、chat template 等）"),
    ("generation_config.json", "生成参数默认配置（max_length、temperature 等）"),
]
optional_files = [
    ("vocab.json", "词表（部分模型使用）"),
    ("merges.txt", "BPE 合并规则（部分模型使用）"),
]
for fname, desc in required_files:
    path = model_dir / fname
    exists = "存在" if path.exists() else "缺失"
    size_mb = f"{path.stat().st_size / 1024 / 1024:.1f} MB" if path.exists() else "—"
    print(f"  {fname:<28} {exists:<4} {size_mb:>10}  ({desc})")

weight_files = sorted(model_dir.glob("*.safetensors")) + sorted(model_dir.glob("*.bin"))
weight_status = "存在" if weight_files else "缺失"
weight_size = sum(path.stat().st_size for path in weight_files) / 1024 / 1024 if weight_files else 0
weight_names = ", ".join(path.name for path in weight_files[:2])
if len(weight_files) > 2:
    weight_names += f" 等 {len(weight_files)} 个文件"
print(f"  {'模型权重文件':<28} {weight_status:<4} {weight_size:>9.1f} MB  ({weight_names or '未找到 safetensors/bin 权重'})")

print("\n三、可选文件")
for fname, desc in optional_files:
    path = model_dir / fname
    exists = "存在" if path.exists() else "不存在（通常不需要）"
    print(f"  {fname:<28} {exists}  ({desc})")

print("\n四、加载 tokenizer")
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_dir)
text = "服务点A到服务点B的候补申请怎么解释？"
ids = tokenizer.encode(text, add_special_tokens=False)
print(f"  输入：{text}")
print(f"  token ids：{ids[:30]}")
print(f"  token 数量：{len(ids)}")
print(f"  词表大小：{tokenizer.vocab_size}")
print(f"  是否有 chat_template：{bool(tokenizer.chat_template)}")

# 额外：展示 chat_template 的角色 token
print("\n五、特殊 token 观察")
for name in ["bos_token", "eos_token", "pad_token", "unk_token"]:
    token = getattr(tokenizer, name, None)
    if token is not None:
        tid = tokenizer.convert_tokens_to_ids(token)
        print(f"  {name}: {repr(token)} → id={tid}")

print("\n要点：本地模型加载前先检查文件完整性，核心是配置、分词器和权重文件。")
print("权重可能是单个 model.safetensors，也可能是 model.safetensors-00001-of-00001.safetensors 这类分片文件。")