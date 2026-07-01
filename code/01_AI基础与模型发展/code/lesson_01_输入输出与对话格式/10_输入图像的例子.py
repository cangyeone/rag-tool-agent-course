"""10 输入图像的例子 —— 调用本地 Qwen3.5 图像理解模型。

学习目标：看懂图片如何作为 messages 输入给本地 Qwen3.5 多模态模型，
并完成一次真实的图片问答调用。

运行方式：
    python code/01_AI基础与模型发展/code/lesson_01_输入输出与对话格式/10_输入图像的例子.py

模型目录：
    rag-tool-agent-course/open_models/Qwen3.5-0.8B
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("10 输入图像的例子 —— 调用本地 Qwen3.5 图像理解模型")
print("=" * 72)

# ------------------------------------------------------------
# 一、确认本地 Qwen3.5-0.8B 是多模态模型
# ------------------------------------------------------------
# 这一步很重要：不要只看名字判断模型能力。
# 这个本地模型的 config.json 里包含 vision_config、image_token_id、video_token_id，
# preprocessor_config.json 里也写着 Qwen3VLProcessor，所以它可以接收图像输入。

model_path = COURSE_ROOT / "open_models" / "Qwen3.5-0.8B"
if not (model_path / "config.json").exists():
    raise SystemExit("未找到 open_models/Qwen3.5-0.8B，请先确认模型已经放在课程根目录的 open_models 下。")

print("\n一、本地模型检查")
print(f"  模型目录：{model_path.relative_to(COURSE_ROOT)}")

try:
    import json
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    preprocessor = json.loads((model_path / "preprocessor_config.json").read_text(encoding="utf-8"))
    print(f"  model_type：{config.get('model_type')}")
    print(f"  architecture：{config.get('architectures')}")
    print(f"  processor_class：{preprocessor.get('processor_class')}")
    print(f"  image_token_id：{config.get('image_token_id')}")
    print(f"  video_token_id：{config.get('video_token_id')}")
    print("  结论：当前 Qwen3.5-0.8B 带视觉配置，可以作为本地多模态模型演示。")
except Exception as exc:
    print(f"  配置读取失败，但仍会尝试加载模型：{exc}")

# ------------------------------------------------------------
# 二、准备一张课堂演示图片
# ------------------------------------------------------------
# 为了保证 Windows、macOS、Linux 都能跑，图片用 PIL 现场生成，
# 不依赖外部文件，也不依赖系统中文字体。

print("\n二、准备演示图片")
image_dir = COURSE_ROOT / "code" / "01_AI基础与模型发展" / "code" / "lesson_01_输入输出与对话格式"
image_path = image_dir / "_demo_qwen35_vl_station_board.jpg"

img = Image.new("RGB", (900, 520), color=(238, 242, 247))
draw = ImageDraw.Draw(img)

try:
    font_big = ImageFont.truetype("Arial.ttf", 54)
    font_mid = ImageFont.truetype("Arial.ttf", 38)
    font_small = ImageFont.truetype("Arial.ttf", 30)
except Exception:
    font_big = ImageFont.load_default()
    font_mid = ImageFont.load_default()
    font_small = ImageFont.load_default()

# 信息屏背景
draw.rectangle([0, 0, 900, 100], fill=(0, 68, 150))
draw.text((40, 26), "Railway Station Service Board", fill="white", font=font_mid)
draw.rectangle([70, 150, 830, 410], fill=(12, 30, 72))
draw.rectangle([70, 150, 830, 410], outline=(255, 204, 64), width=6)

# 用英文和数字，避免不同系统缺中文字体导致图片文字缺失。
draw.text((120, 185), "Train: G107", fill=(255, 225, 90), font=font_big)
draw.text((120, 255), "From: Beijing South", fill="white", font=font_mid)
draw.text((120, 305), "To: Shanghai Hongqiao", fill="white", font=font_mid)
draw.text((120, 355), "Gate: A12    Platform: 3    Time: 13:42", fill=(120, 220, 255), font=font_small)

# 简单画一段轨道，强调这是图像输入。
draw.line([120, 455, 780, 455], fill=(100, 100, 100), width=10)
draw.line([160, 480, 820, 480], fill=(100, 100, 100), width=10)
for x in range(160, 820, 70):
    draw.line([x, 440, x + 40, 500], fill=(140, 120, 90), width=6)

img.save(image_path, quality=95)
print(f"  图片已生成：{image_path.relative_to(COURSE_ROOT)}")
print(f"  图片尺寸：{img.size[0]} x {img.size[1]}")

# ------------------------------------------------------------
# 三、构造多模态 messages
# ------------------------------------------------------------
# 纯文本模型通常是：content = "一句话"
# 多模态模型是：content = [{image}, {text}]
# 这就是课堂里要重点看懂的结构。

question = "请读取图片中的信息，回答订单编号、出发站、到达站、办理窗口、服务窗口和时间。请用中文分点回答。"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": question},
        ],
    }
]

print("\n三、多模态 messages 结构")
print("  messages = [")
print("    {")
print('      "role": "user",')
print('      "content": [')
print(f'        {{"type": "image", "image": "{image_path.relative_to(COURSE_ROOT)}"}},')
print(f'        {{"type": "text", "text": "{question}"}}')
print("      ]")
print("    }")
print("  ]")

# ------------------------------------------------------------
# 四、加载 processor，把图片转成模型输入张量
# ------------------------------------------------------------
# AutoProcessor 会根据 preprocessor_config.json 自动加载 Qwen3VLProcessor。
# 它会把图片转成 pixel_values，把文本转成 input_ids。

print("\n四、processor 处理图像和文本")
processor = AutoProcessor.from_pretrained(str(model_path))
print(f"  processor 类型：{type(processor).__name__}")

text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("\n  apply_chat_template 后的文本片段：")
print("  " + text_prompt[:300].replace("\n", "\\n") + "...")

# 这里直接把 PIL 图片交给 processor，不额外依赖 qwen-vl-utils。
inputs = processor(
    text=[text_prompt],
    images=[img],
    padding=True,
    return_tensors="pt",
)

print("\n  processor 输出张量：")
for key, value in inputs.items():
    if hasattr(value, "shape"):
        print(f"    {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
    else:
        print(f"    {key}: {type(value).__name__}")

# ------------------------------------------------------------
# 五、加载 Qwen3.5 多模态模型并生成回答
# ------------------------------------------------------------

print("\n五、加载本地模型并推理")

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
print(f"  推理设备：{device}")
print(f"  计算精度：{dtype}")

model = None
errors = []

try:
    from transformers import Qwen3_5ForConditionalGeneration
    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(model_path), torch_dtype=dtype)
    print("  使用模型类：Qwen3_5ForConditionalGeneration")
except Exception as exc:
    errors.append(f"Qwen3_5ForConditionalGeneration: {exc}")

if model is None:
    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(str(model_path), torch_dtype=dtype)
        print("  使用模型类：AutoModelForImageTextToText")
    except Exception as exc:
        errors.append(f"AutoModelForImageTextToText: {exc}")

if model is None:
    print("  模型加载失败，尝试过的方式：")
    for err in errors:
        print("  -", err[:500])
    raise SystemExit("请确认当前 transformers 版本支持 Qwen3.5 多模态模型。")

model = model.to(device).eval()
inputs = inputs.to(device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=160,
        do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

# 只解码新生成的部分，去掉输入 prompt。
generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
answer = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0].strip()

# 有些模型会输出 <think>...</think>，课堂展示时去掉内部推理片段。
answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S).strip()

print("\n六、模型回答")
print("-" * 72)
print(answer)
print("-" * 72)

print("\n七、本节要点")
print("  1. 当前本地 Qwen3.5-0.8B 带 vision_config，可用于图像输入演示。")
print("  2. 多模态 messages 的 content 是列表，可以同时放 image 和 text。")
print("  3. processor 会输出 input_ids、pixel_values、image_grid_thw 等张量。")
print("  4. 模型生成时，图片 token 和文本 token 会一起进入 Transformer。")
print("  5. 可以修改 question，观察同一张图片在不同问题下的回答差异。")