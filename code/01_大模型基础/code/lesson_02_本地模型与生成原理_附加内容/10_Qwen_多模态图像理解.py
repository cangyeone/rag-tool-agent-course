"""10 Qwen 多模态图像理解。

学习目标：使用本地 Qwen-VL 模型对行业场景图片进行理解和分析。
了解多模态模型的输入格式（文本 + 图像）、推理方式，以及与纯文本模型的区别。

运行方式：
  python 10_Qwen_多模态图像理解.py
  python 10_Qwen_多模态图像理解.py --image /path/to/your/image.jpg

依赖：本地需有 Qwen2.5-VL 或 Qwen2-VL 模型目录，否则脚本会给出下载指引。
"""

from __future__ import annotations


import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import base64
import io
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")
from typing import Optional

import torch
from PIL import Image, ImageDraw, ImageFont

print("10 Qwen 多模态——本地 VL 模型图像理解")
print("=" * 72)


# ========== 1. 查找本地多模态模型 ==========

def find_vl_model() -> Optional[Path]:
    """在课程根目录 open_models 下查找可用的 VL 模型。"""
    vl_candidates = [
        "Qwen2.5-VL-7B-Instruct",
        "Qwen2.5-VL-3B-Instruct",
        "Qwen2-VL-7B-Instruct",
        "Qwen2-VL-2B-Instruct",
        "Qwen-VL-Chat",
    ]
    open_models = COURSE_ROOT / "open_models"
    for name in vl_candidates:
        candidate = open_models / name
        if candidate.exists() and (candidate / "config.json").exists():
            return candidate
    return None


model_dir = find_vl_model()

if model_dir is None:
    print("\n⚠ 未找到本地 Qwen-VL 模型。")
    print("  本脚本需要用视觉语言模型处理真实图片。")
    print()
    print("  下载方式（选一个）：")
    print("    # 推荐：Qwen2.5-VL-3B（显存友好，约 6GB）")
    print("    pip install huggingface_hub")
    print("    huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \\")
    print("        --local-dir open_models/Qwen2.5-VL-3B-Instruct")
    print()
    print("    # 更大：Qwen2.5-VL-7B（更强推理，约 15GB）")
    print("    huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \\")
    print("        --local-dir open_models/Qwen2.5-VL-7B-Instruct")
    print()
    print("  下面会用模拟方式演示多模态处理流程。")
    print("=" * 72)
    VL_AVAILABLE = False
else:
    print(f"  找到本地 VL 模型：{model_dir.name}")
    print(f"  路径：{model_dir}")
    VL_AVAILABLE = True


# ========== 2. 准备/生成演示图片 ==========

def create_demo_image() -> Image.Image:
    """生成一张模拟的行业场景图片，用于演示（当没有真实图片时）。"""
    img = Image.new("RGB", (640, 400), color=(200, 220, 240))
    draw = ImageDraw.Draw(img)

    # 天空
    draw.rectangle([0, 0, 640, 150], fill=(180, 210, 240))

    # 轨道
    draw.rectangle([0, 280, 640, 400], fill=(160, 140, 120))
    draw.rectangle([0, 290, 640, 295], fill=(100, 100, 100))
    draw.rectangle([0, 310, 640, 315], fill=(100, 100, 100))

    # 枕木
    for x in range(0, 640, 30):
        draw.rectangle([x, 280, x + 10, 320], fill=(120, 80, 60))

    # 信号灯
    draw.rectangle([200, 100, 210, 280], fill=(80, 80, 80))
    draw.ellipse([190, 90, 220, 120], fill=(255, 50, 50))  # 红灯
    draw.ellipse([190, 125, 220, 155], fill=(50, 50, 50))   # 灭

    # 指示牌
    draw.rectangle([400, 150, 520, 200], fill=(0, 100, 0))
    draw.text((410, 160), "K128+450", fill=(255, 255, 255))
    draw.text((410, 178), "限速 45km/h", fill=(255, 255, 255))

    # 围栏
    for y in range(150, 280, 20):
        draw.rectangle([80, y, 85, y + 15], fill=(100, 100, 100))
        draw.rectangle([550, y, 555, y + 15], fill=(100, 100, 100))

    return img


# ========== 3. 加载模型 ==========

if VL_AVAILABLE:
    print("\n一、加载 Qwen-VL 模型")
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备：{device}")

    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        dtype = torch.float16 if device != "cpu" else torch.float32
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            device_map="auto" if device != "mps" else None,
        )
        if device == "mps":
            model = model.to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(model_dir)
        print(f"  模型加载完成，参数量约 {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
        print(f"  processor 类型：{type(processor).__name__}")
        MODEL_LOADED = True
    except ImportError:
        print("  ✗ qwen-vl-utils 或 transformers 版本不支持 Qwen2VL，回退到旧版 API")
        MODEL_LOADED = False
else:
    MODEL_LOADED = False


# ========== 4. 图像推理 ==========

def run_vl_inference(image: Image.Image, question: str) -> str:
    """使用 Qwen-VL 模型进行图像理解。"""
    if not MODEL_LOADED:
        return _simulate_vl_response(image, question)

    from qwen_vl_utils import process_vision_info

    # 构建 messages：包含图像和文本
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    # 用 processor 处理
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].strip()


def _simulate_vl_response(image: Image.Image, question: str) -> str:
    """无 VL 模型时，使用本地 Qwen3.5-0.8B 文本模型分析图像描述。"""
    width, height = image.size
    samples = {
        "top_center": image.getpixel((width // 2, 20)),
        "mid_center": image.getpixel((width // 2, height // 2)),
        "bottom_center": image.getpixel((width // 2, height - 20)),
    }

    def classify_color(rgb):
        r, g, b = rgb[:3]
        if r > 200 and g < 100 and b < 100:
            return "红色（可能是信号灯/警示标志）"
        if g > 150 and r < 150 and b < 150:
            return "绿色（可能是植被/指示牌）"
        if r > 180 and g > 180 and b > 180:
            return "浅色（可能是天空/建筑）"
        if max(r, g, b) - min(r, g, b) < 30:
            return "灰色调（可能是轨道/混凝土设施）"
        return "混合色"

    color_desc = {k: classify_color(v) for k, v in samples.items()}

    # 尝试用本地 Qwen 文本模型做推理
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        qwen_path = COURSE_ROOT / "open_models" / "Qwen3.5-0.8B"
        if not qwen_path.is_dir():
            qwen_path = None
        if qwen_path:
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if device != "cpu" else torch.float32
            tokenizer = AutoTokenizer.from_pretrained(qwen_path)
            model = AutoModelForCausalLM.from_pretrained(qwen_path, torch_dtype=dtype).to(device).eval()

            desc = (
                f"你是一个行业场景分析助手。这是一张 {width}x{height} 的图片的像素采样信息：\n"
                f"上部色彩：{color_desc['top_center']}\n"
                f"中部色彩：{color_desc['mid_center']}\n"
                f"下部色彩：{color_desc['bottom_center']}\n"
                f"请根据这些色彩信息和问题「{question}」给出你的分析。"
            )
            messages = [{"role": "user", "content": desc}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=200, temperature=0.3, do_sample=True,
                                            pad_token_id=tokenizer.eos_token_id)
            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            return tokenizer.decode(generated, skip_special_tokens=True).strip()
    except Exception:
        pass

    return (
        f"（无 VL 模型且文本模型不可用）图片尺寸：{width}×{height}，色彩分布："
        f"上{color_desc['top_center']}，中{color_desc['mid_center']}，下{color_desc['bottom_center']}"
    )


# ========== 5. 运行演示 ==========

arg_parser = argparse.ArgumentParser(description="Qwen-VL 多模态图像理解演示")
arg_parser.add_argument("--image", type=str, default=None, help="要分析的图片路径（可选）")
args = arg_parser.parse_args()

# 准备图片
if args.image:
    image_path = Path(args.image)
    if image_path.exists():
        demo_image = Image.open(image_path).convert("RGB")
        print(f"\n二、使用用户图片：{image_path}")
        print(f"  尺寸：{demo_image.size[0]}×{demo_image.size[1]}")
    else:
        print(f"\n⚠ 图片 {args.image} 不存在，使用演示图片")
        demo_image = create_demo_image()
else:
    print("\n二、使用演示图片（模拟行业场景）")
    demo_image = create_demo_image()

# 保存演示图片，方便后续查看
demo_path = COURSE_ROOT / "code/01_大模型基础/code/lesson_02_本地模型与生成原理/_demo_general_scene.jpg"
demo_image.save(demo_path)
print(f"  图片已保存至：{demo_path.name}")

# ========== 6. 多轮提问演示 ==========

questions = [
    "请描述这张图片中的行业场景，包括可见的设施和标识。",
    "图中是否有信号灯？它显示的是什么状态？",
    "根据里程标和限速牌，这个区段需要注意什么安全事项？",
]

print("\n三、多轮图像理解演示")
print("─" * 48)

for i, question in enumerate(questions, 1):
    print(f"\n  【第 {i} 轮】")
    print(f"  提问：{question}")
    print(f"  模型回答：")
    answer = run_vl_inference(demo_image, question)

    # 格式化输出回答
    for line in answer.split("\n"):
        if line.strip():
            print(f"    {line.strip()}")

print("\n─" * 48)

# ========== 7. 多模态 vs 纯文本对比 ==========

print("\n四、多模态模型 vs 纯文本模型的区别")

comparison = [
    ("输入格式",   "纯文本",              "messages = [{'role':'user','content':'text'}]",
                   "多模态",              "messages = [{'role':'user','content':[{'type':'image'},{'type':'text'}]}]"),
    ("处理方式",   "纯文本",              "tokenizer 直接把文字切成 token",
                   "多模态",              "processor 先把图像切分成 patch，转成视觉 token，再和文本 token 拼接"),
    ("模型架构",   "纯文本",              "只有 Transformer Decoder（因果注意力）",
                   "多模态",              "Transformer Decoder + Vision Encoder（先编码图像再送入语言模型）"),
    ("适用场景",   "纯文本",              "对话、文档分析、代码生成、RAG 问答",
                   "多模态",              "图像描述、OCR、图表理解、工业检测、安全检查"),
    ("示例业务系统 场景", "纯文本",              "客服问答、规则解释、服务编号查询、退款与变更说明",
                   "多模态",              "闸机异常图片分析、服务窗口监控理解、设备状态识别、巡检图像辅助判断"),
]

for group, text_label, text_desc, multimodal_label, multimodal_desc in comparison:
    print(f"\n  ■ {group}")
    print(f"    {text_label}：{text_desc}")
    print(f"    {multimodal_label}：{multimodal_desc}")

# ========== 8. 行业场景应用 ==========

print("\n\n五、多模态在 示例业务系统/行业场景中的应用")

scenarios = [
    {
        "场景": "服务网点监控分析",
        "输入": "服务窗口摄像头截图 + 文字'当前时段拥挤程度如何？'",
        "输出": "识别服务窗口上的人数密度，判断是否需要限流",
        "价值": "替代人工 24 小时盯屏，自动发现异常并告警",
    },
    {
        "场景": "设备状态识别",
        "输入": "信号机/道岔照片 + 文字'设备外观是否正常？'",
        "输出": "识别设备外观异常（破损、锈蚀、倾斜），标记需检查项",
        "价值": "巡检照片自动初筛，降低人工审查工作量",
    },
    {
        "场景": "巡检图像辅助",
        "输入": "轨道扣件照片 + 文字'扣件是否缺失或松动？'",
        "输出": "检测扣件状态，标注异常位置",
        "价值": "配合探伤车图像，快速定位可疑区段",
    },
    {
        "场景": "证件信息提取",
        "输入": "身份证/学生证照片 + 文字'提取姓名和证件号'",
        "输出": "OCR 识别 + 结构化提取关键字段",
        "价值": "学生优惠核验、临时身份证明等场景自动化",
    },
    {
        "场景": "乘客行为分析",
        "输入": "车厢内照片 + 文字'是否有乘客需要帮助？'",
        "输出": "识别乘客异常行为（晕倒、行李遗落、占座冲突迹象）",
        "价值": "提前发现客服介入时机，提升服务质量",
    },
]

for s in scenarios:
    print(f"\n  【{s['场景']}】")
    print(f"    输入：{s['输入']}")
    print(f"    输出：{s['输出']}")
    print(f"    价值：{s['价值']}")

# ========== 9. 总结 ==========

print("\n\n六、本节要点")
print("  1. 多模态模型 = Vision Encoder + Language Model，在处理文本的同时理解图像")
print("  2. 输入格式从纯 text 变为 [image, text] 列表，processor 负责将图像转成视觉 token")
print("  3. 本地 VL 模型可在内网环境处理敏感图片，数据不出服务器")
print("  4. 行业场景中，VL 模型可以用于监控理解、设备识别、巡检辅助、证件提取等")
print("  5. Qwen2.5-VL-3B 仅需 6GB 显存，单张消费级显卡即可运行")

if not VL_AVAILABLE:
    print(f"\n  → 本次为模拟模式。下载模型后可获得真实推理结果。")
    print(f"  → 演示图片已保存至 {demo_path}")