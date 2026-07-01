"""
04 阿里百炼多模态图像解释。

本节演示 OpenAI 兼容格式中的图像输入：
messages[0]["content"] 里同时放 text 和 image_url。

脚本会先在本地生成一张课堂演示图片，再把图片转成 Data URL 传给 qwen3.7-plus。
"""

from __future__ import annotations

import base64
import json
import os
import mimetypes
from pathlib import Path

import requests

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("04 阿里百炼多模态图像解释")
print("=" * 72)

API_BASE = os.getenv("BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("BAILIAN_MODEL", "qwen-plus")
API_KEY = os.getenv("BAILIAN_API_KEY", "").strip()
URL = API_BASE.rstrip("/") + "/chat/completions"

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_PATH = SCRIPT_DIR / "_demo_train_board.png"


def create_demo_image(path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1100, 620), "white")
    draw = ImageDraw.Draw(image)

    try:
        font_big = ImageFont.truetype("Arial.ttf", 60)
        font_mid = ImageFont.truetype("Arial.ttf", 42)
        font_small = ImageFont.truetype("Arial.ttf", 34)
    except Exception:
        font_big = ImageFont.load_default()
        font_mid = ImageFont.load_default()
        font_small = ImageFont.load_default()

    draw.rectangle((0, 0, 1100, 100), fill=(0, 77, 180))
    draw.text((40, 24), "Station Service Board", fill="white", font=font_big)

    rows = [
        ("Train", "G107"),
        ("From", "Beijing South"),
        ("To", "Shanghai Hongqiao"),
        ("Gate", "A12"),
        ("Platform", "3"),
        ("Time", "13:42"),
    ]

    y = 150
    for label, value in rows:
        draw.text((90, y), label, fill=(60, 60, 60), font=font_mid)
        draw.text((420, y), value, fill=(0, 55, 150), font=font_mid)
        y += 70

    draw.rectangle((70, 535, 1030, 580), outline=(240, 110, 0), width=3)
    draw.text((90, 545), "Demo image for multimodal model input", fill=(80, 80, 80), font=font_small)

    image.save(path)


def image_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


if not IMAGE_PATH.exists():
    create_demo_image(IMAGE_PATH)

image_data_url = image_to_data_url(IMAGE_PATH)

prompt = (
    "请读取图片中的信息，回答订单编号、出发站、到达站、办理窗口、服务窗口和时间。"
    "请用中文分点回答。"
)

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
    ],
    "enable_thinking": True,
    "temperature": 0.2,
    "max_tokens": 700,
    "stream": False,
}

print("接口地址：", URL)
print("模型名称：", MODEL)
print("图片路径：", IMAGE_PATH.relative_to(COURSE_ROOT))
print("密钥状态：", "已填写（不打印明文）" if API_KEY else "未填写")

print("\n请求体预览：")
preview = dict(payload)
preview_messages = json.loads(json.dumps(payload["messages"], ensure_ascii=False))
preview_messages[0]["content"][1]["image_url"]["url"] = "data:image/png;base64,<省略很长的图片内容>"
preview["messages"] = preview_messages
print(json.dumps(preview, ensure_ascii=False, indent=2))

if not API_KEY:
    print("\nAPI_KEY 为空，本次只生成图片并展示请求体，不发送真实请求。")
    print("课堂演示时，把百炼 API Key 填入脚本顶部的 API_KEY 变量即可。")
    raise SystemExit(0)

response = requests.post(
    URL,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json=payload,
    timeout=120, 
)

print("\nHTTP 状态码：", response.status_code)

if response.status_code != 200:
    print("请求失败：")
    print(response.text[:2000])
    raise SystemExit(1)

data = response.json()
print("\n请求响应：")
print(data)
answer = data["choices"][0]["message"].get("content", "")

print("\n模型回答：")
print(answer)

print("\nToken 用量：")
print(json.dumps(data.get("usage", {}), ensure_ascii=False, indent=2))

print("\n本节要点：")
print("1. 多模态输入把 text 和 image_url 放在同一个 content 列表里。")
print("2. 本地图片需要先转成 Data URL，或上传到可访问的 HTTPS 地址。")
print("3. 图像理解结果仍然通过 choices[0].message.content 返回。")