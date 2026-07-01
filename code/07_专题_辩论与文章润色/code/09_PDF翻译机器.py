"""09 PDF 翻译机器。

基于 PDF 解析 + DeepSeek API 的文档翻译工具。

功能：
1. 输入 PDF 文件路径，自动解析提取文本
2. 通过 DeepSeek API 进行翻译（翻译 Agent 定义在 system prompt 中）
3. 长文档自动分块翻译，避免超出 token 限制
4. 输出翻译后的 Markdown 文件
5. 可配置源语言和目标语言
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import requests

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
URL = BASE_URL + "/chat/completions"
USER_ID = (
    os.getenv("DEEPSEEK_USER_ID")
    or os.getenv("CLASSROOM_USER_ID")
    or os.getenv("USERNAME")
    or os.getenv("USER")
    or "classroom_user"
).strip()

SOURCE_LANG = os.getenv("TRANSLATE_SOURCE_LANG", "").strip() or "自动检测"
TARGET_LANG = os.getenv("TRANSLATE_TARGET_LANG", "").strip() or "中文"
CHUNK_SIZE = int(os.getenv("TRANSLATE_CHUNK_SIZE", "2000"))

if not API_KEY:
    raise SystemExit(
        "未设置 DEEPSEEK_API_KEY。\n"
        "macOS/Linux: export DEEPSEEK_API_KEY=your_api_key_here\n"
        "Windows PowerShell: $env:DEEPSEEK_API_KEY=\"YOUR_DEEPSEEK_API_KEY\""
    )


# ------------------------------------------------------------
# 一、PDF 解析
# ------------------------------------------------------------

def parse_pdf(pdf_path: Path) -> list[dict]:
    """解析 PDF，返回每页文本记录列表。依次尝试 pypdf / PyPDF2 / PyMuPDF。"""
    pages = []
    parser_name = ""

    try:
        from pypdf import PdfReader
        parser_name = "pypdf"
        reader = PdfReader(str(pdf_path))
        for page_index, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            pages.append({"page": page_index, "text": text})
    except ImportError:
        try:
            from PyPDF2 import PdfReader
            parser_name = "PyPDF2"
            reader = PdfReader(str(pdf_path))
            for page_index, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                pages.append({"page": page_index, "text": text})
        except ImportError:
            try:
                import fitz
                parser_name = "PyMuPDF"
                doc = fitz.open(str(pdf_path))
                for page_index, page in enumerate(doc, 1):
                    text = page.get_text("text") or ""
                    pages.append({"page": page_index, "text": text})
                doc.close()
            except ImportError:
                raise SystemExit(
                    "没有可用的 PDF 解析库。请安装：pip install pypdf 或 pip install pymupdf"
                )

    records = []
    total_chars = 0

    for item in pages:
        raw = item["text"]
        clean = raw.replace("\u00a0", " ")
        clean = re.sub(r"[ \t]+", " ", clean)
        clean = re.sub(r"\n{3,}", "\n\n", clean)
        clean = clean.strip()
        if not clean:
            continue

        records.append({
            "page": item["page"],
            "content": clean,
            "chars": len(clean),
        })
        total_chars += len(clean)

    print(f"  解析库：{parser_name}  |  PDF 页数：{len(pages)}  |  有文本页数：{len(records)}  |  总字符数：{total_chars}")
    return records


# ------------------------------------------------------------
# 二、文本分块
# ------------------------------------------------------------

def split_into_chunks(records: list[dict], chunk_size: int = 2000) -> list[str]:
    """将 PDF 记录按自然段落切分成翻译块，每块不超过 chunk_size 字符。"""
    chunks = []
    current_chunk = ""
    current_pages = []

    for record in records:
        text = record["content"]
        page = record["page"]

        paragraphs = re.split(r"\n\s*\n", text)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if current_chunk and len(current_chunk) + len(para) + 2 > chunk_size:
                page_range = f"第{current_pages[0]}-{current_pages[-1]}页" if len(current_pages) > 1 else f"第{current_pages[0]}页"
                chunks.append(f"[{page_range}]\n{current_chunk.strip()}")
                current_chunk = ""
                current_pages = []

            current_chunk += para + "\n\n"
            if page not in current_pages:
                current_pages.append(page)

    if current_chunk.strip():
        page_range = f"第{current_pages[0]}-{current_pages[-1]}页" if len(current_pages) > 1 else f"第{current_pages[0]}页"
        chunks.append(f"[{page_range}]\n{current_chunk.strip()}")

    return chunks


# ------------------------------------------------------------
# 三、翻译 Agent（system prompt）
# ------------------------------------------------------------

TRANSLATOR_SYSTEM = {
    "role": "system",
    "content": (
        f"你是一位专业的文档翻译专家。\n"
        f"源语言：{SOURCE_LANG}\n"
        f"目标语言：{TARGET_LANG}\n\n"
        "翻译要求：\n"
        "1. 保持原文的语义和风格，不要添加或删除内容。\n"
        "2. 专业术语翻译准确，保持一致。\n"
        "3. 保留原文中的 Markdown 格式（标题、列表、代码块等）。\n"
        "4. 保留原文中的页码标记（如 [第X页]）。\n"
        "5. 直接输出翻译后的文本，不要添加任何解释或说明。\n"
        "6. 遇到数字、日期、专有名词等，保持原样或根据目标语言习惯转换。"
    ),
}


# ------------------------------------------------------------
# 四、DeepSeek API 调用
# ------------------------------------------------------------

def translate_chunk(chunk_text: str, chunk_index: int, total: int,
                     max_tokens: int = 4000) -> str:
    """翻译单个文本块，使用流式输出。"""
    print(f"\n--- 翻译第 {chunk_index}/{total} 块 ({len(chunk_text)} 字符) ---")

    messages = [
        TRANSLATOR_SYSTEM,
        {"role": "user", "content": f"请将以下文本翻译为{TARGET_LANG}：\n\n{chunk_text}"},
    ]

    payload = {
        "model": MODEL,
        "messages": messages,
        "thinking": {"type": "disabled"},
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "stream": True,
        "user_id": USER_ID,
    }

    start = time.time()
    response = requests.post(
        URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
        stream=True,
    )

    if response.status_code != 200:
        print(f"请求失败 HTTP {response.status_code}")
        print(response.text[:1000])
        return f"[翻译失败: 第{chunk_index}块]"

    translated = ""
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line or raw_line == "data: [DONE]":
            continue
        if not raw_line.startswith("data: "):
            continue

        chunk_data = json.loads(raw_line[len("data: "):])
        delta = chunk_data.get("choices", [{}])[0].get("delta", {})
        text_piece = delta.get("content") or ""

        if text_piece:
            print(text_piece, end="", flush=True)
            translated += text_piece

    elapsed = time.time() - start
    print(f"\n(耗时 {elapsed:.1f}s)")

    return translated.strip()


# ------------------------------------------------------------
# 五、主流程
# ------------------------------------------------------------

print("=" * 72)
print("09 PDF 翻译机器")
print("=" * 72)
print("模型：", MODEL)
print("源语言：", SOURCE_LANG)
print("目标语言：", TARGET_LANG)
print("分块大小：", CHUNK_SIZE, "字符")
print()

# 输入 PDF 路径
pdf_from_env = os.getenv("TRANSLATE_PDF_PATH", "").strip()
if pdf_from_env:
    pdf_path = Path(pdf_from_env)
    if not pdf_path.is_absolute():
        pdf_path = COURSE_ROOT / pdf_path
else:
    # 命令行参数
    import sys
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
        if not pdf_path.is_absolute():
            pdf_path = COURSE_ROOT / pdf_path
    else:
        raise SystemExit(
            "请指定要翻译的 PDF 文件：\n"
            "  方式1: export TRANSLATE_PDF_PATH=docs/paper.pdf\n"
            "  方式2: python 脚本名.py docs/paper.pdf"
        )

if not pdf_path.exists():
    raise SystemExit(f"文件不存在：{pdf_path}")

print(f"PDF 文件：{pdf_path.relative_to(COURSE_ROOT)}")
print(f"文件大小：{round(pdf_path.stat().st_size / 1024, 2)} KB")

# 解析 PDF
print("\n解析 PDF...")
records = parse_pdf(pdf_path)

if not records:
    raise SystemExit("PDF 中没有可提取的文本。可能是扫描件，需要 OCR 处理。")

full_text = "\n\n".join(r["content"] for r in records)
print(f"\n原始文本总字符数：{len(full_text)}")

# 分块
if len(full_text) <= CHUNK_SIZE:
    chunks = [full_text]
else:
    chunks = split_into_chunks(records, CHUNK_SIZE)

print(f"分块数：{len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"  第{i+1}块: {len(chunk)} 字符")
    preview = chunk[:100].replace("\n", " ")
    print(f"    预览: {preview}...")

# 翻译
print("\n开始翻译...")
translated_chunks = []
total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

for i, chunk in enumerate(chunks, 1):
    result = translate_chunk(chunk, i, len(chunks))
    translated_chunks.append(result)

# 合并结果
translated_full = "\n\n---\n\n".join(translated_chunks)

# 保存结果
output_dir = COURSE_ROOT / "code/07_专题_辩论与文章润色/code/output"
output_dir.mkdir(parents=True, exist_ok=True)

safe_name = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", pdf_path.stem)[:50]
md_path = output_dir / f"translated_{safe_name}.md"

md_content = f"""# {pdf_path.stem}（翻译）

> 源语言：{SOURCE_LANG}  →  目标语言：{TARGET_LANG}
> 原始文件：{pdf_path.relative_to(COURSE_ROOT)}
> 翻译时间：{time.strftime('%Y-%m-%d %H:%M:%S')}
> 总字符数：{len(full_text)}  →  翻译后字符数：{len(translated_full)}

---

{translated_full}
"""

md_path.write_text(md_content, encoding="utf-8")

print()
print("=" * 72)
print("翻译完成")
print("=" * 72)
print(f"原始字符数：{len(full_text)}")
print(f"翻译后字符数：{len(translated_full)}")
print(f"输出文件：{md_path.relative_to(COURSE_ROOT)}")

print("\n课堂观察点：")
print("1. PDF 解析依次尝试 pypdf / PyPDF2 / PyMuPDF，哪个能用用哪个。")
print("2. 翻译 Agent 的定义全部放在 system prompt 中，包括语言方向、风格要求。")
print("3. 长文档自动按自然段落分块，每块约 2000 字符，避免超出 token 限制。")
print("4. 使用流式输出（stream=True）逐字显示翻译进度。")
print("5. 输出为 Markdown 格式，保留页码标记和文档结构。")