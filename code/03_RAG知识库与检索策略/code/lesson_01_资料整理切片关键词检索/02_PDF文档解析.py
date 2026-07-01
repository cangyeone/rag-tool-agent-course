"""02_PDF文档解析。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
学习目标：把 PDF 资料解析成 RAG 可以继续处理的文本。

这个脚本演示三件事：
1. 选择一份 PDF 文件
2. 按页提取文本，并保留页码、来源等元数据
3. 保存为 JSON 和 TXT，供后续切片、关键词检索、向量检索使用

运行方式：
    python code/03_RAG知识库与检索策略/code/lesson_01_资料整理切片关键词检索/02_PDF文档解析.py

如果要解析自己的 PDF，可以设置相对路径：
    macOS / Linux:
        export RAG_PDF_PATH="docs/example.pdf"

    Windows PowerShell:
        $env:RAG_PDF_PATH="docs/example.pdf"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("=" * 72)
print("02_PDF文档解析 —— 从 PDF 到可入库文本")
print("=" * 72)

# ------------------------------------------------------------
# 一、选择 PDF 文件
# ------------------------------------------------------------
# 课堂默认用课程中已经生成好的 PDF 讲义。
# 如果设置了 RAG_PDF_PATH，就优先解析用户指定的 PDF。

default_pdf = (
    "01_6月22日_RAG工具使用培训/"
    "docs/rendered/"
    "示例业务系统_RAG工具使用培训课程方案_2026年6月22日.pdf"
)

pdf_from_env = os.getenv("RAG_PDF_PATH", "").strip()
pdf_rel_path = pdf_from_env or default_pdf
pdf_path = Path(pdf_rel_path)
if not pdf_path.is_absolute():
    pdf_path = COURSE_ROOT / pdf_path

if not pdf_path.exists():
    print("\n没有找到指定 PDF：", pdf_path)
    print("开始在课程目录中自动查找 PDF。")
    candidates = sorted(COURSE_ROOT.rglob("*.pdf"))
    if not candidates:
        raise SystemExit("课程目录中没有找到 PDF 文件。请把 PDF 放到 rag-tool-agent-course 下再运行。")
    pdf_path = candidates[0]

print("\n一、PDF 文件")
print("  文件路径：", pdf_path.relative_to(COURSE_ROOT))
print("  文件大小：", round(pdf_path.stat().st_size / 1024, 2), "KB")

# ------------------------------------------------------------
# 二、选择 PDF 解析库
# ------------------------------------------------------------
# 课堂环境里可能安装的是 pypdf、PyPDF2 或 PyMuPDF。
# 这里按顺序尝试，哪个能用就用哪个。

parser_name = ""
pages = []

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
                "当前环境没有可用的 PDF 解析库。可安装其中一个：pip install pypdf 或 pip install pymupdf"
            )

print("\n二、解析库")
print("  使用解析库：", parser_name)
print("  PDF 页数：", len(pages))

# ------------------------------------------------------------
# 三、清洗每页文本
# ------------------------------------------------------------
# PDF 提取出来的文本经常有多余空格、连续空行、页眉页脚。
# 这里先做最基础的清洗：合并多余空白，保留页码和来源。

records = []
total_chars = 0

for item in pages:
    raw_text = item["text"]
    clean_text = raw_text.replace("\u00a0", " ")
    clean_text = re.sub(r"[ \t]+", " ", clean_text)
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
    clean_text = clean_text.strip()

    if not clean_text:
        continue

    record = {
        "title": pdf_path.stem,
        "source": str(pdf_path.relative_to(COURSE_ROOT)),
        "page": item["page"],
        "content": clean_text,
        "chars": len(clean_text),
    }
    records.append(record)
    total_chars += len(clean_text)

print("\n三、页级文本预览")
print(f"  有文本页数：{len(records)}")
print(f"  总字符数：{total_chars}")

for record in records[:3]:
    preview = record["content"][:180].replace("\n", " ")
    print(f"\n  第 {record['page']} 页 | {record['chars']} 字符")
    print("  " + preview + ("..." if len(record["content"]) > 180 else ""))

if len(records) > 3:
    print(f"\n  其余 {len(records) - 3} 页已省略预览。")

# ------------------------------------------------------------
# 四、保存解析结果
# ------------------------------------------------------------
# JSON 适合程序继续处理，因为它保留了 title/source/page/content。
# TXT 适合人工检查，也可以直接给后续切片脚本做输入。

lesson_dir = COURSE_ROOT / (
    "code/03_RAG知识库与检索策略/"
    "code/lesson_01_资料整理切片关键词检索"
)
knowledge_dir = lesson_dir / "knowledge_base"
knowledge_dir.mkdir(parents=True, exist_ok=True)

json_path = knowledge_dir / "parsed_pdf_pages.json"
txt_path = knowledge_dir / "parsed_pdf_text.txt"

output = {
    "parsed_at": datetime.now().isoformat(timespec="seconds"),
    "parser": parser_name,
    "pdf": str(pdf_path.relative_to(COURSE_ROOT)),
    "page_count": len(pages),
    "text_page_count": len(records),
    "total_chars": total_chars,
    "pages": records,
}

json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

txt_blocks = []
for record in records:
    txt_blocks.append(
        f"【来源】{record['source']}\n"
        f"【页码】第 {record['page']} 页\n"
        f"{record['content']}"
    )
txt_path.write_text(("\n\n" + "=" * 60 + "\n\n").join(txt_blocks), encoding="utf-8")

print("\n四、保存结果")
print("  JSON：", json_path.relative_to(COURSE_ROOT))
print("  TXT ：", txt_path.relative_to(COURSE_ROOT))

# ------------------------------------------------------------
# 五、入库前检查
# ------------------------------------------------------------

print("\n五、入库前检查")
if not records:
    print("  没有提取到可用文本。这个 PDF 可能是扫描件，需要 OCR。")
else:
    avg_chars = total_chars / len(records)
    print("  平均每页字符数：", round(avg_chars, 1))
    print("  后续建议：")
    print("  1. 先人工查看 parsed_pdf_text.txt，确认文字顺序是否正常。")
    print("  2. 再把 pages 里的 content 交给切片脚本。")
    print("  3. 如果是扫描件或图片型 PDF，需要先做 OCR，再进入切片。")

print("\n" + "=" * 72)
print("学习要点：")
print("  1. PDF 解析不是 RAG 的终点，只是把资料变成可处理文本。")
print("  2. 页码、来源、标题要保留，后面回答才能给出处。")
print("  3. 解析后必须抽查文本质量，避免乱码、漏页、顺序错乱。")