"""parser —— PDF/Markdown 文档解析与文本切片。

提供函数：
    parse_pdf(filepath: Path) -> list[dict]
        解析 PDF，按页提取文本，保留页码、来源等元数据。

    parse_markdown(filepath: Path) -> list[dict]
        解析 Markdown，按标题层级分段提取文本。

    chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]
        将长文本按段落边界切分，长段落用滑动窗口切分，保留 overlap。

    collect_documents(knowledge_dir: Path) -> list[dict]
        遍历 knowledge 目录，收集所有 .pdf/.md 文件并解析为文档记录。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def _file_hash(filepath: Path) -> str:
    """计算文件 MD5 哈希，用于增量更新检测。"""
    return hashlib.md5(filepath.read_bytes()).hexdigest()


def parse_pdf(filepath: Path) -> list[dict]:
    """解析 PDF 文件，返回每页的文本记录。

    按顺序尝试 pypdf → PyPDF2 → PyMuPDF，使用第一个可用的解析库。
    每页生成一条记录，包含 title / source / page / content / chars。
    """
    pages: list[dict] = []
    parser_name = ""

    try:
        from pypdf import PdfReader
        parser_name = "pypdf"
        reader = PdfReader(str(filepath))
        for page_index, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            pages.append({"page": page_index, "text": text})
    except ImportError:
        try:
            from PyPDF2 import PdfReader
            parser_name = "PyPDF2"
            reader = PdfReader(str(filepath))
            for page_index, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                pages.append({"page": page_index, "text": text})
        except ImportError:
            try:
                import fitz
                parser_name = "PyMuPDF"
                doc = fitz.open(str(filepath))
                for page_index, page in enumerate(doc, 1):
                    text = page.get_text("text") or ""
                    pages.append({"page": page_index, "text": text})
                doc.close()
            except ImportError:
                return []

    records: list[dict] = []
    for item in pages:
        raw_text = item["text"]
        clean_text = raw_text.replace("\u00a0", " ")
        clean_text = re.sub(r"[ \t]+", " ", clean_text)
        clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
        clean_text = clean_text.strip()
        if not clean_text:
            continue
        records.append({
            "title": filepath.stem,
            "source": str(filepath),
            "page": item["page"],
            "content": clean_text,
            "chars": len(clean_text),
        })
    return records


def parse_markdown(filepath: Path) -> list[dict]:
    """解析 Markdown 文件，按标题层级分段。

    将 ## 和 # 标题作为分段边界，每段生成一条记录。
    代码块（```）内部不做切分。
    """
    try:
        text = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    if len(text.strip()) < 20:
        return []

    sections: list[dict] = []
    in_code_block = False
    lines = text.split("\n")
    current_title = filepath.stem
    current_lines: list[str] = []

    section_index = 0

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            current_lines.append(line)
            continue

        if not in_code_block and re.match(r"^#{1,3}\s", stripped):
            if current_lines:
                section_text = "\n".join(current_lines).strip()
                if len(section_text) >= 20:
                    section_index += 1
                    sections.append({
                        "title": current_title,
                        "source": str(filepath),
                        "section": section_index,
                        "content": section_text,
                        "chars": len(section_text),
                    })
            current_title = re.sub(r"^#+\s*", "", stripped).strip() or filepath.stem
            current_lines = []
            continue

        current_lines.append(line)

    if current_lines:
        section_text = "\n".join(current_lines).strip()
        if len(section_text) >= 20:
            section_index += 1
            sections.append({
                "title": current_title,
                "source": str(filepath),
                "section": section_index,
                "content": section_text,
                "chars": len(section_text),
            })

    return sections


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 500) -> list[str]:
    """将长文本按段落边界切分，长段落用滑动窗口切分，保留 overlap。

    Args:
        text: 待切分的原始文本。
        chunk_size: 每个块的目标最大字符数。
        overlap: 相邻块之间的重叠字符数。

    Returns:
        切分后的文本块列表。
    """
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip() and len(p.strip()) >= 10
    ]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip() if current else para
            continue
        if current:
            chunks.append(current)
        if len(para) <= chunk_size:
            current = para
        else:
            for i in range(0, max(len(para) - overlap, 1), chunk_size - overlap):
                piece = para[i : i + chunk_size]
                if len(piece) >= 20:
                    chunks.append(piece)
            current = ""
    if current:
        chunks.append(current)

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    merged: list[str] = []
    for idx, chunk in enumerate(chunks):
        if idx == 0:
            merged.append(chunk)
            continue
        prev_tail = chunks[idx - 1][-overlap:]
        merged.append(f"{prev_tail}\n{chunk}")
    return merged


def collect_documents(knowledge_dir: Path) -> list[dict]:
    """遍历 knowledge 目录，收集并解析所有 .pdf 和 .md 文件。

    对每个文件：
      - PDF：按页解析为多条记录
      - Markdown：按标题分段，再对每段切片
    返回统一的文档记录列表，每条记录包含 source / page / content 等字段。

    Args:
        knowledge_dir: 知识库源文件目录路径。

    Returns:
        文档记录列表，每条记录包含 title, source, file_ext, file_hash,
        page, content, chars 字段。
    """
    if not knowledge_dir.is_dir():
        return []

    records: list[dict] = []
    pdf_count = 0
    md_count = 0

    for filepath in sorted(knowledge_dir.rglob("*")):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() == ".pdf":
            pdf_pages = parse_pdf(filepath)
            fhash = _file_hash(filepath)
            for record in pdf_pages:
                record["file_ext"] = ".pdf"
                record["file_hash"] = fhash
                record["title"] = filepath.stem
                record["source"] = str(filepath.relative_to(knowledge_dir.parent))
            records.extend(pdf_pages)
            pdf_count += 1
        elif filepath.suffix.lower() == ".md":
            md_sections = parse_markdown(filepath)
            fhash = _file_hash(filepath)
            for record in md_sections:
                record["file_ext"] = ".md"
                record["file_hash"] = fhash
                record["title"] = filepath.stem
                record["source"] = str(filepath.relative_to(knowledge_dir.parent))
            records.extend(md_sections)
            md_count += 1

    return records


def collect_and_chunk(
    knowledge_dir: Path,
    chunk_size: int = 2000,
    chunk_overlap: int = 500,
) -> list[dict]:
    """遍历 knowledge 目录解析文档并切片。

    等同于 collect_documents() + 对每条记录的 content 执行 chunk_text()，
    每条切片记录保留原文档的 source / title / page / file_hash 等元数据。

    Args:
        knowledge_dir: 知识库源文件目录。
        chunk_size: 切片最大字符数。
        chunk_overlap: 切片重叠字符数。

    Returns:
        切片后的 chunk 记录列表。
    """
    docs = collect_documents(knowledge_dir)
    chunks: list[dict] = []

    for doc in docs:
        chunk_texts = chunk_text(doc["content"], chunk_size=chunk_size, overlap=chunk_overlap)
        for pos, chunk_content in enumerate(chunk_texts, start=1):
            chunks.append({
                "title": doc["title"],
                "source": doc["source"],
                "file_ext": doc.get("file_ext", ""),
                "file_hash": doc.get("file_hash", ""),
                "page": doc.get("page", 1),
                "chunk_index": pos,
                "content": chunk_content,
                "content_length": len(chunk_content),
            })

    return chunks