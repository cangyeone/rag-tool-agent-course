"""01_资料清单盘点。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
学习目标：理解 RAG 项目的第一步——资料盘点。掌握如何扫描、登记、
         校验知识库文档，确保入库资料完整可用。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import json
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")
from datetime import datetime


# ── 从课程目录加载真实文档 ──
def _load_real_docs():
    real_docs = []
    sample_files = [
        ("code/01_AI基础与模型发展/README.md", 350),
        ("code/02_大模型接口与业务指令/README.md", 350),
        ("code/03_RAG知识库与检索策略/README.md", 350),
        ("README.md", 350),
    ]
    for rel_path, max_chars in sample_files:
        file_path = COURSE_ROOT / rel_path
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")[:max_chars]
            title = rel_path.split("/")[1] if rel_path.startswith("code/") else "课程总览"
            real_docs.append({"title": title, "source": rel_path, "content": content})
    return real_docs

docs = _load_real_docs()


print("=" * 72)
print("01_资料清单盘点 —— RAG 知识库第一步")
print("=" * 72)

# ── 第一步：扫描资料目录（模拟文件系统扫描） ──
print("\n一、模拟扫描资料目录")
print("  扫描路径：./knowledge_base/")
knowledge_dir = COURSE_ROOT / "code/03_RAG知识库与检索策略/code/lesson_01_资料整理切片关键词检索/knowledge_base"
knowledge_dir.mkdir(exist_ok=True)
file_types = {".txt": 0, ".pdf": 0, ".docx": 0, ".md": 0, ".json": 0}
print(f"  发现目录：{knowledge_dir}")
print(f"  实际目录存在：{knowledge_dir.exists()}，条目数：{len(list(knowledge_dir.iterdir())) if knowledge_dir.exists() else 0}")

# ── 第二步：盘点文档清单 ──
print("\n二、文档清单盘点")
print(f"  {'序号':<6}{'标题':<16}{'来源':<10}{'字数':<6}{'创建时间'}")
print("  " + "-" * 56)
total_chars = 0
for i, doc in enumerate(docs, 1):
    word_count = len(doc["content"])
    total_chars += word_count
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"  {i:<6}{doc['title']:<16}{doc['source']:<10}{word_count:<6}{ts}")
print(f"\n  合计：{len(docs)} 篇文档，{total_chars} 字符")

# ── 第三步：入库前校验（完整性检查） ──
print("\n三、入库前完整性校验")
required_fields = ["title", "source", "content"]
checklist = []
for doc in docs:
    missing = [f for f in required_fields if not doc.get(f)]
    issues = []
    if not doc.get("title"):
        issues.append("缺少标题")
    if not doc.get("source"):
        issues.append("缺少来源")
    if not doc.get("content"):
        issues.append("缺少正文")
    if len(doc.get("content", "")) < 10:
        issues.append("正文过短（<10字）")
    status = "✓ 合格" if not issues else f"✗ {'; '.join(issues)}"
    checklist.append((doc["title"], status))

print(f"  {'文档名称':<16}{'校验结果'}")
print("  " + "-" * 36)
for title, status in checklist:
    print(f"  {title:<16}{status}")

# ── 第四步：内容质量初筛 ──
print("\n四、内容质量初筛指标")
for doc in docs:
    content = doc["content"]
    has_chinese = any('\u4e00' <= ch <= '\u9fff' for ch in content)
    has_digit = any(ch.isdigit() for ch in content)
    has_示例业务系统 = "示例业务系统" in content
    quality_flags = []
    if has_chinese:
        quality_flags.append("含中文")
    if has_digit:
        quality_flags.append("含数字")
    if has_示例业务系统:
        quality_flags.append("引用示例业务系统")
    print(f"  {doc['title']}: {', '.join(quality_flags) if quality_flags else '无特征'}")

# ── 第五步：生成盘点报告 ──
print("\n五、生成盘点报告")
report = {
    "盘点时间": datetime.now().isoformat(),
    "文档总数": len(docs),
    "总字符数": total_chars,
    "校验通过": sum(1 for _, s in checklist if s.startswith("✓")),
    "校验异常": sum(1 for _, s in checklist if s.startswith("✗")),
    "文档列表": [{"title": d["title"], "chars": len(d["content"])} for d in docs]
}
report_path = knowledge_dir / "inventory_report.json"
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  报告已保存至：{report_path}")
print(f"  报告内容：{json.dumps(report, ensure_ascii=False)}")

print("\n" + "=" * 72)
print("学习要点：")
print("  1. 文件系统扫描 → 发现知识来源")
print("  2. 字段完整性校验 → 确保入库数据可用")
print("  3. 内容质量检查 → 过滤低质量文档")
print("  4. 生成盘点报告 → 可追溯、可审计")