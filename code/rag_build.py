"""RAG 知识库构建 —— 扫描课程资料，用 BGE-m3 编码，存入 FAISS 向量库。

运行方式：
    cd rag-tool-agent-course
    python code/rag_build.py

输出目录：code/rag_kb/（FAISS 索引 + JSONL 元数据 + NPY 向量矩阵）
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

COURSE_ROOT = Path(__file__).resolve().parent.parent
if not (COURSE_ROOT / "code").is_dir():
    COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

CODE_ROOT = COURSE_ROOT / "code"
MODEL_DIR = COURSE_ROOT / "open_models" / "bge-m3"
KB_DIR = CODE_ROOT / "rag_kb"
FAISS_PATH = KB_DIR / "faiss.index"
NPY_PATH = KB_DIR / "vectors.npy"
JSONL_PATH = KB_DIR / "metadata.jsonl"
META_PATH = KB_DIR / "index_meta.json"

CHUNK_SIZE = 500   # 每块最多 500 字
CHUNK_OVERLAP = 100

print("=" * 60)
print("RAG 知识库构建 —— BGE-m3 + FAISS")
print("=" * 60)

if not MODEL_DIR.is_dir():
    raise SystemExit(f"未找到 BGE-m3 模型：{MODEL_DIR}")


# ========== 1. 收集课程资料 ==========

def chunk_text(text: str, title: str, source: str, doc_type: str) -> list[dict]:
    """将长文本按段落 + 滑动窗口切成 chunks。"""
    chunks = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    for para in paragraphs:
        if len(para) <= CHUNK_SIZE:
            chunks.append({"content": para, "title": title, "source": source, "type": doc_type})
            continue

        # 长段落滑动窗口切分
        for i in range(0, max(len(para) - CHUNK_OVERLAP, 1), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = para[i:i + CHUNK_SIZE]
            if len(chunk) >= 30:
                chunks.append({"content": chunk, "title": title, "source": source, "type": doc_type})

    return chunks


def load_course_docs() -> list[dict]:
    """扫描 code 目录下所有 .md 文件，收集资料。"""
    all_docs = []
    seen_paths: set[Path] = set()

    # ── 1. 递归扫描所有 .md 文件 ──
    md_files = sorted(CODE_ROOT.rglob("*.md"))
    for f in md_files:
        # 跳过隐藏目录、__pycache__、rag_kb 输出目录
        rel = f.relative_to(CODE_ROOT)
        if any(part.startswith(".") or part == "__pycache__" or part == "rag_kb" for part in rel.parts):
            continue
        if f in seen_paths:
            continue
        seen_paths.add(f)
        try:
            text = f.read_text(encoding="utf-8", errors="ignore").strip()
            if not text or len(text) < 30:
                continue
            rel_str = str(f.relative_to(COURSE_ROOT))
            # 根据文件名推断标题
            if f.name == "README.md":
                title = f.parent.name
            else:
                title = f.stem
            # 按类型分类
            if "RAG" in title or "rag" in title.lower() or "检索" in title or "向量" in title:
                doc_type = "RAG知识库"
            elif "工具" in title or "Agent" in title or "agent" in title:
                doc_type = "工具调用"
            elif "模型" in title or "AI" in title or "DeepSeek" in title or "接口" in title:
                doc_type = "模型与接口"
            elif f.suffix == ".md" and f.with_suffix(".py").exists():
                doc_type = "代码说明"
            else:
                doc_type = "课程资料"
            all_docs.extend(chunk_text(text[:2000], title, rel_str, doc_type))
        except Exception:
            pass

    # 业务系统知识
    business = [
        {
            "title": "候补申请规则",
            "content": "候补申请不能保证一定成功。候补申请兑现结果取决于退款、变更、新增席位和排队顺序。用户提交候补申请订单后，需要关注候补申请截止时间和兑现结果，截止时间为服务开始前 2 小时。最终以官方页面显示为准。",
        },
        {
            "title": "退款费用规则",
            "content": "服务开始前 8 天以上免手续费；48 小时至 8 天收价格 5%；24 至 48 小时收 10%；不足 24 小时收 20%。变更后的订单在春运期间不办理退款款。具体费用以订单页面和官方规则为准。",
        },
        {
            "title": "变更规则",
            "content": "服务开始前可免费变更一次。变更后的订单日期在春运期间的不办理退款款。变更只能变更日期、服务编号和服务类型，不能变更起点和终点。",
        },
        {
            "title": "学生优惠票规则",
            "content": "学生优惠票每年 6 月 1 日至 9 月 30 日、12 月 1 日至 3 月 31 日可购买。需完成优惠资质核验。使用服务区间、优惠次数和系统核验结果都会影响能否购买。",
        },
        {
            "title": "服务点信息",
            "content": "服务点A、服务点B、杭州东等服务网点在系统中同时存在中文服务点名称、拼音、服务点编码和城市信息。检索服务点资料时，关键词和语义检索可以结合使用。",
        },
        {
            "title": "客服质检标准",
            "content": "客服质检关注：是否准确理解问题、是否引用有效规则、是否避免过度承诺、是否给出可操作建议，以及是否提醒用户以官方页面和业务系统为准。",
        },
        {
            "title": "GPR 安全检查",
            "content": "地质雷达图像分析需要结合里程、病害位置、图像异常、人工复核记录和维修处置建议。多模态模型可以辅助解释图像，但最终结论需要专业人员确认。",
        },
        {
            "title": "RAG 基本流程",
            "content": "RAG（检索增强生成）的核心流程：用户提问 → 检索相关文档 → 将文档作为上下文拼入 prompt → LLM 基于上下文生成回答。RAG 能有效减少大模型幻觉，让回答有据可查。",
        },
        {
            "title": "向量检索原理",
            "content": "向量检索将文本通过 Embedding 模型（如 BGE-m3）编码为高维向量，通过计算向量之间的余弦相似度或内积来找到语义最相近的文档。FAISS 是常用的高性能向量检索库。",
        },
        {
            "title": "混合检索",
            "content": "混合检索 = 关键词检索（BM25）+ 向量检索 + RRF 排名融合。关键词检索擅长精确匹配（如服务编号号 ORD-1001），向量检索擅长语义匹配（如'没票了怎么办'），两者互补。",
        },
        {
            "title": "模型后端选择",
            "content": "可选择的后端包括：transformers 本地加载、Ollama 本地服务、DeepSeek 云端 API、OpenAI API、vLLM 生产推理。按场景选择：本地开发用 transformers/Ollama，生产用 DeepSeek/vLLM。",
        },
    ]
    for item in business:
        all_docs.append({
            "content": item["content"],
            "title": item["title"],
            "source": "业务系统知识",
            "type": "业务知识",
        })

    return all_docs


print("\n一、收集课程资料")
docs = load_course_docs()
print(f"  共收集 {len(docs)} 个文本块")

# 统计来源
from collections import Counter
type_counts = Counter(d["type"] for d in docs)
for t, c in type_counts.most_common():
    print(f"    {t}: {c} 条")

if len(docs) < 10:
    raise SystemExit("收集到的资料太少，请检查课程目录结构。")


# ========== 2. 加载 BGE-m3 并编码 ==========

print(f"\n二、加载 BGE-m3 模型")
print(f"  路径: {MODEL_DIR.relative_to(COURSE_ROOT)}")

start = time.perf_counter()
model = SentenceTransformer(str(MODEL_DIR), device="cpu")
load_ms = (time.perf_counter() - start) * 1000
print(f"  加载耗时: {load_ms:.0f} ms")

texts = [d["content"] for d in docs]
print(f"\n三、生成向量（共 {len(texts)} 条）")
start = time.perf_counter()
vectors = model.encode(texts, batch_size=2, normalize_embeddings=True, show_progress_bar=True)
vectors = np.asarray(vectors, dtype="float32")
encode_ms = (time.perf_counter() - start) * 1000

dim = vectors.shape[1]
print(f"  向量维度: {dim}")
print(f"  编码耗时: {encode_ms:.0f} ms")
print(f"  首条向量前 8 个数: {vectors[0][:8]}")


# ========== 3. 写入 FAISS 索引和元数据 ==========

print(f"\n四、保存向量库到 {KB_DIR.relative_to(COURSE_ROOT)}/")
KB_DIR.mkdir(parents=True, exist_ok=True)

# FAISS 索引（内积 = 余弦相似度，因为向量已归一化）
index = faiss.IndexFlatIP(dim)
index.add(vectors)
faiss.write_index(index, str(FAISS_PATH))
np.save(NPY_PATH, vectors)
print(f"  FAISS 索引: {FAISS_PATH.relative_to(COURSE_ROOT)} ({index.ntotal} 条)")
print(f"  向量矩阵:   {NPY_PATH.relative_to(COURSE_ROOT)} ({vectors.shape})")

# JSONL 元数据
with JSONL_PATH.open("w", encoding="utf-8") as f:
    for row_id, doc in enumerate(docs):
        meta = {
            "row_id": row_id,
            "title": doc["title"],
            "source": doc["source"],
            "type": doc["type"],
            "content": doc["content"],
        }
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
print(f"  元数据:     {JSONL_PATH.relative_to(COURSE_ROOT)} ({len(docs)} 条)")

# 索引说明
meta = {
    "version": "1.0",
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "embedding_model": "bge-m3",
    "model_path": str(MODEL_DIR.relative_to(COURSE_ROOT)),
    "vector_dim": int(dim),
    "total_chunks": len(docs),
    "similarity": "内积（向量已归一化，等价于余弦相似度）",
    "files": {
        "faiss_index": FAISS_PATH.name,
        "vector_matrix": NPY_PATH.name,
        "metadata": JSONL_PATH.name,
    },
}
META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  索引说明:   {META_PATH.relative_to(COURSE_ROOT)}")

print(f"\n五、构建完成")
print(f"  知识库包含 {len(docs)} 条记录，覆盖 {len(type_counts)} 类资料。")
print(f"  下一步运行: python code/rag_bot.py")
