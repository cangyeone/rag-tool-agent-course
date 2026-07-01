"""06_BGE-m3真实向量库写入。

本脚本放在“混合检索与 RRF”这一节，用真实 Embedding 数据替换手写向量：
1. 加载本地 open_models/bge-m3
2. 将课程资料和业务样例编码成 1024 维真实向量
3. 写入 FAISS 索引
4. 写入 JSONL 元数据
5. 写入 NPY 向量矩阵，便于课堂观察向量数值

运行方式：
    cd rag-tool-agent-course
    python code/03_RAG知识库与检索策略/code/lesson_03_混合检索与RRF/06_BGE-m3真实向量库写入.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = COURSE_ROOT / "open_models" / "bge-m3"
VECTOR_DIR = SCRIPT_DIR / "真实BGE向量库"
FAISS_PATH = VECTOR_DIR / "bge_m3_示例业务系统.index"
NPY_PATH = VECTOR_DIR / "bge_m3_vectors.npy"
JSONL_PATH = VECTOR_DIR / "bge_m3_metadata.jsonl"
META_PATH = VECTOR_DIR / "bge_m3_index_meta.json"

print("=" * 72)
print("06_BGE-m3真实向量库写入 —— BGE-m3 + FAISS")
print("=" * 72)

if not MODEL_DIR.is_dir():
    raise SystemExit(f"未找到本地 BGE-m3 模型：{MODEL_DIR.relative_to(COURSE_ROOT)}")


def load_documents() -> list[dict]:
    docs: list[dict] = []

    course_files = [
        ("课程总览", "README.md", 900),
        ("AI基础与模型发展", "code/01_AI基础与模型发展/README.md", 900),
        ("大模型接口与业务指令", "code/02_大模型接口与业务指令/README.md", 900),
        ("RAG知识库与检索策略", "code/03_RAG知识库与检索策略/README.md", 900),
    ]

    for title, rel_path, max_chars in course_files:
        path = COURSE_ROOT / rel_path
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            docs.append({
                "id": f"course_{len(docs) + 1:03d}",
                "title": title,
                "source": rel_path,
                "type": "课程资料",
                "content": text[:max_chars],
            })

    business_docs = [
        ("候补申请规则", "候补申请不能保证一定成功。候补申请兑现结果取决于退款、变更、新增席位和排队顺序。用户提交候补申请订单后，需要关注候补申请截止时间和兑现结果，最终以官方页面显示为准。"),
        ("退款变更提醒", "办理退款款和变更前，应先确认订单类型、服务开始时间、订单状态和手续费规则。不同时间点办理退款款可能产生不同费用，具体金额以订单页面和官方规则为准。"),
        ("服务点名称与编码", "北京南、上海虹桥、杭州东等服务网点在系统中通常同时存在中文站名、拼音、服务点编码和城市信息。检索服务点资料时，关键词和语义检索可以结合使用。"),
        ("学生优惠票说明", "学生优惠票通常需要符合学生资质、使用服务区间和优惠次数等条件。系统核验结果、优惠区间和下单时间都会影响能否购买学生优惠票。"),
        ("GPR安全检查记录", "地质雷达图像分析需要结合里程、病害位置、图像异常、人工复核记录和维修处置建议。多模态模型可以辅助解释图像，但最终结论需要专业人员确认。"),
        ("客服质检摘要", "客服质检通常关注是否准确理解问题、是否引用有效规则、是否避免过度承诺、是否给出可操作建议，以及是否提醒用户以官方页面和业务系统为准。"),
    ]

    for title, content in business_docs:
        docs.append({
            "id": f"biz_{len(docs) + 1:03d}",
            "title": title,
            "source": "课堂业务样例",
            "type": "业务样例",
            "content": content,
        })

    return docs


docs = load_documents()
if not docs:
    raise SystemExit("没有加载到资料，请检查课程目录。")

print("\n一、待入库资料")
for i, doc in enumerate(docs, 1):
    print(f"{i:02d}. {doc['title']} | {doc['type']} | 字数={len(doc['content'])}")

print("\n二、加载 BGE-m3 模型")
print("模型路径：", MODEL_DIR.relative_to(COURSE_ROOT))
start = time.perf_counter()
model = SentenceTransformer(str(MODEL_DIR), device="cpu")
print(f"模型加载耗时：{(time.perf_counter() - start) * 1000:.1f} ms")

print("\n三、生成真实 Embedding")
texts = [doc["content"] for doc in docs]
start = time.perf_counter()
vectors = model.encode(texts, batch_size=4, normalize_embeddings=True, show_progress_bar=True)
vectors = np.asarray(vectors, dtype="float32")
print(f"向量矩阵形状：{vectors.shape}")
print(f"向量 dtype：{vectors.dtype}")
print(f"编码耗时：{(time.perf_counter() - start) * 1000:.1f} ms")
print("第 1 条向量前 12 个数：")
print(np.array2string(vectors[0][:12], precision=6, separator=", "))
print("第 1 条向量 L2 长度：", float(np.linalg.norm(vectors[0])))

print("\n四、写入 FAISS + 元数据")
VECTOR_DIR.mkdir(parents=True, exist_ok=True)
dim = vectors.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(vectors)
faiss.write_index(index, str(FAISS_PATH))
np.save(NPY_PATH, vectors)

with JSONL_PATH.open("w", encoding="utf-8") as f:
    for row_id, doc in enumerate(docs):
        item = {
            "row_id": row_id,
            "id": doc["id"],
            "title": doc["title"],
            "source": doc["source"],
            "type": doc["type"],
            "content": doc["content"],
            "content_length": len(doc["content"]),
        }
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

meta = {
    "version": "1.0",
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "embedding_model": "bge-m3",
    "model_path": str(MODEL_DIR.relative_to(COURSE_ROOT)),
    "vector_dim": int(dim),
    "normalize_embeddings": True,
    "similarity": "inner_product，向量已归一化，所以等价于 cosine",
    "total_documents": len(docs),
    "faiss_index": FAISS_PATH.name,
    "vector_matrix": NPY_PATH.name,
    "metadata": JSONL_PATH.name,
}
META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

print("FAISS 索引：", FAISS_PATH.relative_to(COURSE_ROOT))
print("向量矩阵：", NPY_PATH.relative_to(COURSE_ROOT))
print("元数据：", JSONL_PATH.relative_to(COURSE_ROOT))
print("索引说明：", META_PATH.relative_to(COURSE_ROOT))

print("\n课堂观察点")
print("1. 这里的向量来自 BGE-m3，不是手写 numpy 数组，也不是字符哈希。")
print("2. FAISS 负责向量检索；JSONL 保存标题、来源、正文等可追溯信息。")
print("3. 下一步运行 07_BGE-m3真实向量检索与RRF.py，把 BGE 检索和 BM25/RRF 放在一起比较。")
