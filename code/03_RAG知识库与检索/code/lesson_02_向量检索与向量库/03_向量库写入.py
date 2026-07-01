"""03_向量库写入。

本脚本使用真实 Embedding 模型构建向量库：
1. 从本地 open_models/bge-m3 加载 BGE-m3
2. 把课堂资料和业务样例文本编码成 1024 维向量
3. 写入 FAISS 向量索引
4. 写入 JSONL 元数据
5. 写入 NPY 向量矩阵，方便课堂观察真实向量数值

运行方式：
    cd rag-tool-agent-course
    python code/03_RAG知识库与检索/code/lesson_02_向量检索与向量库/03_向量库写入.py
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
print("03_向量库写入 —— BGE-m3 + FAISS 真实向量库")
print("=" * 72)

if not MODEL_DIR.is_dir():
    raise SystemExit(f"未找到本地 BGE-m3 模型：{MODEL_DIR.relative_to(COURSE_ROOT)}")


def load_course_snippets() -> list[dict]:
    """准备一批可检索资料。

    这里混合两类文本：
    - 课程 README：让知识库能回答课程内容
    - 业务样例片段：让向量检索效果更容易观察
    """
    docs: list[dict] = []

    sample_files = [
        ("课程总览", "README.md", 900),
        ("大模型基础", "code/01_大模型基础/README.md", 900),
        ("模型接口与指令设计", "code/02_模型接口与指令设计/README.md", 900),
        ("RAG知识库与检索", "code/03_RAG知识库与检索/README.md", 900),
        ("工具调用路由与上下文", "code/04_工具调用路由与上下文/README.md", 900),
    ]

    for title, rel_path, max_chars in sample_files:
        file_path = COURSE_ROOT / rel_path
        if file_path.exists():
            text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            docs.append({
                "id": f"course_{len(docs) + 1:03d}",
                "title": title,
                "source": rel_path,
                "content": text[:max_chars],
                "type": "课程资料",
            })

    business_docs = [
        {
            "title": "候补申请规则",
            "content": "候补申请不能保证一定成功。候补申请兑现结果取决于退款、变更、新增席位和排队顺序。用户提交候补申请订单后，需要关注候补申请截止时间和兑现结果，最终以官方页面显示为准。",
        },
        {
            "title": "退款变更提醒",
            "content": "办理退款款和变更前，应先确认订单类型、服务开始时间、订单状态和手续费规则。不同时间点办理退款款可能产生不同费用，具体金额以订单页面和官方规则为准。",
        },
        {
            "title": "服务点名称与编码",
            "content": "服务点A、服务点B、杭州东等服务网点在系统中通常同时存在中文服务点名称、拼音、服务点编码和城市信息。检索服务点资料时，关键词和语义检索可以结合使用。",
        },
        {
            "title": "学生优惠票说明",
            "content": "学生优惠票通常需要符合学生资质、使用服务区间和优惠次数等条件。系统核验结果、优惠区间和创建订单时间都会影响能否购买学生优惠票。",
        },
        {
            "title": "GPR安全检查记录",
            "content": "地质雷达图像分析需要结合里程、病害位置、图像异常、人工复核记录和维修处置建议。多模态模型可以辅助解释图像，但最终结论需要专业人员确认。",
        },
        {
            "title": "客服质检摘要",
            "content": "客服质检通常关注是否准确理解问题、是否引用有效规则、是否避免过度承诺、是否给出可操作建议，以及是否提醒用户以官方页面和业务系统为准。",
        },
    ]

    for item in business_docs:
        docs.append({
            "id": f"biz_{len(docs) + 1:03d}",
            "title": item["title"],
            "source": "课堂业务样例",
            "content": item["content"],
            "type": "业务样例",
            "page": 1, 
        })

    return docs


docs = load_course_snippets()
if not docs:
    raise SystemExit("没有加载到任何资料，请检查课程目录。")

print("\n一、待入库资料")
for i, doc in enumerate(docs, 1):
    print(f"{i:02d}. {doc['title']} | {doc['type']} | 字数={len(doc['content'])}")

print("\n二、加载 BGE-m3 模型")
print("模型路径：", MODEL_DIR.relative_to(COURSE_ROOT))
# 文本->向量
start = time.perf_counter()
model = SentenceTransformer(str(MODEL_DIR), device="cpu")
load_ms = (time.perf_counter() - start) * 1000
print(f"加载耗时：{load_ms:.1f} ms")

texts = [doc["content"] for doc in docs]

print("\n三、生成真实文本向量")
start = time.perf_counter()
vectors = model.encode(
    texts,
    batch_size=4,
    normalize_embeddings=True,
    show_progress_bar=True,
)
vectors = np.asarray(vectors, dtype="float32")
encode_ms = (time.perf_counter() - start) * 1000

print(f"向量矩阵形状：{vectors.shape}")
print(f"向量 dtype：{vectors.dtype}")
print(f"编码耗时：{encode_ms:.1f} ms")
print("第 1 条向量前 12 个数：")
print(np.array2string(vectors[0][:12], precision=6, separator=", "))
print("第 1 条向量 L2 长度：", float(np.linalg.norm(vectors[0])))

print("\n四、写入 FAISS 索引")
VECTOR_DIR.mkdir(parents=True, exist_ok=True)
dim = vectors.shape[1]

# 因为向量已经 normalize_embeddings=True，内积就等价于余弦相似度。
index = faiss.IndexFlatIP(dim)
#index.add(vectors)
idxs =  [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]



text_info_database = {}
for idx, info in zip(idxs, docs):
    text_info_database[idx] = info


index.add_with_ids(vectors, idxs)

V, IDs = index.search([vq], k=10)
for sids in IDs:
    text = test_info_database[sids]
# 向量数据库保存
faiss.write_index(index, str(FAISS_PATH))
np.save(NPY_PATH, vectors)

print("FAISS 索引类型：IndexFlatIP")
print("索引向量数：", index.ntotal)
print("向量维度：", dim)
print("索引文件：", FAISS_PATH.relative_to(COURSE_ROOT))
print("向量矩阵：", NPY_PATH.relative_to(COURSE_ROOT))

print("\n五、写入元数据")
with JSONL_PATH.open("w", encoding="utf-8") as f:
    for row_id, doc in enumerate(docs):
        metadata = {
            "row_id": row_id,
            "id": doc["id"],
            "title": doc["title"],
            "source": doc["source"],
            "type": doc["type"],
            "content": doc["content"],
            "content_length": len(doc["content"]),
        }
        f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

meta = {
    "version": "1.0",
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "embedding_model": "bge-m3",
    "model_path": str(MODEL_DIR.relative_to(COURSE_ROOT)),
    "vector_dim": int(dim),
    "normalize_embeddings": True,
    "similarity": "inner_product，因向量已归一化，等价于 cosine",
    "total_documents": len(docs),
    "faiss_index": str(FAISS_PATH.relative_to(VECTOR_DIR)),
    "vector_matrix": str(NPY_PATH.relative_to(VECTOR_DIR)),
    "metadata": str(JSONL_PATH.relative_to(VECTOR_DIR)),
}
META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

print("元数据文件：", JSONL_PATH.relative_to(COURSE_ROOT))
print("索引说明：", META_PATH.relative_to(COURSE_ROOT))

print("\n六、课堂观察点")
print("1. 这里的向量不是手写数字，也不是字符哈希，而是 BGE-m3 对文本编码后的真实 embedding。")
print("2. BGE-m3 输出 1024 维向量，数值本身不容易人工解释，但相似文本会在向量空间中更接近。")
print("3. FAISS 只存向量索引，不存原文；原文、标题、来源要放在 JSONL 元数据中。")
print("4. 读取检索请运行 04_向量库读取检索.py。")
