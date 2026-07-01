"""08 真实向量库 + 关键词检索：从几百个工具中召回候选工具。

前面的脚本用轻量词袋相似度模拟“向量召回”。
这个脚本改成真实版本：

1. 读取 tool_catalog_common.py 里的几百个工具。
2. 用本地 open_models/bge-m3 生成工具 description embedding。
3. 把向量写入 FAISS 向量库。
4. 用 BM25 做关键词检索。
5. 用 BGE + FAISS 做向量检索。
6. 用 RRF 融合关键词结果和向量结果。
7. 对融合后的少量工具注入完整 schema。

运行方式：
    cd rag-tool-agent-course
    python code/05_工具调用与Router/code/lesson_02_模型路由与参数抽取/08_真实向量库与关键词混合召回.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from tool_catalog_common import inject_full_schema, rule_filter, tokenize, tool_registry

print("08 真实向量库 + 关键词检索：从几百个工具中召回候选工具")
print("=" * 88)


def find_course_root() -> Path:
    """找到 rag-tool-agent-course 根目录，避免硬编码绝对路径。"""
    candidates = [Path.cwd().resolve(), *Path(__file__).resolve().parents]
    for path in candidates:
        if (path / "code").is_dir() and (path / "open_models").is_dir():
            return path
    raise SystemExit("未找到 rag-tool-agent-course 根目录。请从 rag-tool-agent-course 目录运行，或确认 open_models 存在。")


COURSE_ROOT = find_course_root()
MODEL_DIR = COURSE_ROOT / "open_models" / "bge-m3"
if not MODEL_DIR.is_dir():
    raise SystemExit("未找到 open_models/bge-m3。请确认 BGE-m3 模型已经放在课程根目录的 open_models/bge-m3。")

CURRENT_DIR = Path(__file__).resolve().parent
VECTOR_DB_DIR = CURRENT_DIR / "工具检索向量库"
VECTOR_DB_DIR.mkdir(exist_ok=True)

INDEX_PATH = VECTOR_DB_DIR / "tool_catalog_bge_m3.faiss"
METADATA_PATH = VECTOR_DB_DIR / "tool_catalog_metadata.jsonl"
VECTOR_META_PATH = VECTOR_DB_DIR / "tool_catalog_index_meta.json"


def tool_text(tool: dict) -> str:
    """用于检索的工具文本。

    这里不放完整 schema，只放轻量 tool card。
    原因：召回阶段关注“这个工具大概做什么”，完整参数定义留到最后注入。
    """
    return "\n".join([
        f"工具名称：{tool['name']}",
        f"工具分组：{tool['group']}",
        f"工具描述：{tool['description']}",
        f"轻量参数：{json.dumps(tool['short_schema'], ensure_ascii=False)}",
    ])


tool_docs = [tool_text(tool) for tool in tool_registry]
embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """全脚本只加载一次 BGE-m3，避免每个查询都重复加载模型。"""
    global embedding_model
    if embedding_model is None:
        print("\n加载 BGE-m3 embedding 模型：", MODEL_DIR)
        embedding_model = SentenceTransformer(str(MODEL_DIR), local_files_only=True)
    return embedding_model


def build_or_load_vector_db() -> tuple[faiss.Index, list[dict]]:
    """构建或读取 FAISS 向量库。"""
    if INDEX_PATH.exists() and METADATA_PATH.exists() and VECTOR_META_PATH.exists():
        print("\n读取已有 FAISS 向量库：")
        print(" ", INDEX_PATH)
        index = faiss.read_index(str(INDEX_PATH))
        metadata = []
        with METADATA_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                metadata.append(json.loads(line))
        return index, metadata

    print("\n未发现现成向量库，开始构建。")
    print("BGE-m3 模型目录：", MODEL_DIR)
    print("工具数量：", len(tool_docs))

    model = get_embedding_model()
    vectors = model.encode(
        tool_docs,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    metadata = []
    for i, tool in enumerate(tool_registry):
        metadata.append({
            "row_id": i,
            "name": tool["name"],
            "group": tool["group"],
            "description": tool["description"],
            "short_schema": tool["short_schema"],
        })

    faiss.write_index(index, str(INDEX_PATH))
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        for item in metadata:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    VECTOR_META_PATH.write_text(
        json.dumps({
            "model": "bge-m3",
            "model_dir": "open_models/bge-m3",
            "vector_count": int(vectors.shape[0]),
            "dimension": int(vectors.shape[1]),
            "index_type": "faiss.IndexFlatIP",
            "normalized": True,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("向量库已写入：", INDEX_PATH)
    return index, metadata


def bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """一个极简 BM25。

    课堂重点是解释关键词检索逻辑，所以直接写在脚本里。
    真实系统可以替换成 Elasticsearch、OpenSearch、Lucene、tantivy 等。
    """
    tokenized_docs = [tokenize(doc) for doc in docs]
    query_tokens = tokenize(query)

    doc_count = len(tokenized_docs)
    avgdl = sum(len(doc) for doc in tokenized_docs) / max(doc_count, 1)

    df = {}
    for doc in tokenized_docs:
        for token in set(doc):
            df[token] = df.get(token, 0) + 1

    scores = []
    for doc in tokenized_docs:
        length = len(doc)
        tf = {}
        for token in doc:
            tf[token] = tf.get(token, 0) + 1

        score = 0.0
        for token in query_tokens:
            if token not in tf:
                continue
            idf = math.log(1 + (doc_count - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5))
            numerator = tf[token] * (k1 + 1)
            denominator = tf[token] + k1 * (1 - b + b * length / avgdl)
            score += idf * numerator / denominator
        scores.append(score)
    return scores


def bm25_search(query: str, allowed_row_ids: set[int] | None = None, top_k: int = 8) -> list[dict]:
    scores = bm25_scores(query, tool_docs)
    row_ids = range(len(scores)) if allowed_row_ids is None else allowed_row_ids
    order = sorted(row_ids, key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {
            "row_id": i,
            "name": tool_registry[i]["name"],
            "group": tool_registry[i]["group"],
            "score": scores[i],
            "source": "bm25",
        }
        for i in order
        if scores[i] > 0
    ]


def vector_search(query: str, index: faiss.Index, metadata: list[dict], allowed_row_ids: set[int] | None = None, top_k: int = 8) -> list[dict]:
    model = get_embedding_model()
    query_vector = model.encode([query], normalize_embeddings=True).astype("float32")
    # FAISS IndexFlatIP 本身不带元数据过滤。
    # 课堂里用 overfetch + Python 过滤模拟“带工具组过滤的向量召回”。
    overfetch = min(len(metadata), max(top_k * 8, top_k))
    scores, ids = index.search(query_vector, overfetch)

    results = []
    for score, row_id in zip(scores[0], ids[0]):
        if row_id < 0:
            continue
        if allowed_row_ids is not None and int(row_id) not in allowed_row_ids:
            continue
        item = metadata[int(row_id)]
        results.append({
            "row_id": int(row_id),
            "name": item["name"],
            "group": item["group"],
            "score": float(score),
            "source": "vector",
        })
        if len(results) >= top_k:
            break
    return results


def rrf_fusion(result_lists: list[list[dict]], k: int = 60, top_n: int = 8) -> list[dict]:
    """Reciprocal Rank Fusion。

    只看排名，不直接混合 BM25 分数和向量分数。
    这样 BM25 和向量检索分数尺度不同也没关系。
    """
    fused = {}
    traces = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list, start=1):
            row_id = item["row_id"]
            fused[row_id] = fused.get(row_id, 0.0) + 1.0 / (k + rank)
            traces.setdefault(row_id, []).append({
                "source": item["source"],
                "rank": rank,
                "raw_score": round(item["score"], 4),
            })

    ordered = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)[:top_n]
    results = []
    for row_id, score in ordered:
        tool = tool_registry[row_id]
        results.append({
            "row_id": row_id,
            "name": tool["name"],
            "group": tool["group"],
            "description": tool["description"],
            "rrf_score": round(score, 6),
            "trace": traces[row_id],
            "tool": tool,
        })
    return results


def print_results(title: str, results: list[dict]) -> None:
    print("\n" + title)
    print("-" * 88)
    for rank, item in enumerate(results, start=1):
        print(f"{rank:02d}. {item['name']:<28} group={item['group']:<12} score={item['score']:.4f}")


def run_query(query: str) -> None:
    print("\n" + "=" * 88)
    print("用户问题：", query)
    print("=" * 88)

    index, metadata = build_or_load_vector_db()

    candidates, matched_groups, rule_reasons = rule_filter(query)
    allowed_row_ids = {tool_registry.index(tool) for tool in candidates}

    print("\n零、规则过滤")
    print("命中工具组：", matched_groups if matched_groups else "未命中，使用全量工具")
    print("规则理由：")
    for reason in rule_reasons:
        print("  -", reason)
    print(f"规则候选工具数：{len(candidates)} / {len(tool_registry)}")

    bm25_results = bm25_search(query, allowed_row_ids=allowed_row_ids, top_k=8)
    vector_results = vector_search(query, index=index, metadata=metadata, allowed_row_ids=allowed_row_ids, top_k=8)
    fused_results = rrf_fusion([bm25_results, vector_results], top_n=6)

    print_results("一、BM25 关键词检索结果", bm25_results)
    print_results("二、BGE-m3 + FAISS 向量检索结果", vector_results)

    print("\n三、RRF 融合结果")
    print("-" * 88)
    for rank, item in enumerate(fused_results, start=1):
        print(f"{rank:02d}. {item['name']:<28} group={item['group']:<12} rrf={item['rrf_score']}")
        print("    来源排名：", item["trace"])
        print("    描述：", item["description"])

    injected = inject_full_schema(fused_results[:5])
    print("\n四、最终 schema 注入")
    print(f"只注入前 {len(injected)} 个工具的完整 schema：")
    print([schema["function"]["name"] for schema in injected])
    print("\n第一个注入 schema 示例：")
    print(json.dumps(injected[0], ensure_ascii=False, indent=2))


demo_queries = [
    "帮我找一下张三上周发来的邮件，看看附件里有没有培训课程材料。",
    "在代码仓库里搜索 DeepSeek tool_calls 相关代码，并看看哪里解析 arguments。",
    "帮我查一下知识库里关于候补申请的规则。",
]

for demo_query in demo_queries:
    run_query(demo_query)

print("\n课堂结论")
print("1. BM25 适合精确词：邮件、Gmail、tool_calls、arguments、知识库。")
print("2. 向量检索适合语义相近：找资料、查文档、生成课程材料、定位代码。")
print("3. 几百个工具时，可以先用 BM25 + 向量召回，再用 RRF 融合。")
print("4. 最终仍然只给少量工具的完整 schema，而不是把全部工具塞给模型。")