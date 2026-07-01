"""rag_engine —— RAG 问答引擎。

核心类：
    KnowledgeBaseRAG
        加载向量库，提供 search() 检索和 ask_stream() 流式问答。

使用示例：
    rag = KnowledgeBaseRAG(db_dir=Path("kb_data"), model_dir=Path("open_models/bge-m3"))
    rag.load()
    results = rag.search("什么是候补申请？", top_k=5)
    for piece in rag.ask_stream("什么是候补申请？"):
        print(piece, end="")
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Generator

import numpy as np
import requests

from .vectorstore import load_vectorstore, get_kb_stats

SYSTEM_PROMPT = (
    "你是个人知识库助手，回答简洁专业、条理清晰。"
    "请根据提供的参考资料回答问题。如果参考资料不足以回答，请诚实说明。"
    "涉及业务规则时注明以官方页面为准。"
)


class KnowledgeBaseRAG:
    """个人知识库 RAG 问答引擎。

    属性：
        db_dir:      向量库目录
        model_dir:    BGE-m3 本地模型路径
        api_key:      DeepSeek API Key
        base_url:     API Base URL
        llm_model:    使用的 LLM 模型名
        temperature:  LLM 采样温度
        max_tokens:   最大输出 token 数
        top_k:        检索返回的文档数量
        index:        FAISS 索引对象
        metadata:     JSONL 元数据列表
        model:        SentenceTransformer 编码模型
    """

    def __init__(
        self,
        db_dir: Path,
        model_dir: Path,
        api_key: str = "",
        base_url: str = "https://api.deepseek.com",
        llm_model: str = "deepseek-chat",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        top_k: int = 5,
    ):
        self.db_dir = db_dir
        self.model_dir = model_dir
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k

        self.index = None
        self.metadata: list[dict] = []
        self.model = None
        self._loaded = False

    def load(self) -> bool:
        """加载向量库和编码模型。

        Returns:
            bool: 是否加载成功。
        """
        store = load_vectorstore(self.db_dir, self.model_dir)
        if store is None:
            return False
        self.index = store["index"]
        self.metadata = store["metadata"]
        self.model = store["model"]
        self._loaded = True
        return True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """向量检索，返回最相似的 top_k 条文档。

        Args:
            query: 查询文本。
            top_k: 返回数量，默认为 self.top_k。

        Returns:
            检索结果列表，每条包含 title / source / content / score / page 等字段。
        """
        if not self._loaded:
            return []

        k = top_k if top_k is not None else self.top_k
        k = min(k, len(self.metadata))

        query_vec = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )
        query_vec = np.asarray(query_vec, dtype="float32")

        scores, row_ids = self.index.search(query_vec, k)

        results: list[dict] = []
        for score, row_id in zip(scores[0], row_ids[0]):
            if row_id < 0 or row_id >= len(self.metadata):
                continue
            row = dict(self.metadata[int(row_id)])
            row["score"] = float(score)
            results.append(row)
        return results

    def _build_prompt(self, question: str, retrieved: list[dict]) -> list[dict]:
        """将检索结果拼入 prompt 消息列表。"""
        context_parts: list[str] = []
        for i, doc in enumerate(retrieved, 1):
            context_parts.append(
                f"[{i}] {doc.get('title', '')} "
                f"(来源: {doc.get('source', '')}, 相关度: {doc.get('score', 0):.3f})\n"
                f"{doc.get('content', '')}"
            )
        context = "\n\n".join(context_parts)

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"参考资料：\n{context}\n\n"
                    f"问题：{question}\n\n"
                    "请基于以上参考资料回答。如果资料不足以回答，请诚实说明。"
                ),
            },
        ]

    def ask_stream(
        self,
        question: str,
        history: list[dict] | None = None,
        top_k: int | None = None,
    ) -> Generator[str, None, None]:
        """流式问答：检索 + LLM 流式生成。

        Args:
            question: 用户问题。
            history: 历史对话消息列表。
            top_k: 检索数量。

        Yields:
            str: 每次 yield 一个文本片段。
        """
        if not self._loaded:
            yield "[错误] 知识库未加载，请先构建向量数据库。"
            return

        if not self.api_key:
            yield "[错误] 未设置 DEEPSEEK_API_KEY。"
            return

        retrieved = self.search(question, top_k=top_k)
        prompt_msgs = self._build_prompt(question, retrieved)

        full_messages = (history or [])[-6:] + prompt_msgs

        chat_url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.llm_model,
            "messages": full_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": 0.9,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(
                chat_url, headers=headers, json=payload, timeout=300, stream=True
            )
            resp.raise_for_status()

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    if chunk.get("choices"):
                        delta = chunk["choices"][0].get("delta", {})
                        if delta.get("content"):
                            yield delta["content"]
                except json.JSONDecodeError:
                    pass

        except requests.Timeout:
            yield "\n[请求超时，请重试]"
        except requests.HTTPError as e:
            yield f"\n[API 错误: {e.response.status_code}]"
        except Exception as e:
            yield f"\n[异常: {type(e).__name__}: {e}]"

    def ask(self, question: str, history: list[dict] | None = None,
            top_k: int | None = None) -> str:
        """非流式问答：检索 + LLM 生成，返回完整答案字符串。

        Args:
            question: 用户问题。
            history: 历史对话消息列表。
            top_k: 检索数量。

        Returns:
            str: 完整答案。
        """
        result: list[str] = []
        for piece in self.ask_stream(question, history=history, top_k=top_k):
            result.append(piece)
        return "".join(result)

    def get_stats(self) -> dict:
        """获取知识库统计信息。"""
        return get_kb_stats(self.db_dir)