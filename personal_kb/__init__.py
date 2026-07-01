"""personal_kb —— 个人知识库系统。

模块：
    parser       - PDF/Markdown 文档解析与切片
    vectorstore  - FAISS 向量库构建 + SQLite 元数据管理
    rag_engine   - RAG 问答引擎（检索 + LLM 生成）
    web_ui       - Flask Web 界面（对话 + 数据库更新）
"""

__version__ = "1.0.0"