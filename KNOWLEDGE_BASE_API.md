# 个人知识库系统 (Personal Knowledge Base) — API 文档

> **版本**: 1.0.0  
> **Python**: 3.10+  
> **模型**: BGE-m3 (1024维, 本地部署)  
> **向量库**: FAISS IndexFlatIP  
> **元数据**: SQLite + JSONL  
> **Web 框架**: Flask  
> **LLM**: DeepSeek API (chat/completions)

---

## 目录

1. [系统架构](#1-系统架构)
2. [目录结构](#2-目录结构)
3. [模块 API](#3-模块-api)
   - [3.1 parser — 文档解析与切片](#31-parser--文档解析与切片)
   - [3.2 vectorstore — 向量数据库管理](#32-vectorstore--向量数据库管理)
   - [3.3 rag_engine — RAG 问答引擎](#33-rag_engine--rag-问答引擎)
   - [3.4 web_ui — Web 界面](#34-web_ui--web-界面)
4. [Web API 端点](#4-web-api-端点)
5. [数据流图](#5-数据流图)
6. [数据库设计 (SQLite)](#6-数据库设计-sqlite)
7. [配置说明](#7-配置说明)
8. [使用指南](#8-使用指南)

---

## 1. 系统架构

```
┌───────────────────────────────────────────────────────────┐
│                    knowledge/  (源文件)                     │
│                 *.pdf          *.md                        │
└───────────────────────┬───────────────────────────────────┘
                        │ ① 遍历 & 解析
                        ▼
┌───────────────────────────────────────────────────────────┐
│                  parser.py                                │
│  parse_pdf()  →  按页提取文本 + 保留页码/来源元数据         │
│  parse_markdown() → 按标题分段提取文本                     │
│  chunk_text()  →  段落边界切分 + 滑动窗口重叠              │
│  collect_and_chunk() → 统一入口，遍历目录 + 解析 + 切片    │
└───────────────────────┬───────────────────────────────────┘
                        │ ② 切片文本
                        ▼
┌───────────────────────────────────────────────────────────┐
│                  vectorstore.py                           │
│  build_vectorstore()                                     │
│    ├─ MD5 增量检测（对比 SQLite 中已入库文件的哈希）        │
│    ├─ BGE-m3 编码 (1024维, normalize_embeddings=True)      │
│    ├─ FAISS IndexFlatIP 写入                              │
│    ├─ SQLite 元数据写入 (documents / chunks / index_meta)  │
│    └─ JSONL + index_meta.json 备份                        │
│                                                           │
│  load_vectorstore()  → 加载已有向量库                      │
│  get_kb_stats()      → 统计信息查询                        │
└───────────────────────┬───────────────────────────────────┘
                        │ ③ 向量 + 元数据
                        ▼
┌───────────────────────────────────────────────────────────┐
│                  kb_data/  (输出)                          │
│  faiss.index   ← FAISS 向量索引                           │
│  vectors.npy   ← 向量矩阵 (NumPy)                         │
│  metadata.jsonl ← JSONL 元数据                            │
│  metadata.db   ← SQLite 数据库                            │
│  index_meta.json ← 索引说明                               │
└───────────────────────┬───────────────────────────────────┘
                        │ ④ 加载
                        ▼
┌───────────────────────────────────────────────────────────┐
│                  rag_engine.py                            │
│  KnowledgeBaseRAG                                        │
│    ├─ load()          → 加载 FAISS + 元数据 + BGE-m3      │
│    ├─ search(query)   → 向量检索 Top-K                    │
│    ├─ ask_stream()    → 流式 RAG 问答 (DeepSeek API)      │
│    └─ ask()           → 同步问答                          │
└───────────────────────┬───────────────────────────────────┘
                        │ ⑤ HTTP API
                        ▼
┌───────────────────────────────────────────────────────────┐
│                  web_ui.py (Flask)                        │
│  GET  /              → 聊天界面 (HTML/JS)                  │
│  POST /api/chat      → SSE 流式问答                        │
│  POST /api/update    → 触发向量库更新                      │
│  GET  /api/status    → 知识库状态                          │
└───────────────────────────────────────────────────────────┘
```

---

## 2. 目录结构

```
rag-tool-agent-course/
├── knowledge/                  # ← 用户放入 PDF/MD 文件的位置
│   ├── example.pdf
│   └── notes.md
├── kb_data/                    # ← 向量库输出（自动生成）
│   ├── faiss.index
│   ├── vectors.npy
│   ├── metadata.jsonl
│   ├── metadata.db
│   └── index_meta.json
├── personal_kb/                # ← 核心代码包
│   ├── __init__.py
│   ├── parser.py               # 文档解析 + 切片
│   ├── vectorstore.py          # FAISS + SQLite 管理
│   ├── rag_engine.py           # RAG 问答引擎
│   ├── web_ui.py               # Flask Web 服务
│   └── templates/
│       └── index.html          # 聊天界面 HTML
├── run_kb.py                   # ← 入口脚本
├── open_models/bge-m3/         # ← BGE-m3 模型文件
└── KNOWLEDGE_BASE_API.md       # ← 本文档
```

---

## 3. 模块 API

### 3.1 parser — 文档解析与切片

**文件**: `personal_kb/parser.py`

#### `parse_pdf(filepath: Path) -> list[dict]`

解析 PDF 文件，按页提取文本。

- **参数**:
  - `filepath` (`pathlib.Path`): PDF 文件路径。
- **返回**: 列表，每项为:
  ```python
  {
      "title": str,      # PDF 文件名（不含扩展名）
      "source": str,     # 文件相对路径
      "page": int,       # 页码（从 1 开始）
      "content": str,    # 清洗后的页面文本
      "chars": int,      # 字符数
  }
  ```
- **解析库回退链**: `pypdf` → `PyPDF2` → `PyMuPDF`，使用第一个可用的。
- **清洗规则**: 替换 `\u00a0` 为空格 → 合并连续空白 → 合并连续空行（3+ → 2）→ strip。

#### `parse_markdown(filepath: Path) -> list[dict]`

解析 Markdown 文件，按标题层级分段。

- **参数**:
  - `filepath` (`pathlib.Path`): Markdown 文件路径。
- **返回**: 列表，每项为:
  ```python
  {
      "title": str,      # 最近的一级/二级标题文本，或文件名
      "source": str,     # 文件相对路径
      "section": int,    # 段落序号
      "content": str,    # 段落文本
      "chars": int,      # 字符数
  }
  ```
- **分段规则**: 以 `#` / `##` / `###` 开头的行作为段落边界，代码块（\`\`\`）内不切分。

#### `chunk_text(text: str, chunk_size: int = 2000, overlap: int = 500) -> list[str]`

将长文本按段落边界切分，长段落用滑动窗口切分。

- **参数**:
  - `text` (`str`): 待切分的原始文本。
  - `chunk_size` (`int`): 每个块的目标最大字符数，默认 2000。
  - `overlap` (`int`): 相邻块之间的重叠字符数，默认 500。
- **返回**: `list[str]`，切分后的文本块列表。
- **算法**:
  1. 按 `\n\s*\n` 拆分为段落，过滤掉 <10 字符的短段落。
  2. 贪心拼接段落直到接近 chunk_size。
  3. 超长段落用滑动窗口切分，步长 = chunk_size - overlap。
  4. 在相邻块头部拼接上一块的尾部 overlap 字符。

#### `collect_documents(knowledge_dir: Path) -> list[dict]`

遍历知识库目录，收集并解析所有 `.pdf` 和 `.md` 文件。

- **参数**:
  - `knowledge_dir` (`pathlib.Path`): 知识库源文件目录。
- **返回**: 文档记录列表，每项额外包含 `file_ext` (".pdf"|".md") 和 `file_hash` (MD5) 字段。

#### `collect_and_chunk(knowledge_dir, chunk_size=2000, chunk_overlap=500) -> list[dict]`

`collect_documents()` + `chunk_text()` 的组合入口。

- **返回**: 切片记录列表，每项为:
  ```python
  {
      "title": str,
      "source": str,
      "file_ext": str,
      "file_hash": str,
      "page": int,
      "chunk_index": int,
      "content": str,
      "content_length": int,
  }
  ```

---

### 3.2 vectorstore — 向量数据库管理

**文件**: `personal_kb/vectorstore.py`

#### `build_vectorstore(knowledge_dir, db_dir, model_dir, chunk_size=2000, chunk_overlap=500, force_rebuild=False) -> dict`

构建/更新向量数据库（核心函数）。

- **参数**:
  | 参数 | 类型 | 默认值 | 说明 |
  |------|------|--------|------|
  | `knowledge_dir` | `Path` | — | 知识源文件目录 |
  | `db_dir` | `Path` | — | 向量库输出目录 |
  | `model_dir` | `Path` | — | BGE-m3 本地模型目录 |
  | `chunk_size` | `int` | 2000 | 切片最大字符数 |
  | `chunk_overlap` | `int` | 500 | 切片重叠字符数 |
  | `force_rebuild` | `bool` | False | 是否强制全量重建 |

- **返回**:
  ```python
  {
      "status": str,          # "ok" | "no_changes" | "no_files" | "error"
      "total_files": int,     # 扫描到的文件总数
      "new_files": int,       # 新增/修改的文件数
      "skipped_files": int,   # 跳过的未变更文件数
      "total_chunks": int,    # 总切片数
      "vector_dim": int,      # 向量维度 (1024)
      "build_time_ms": float, # 构建耗时 (ms)
  }
  ```

- **增量更新机制**:
  1. 扫描 `knowledge_dir` 下所有 `.pdf`/`.md` 文件。
  2. 对每个文件计算 MD5 哈希，与 SQLite `documents` 表中存储的哈希对比。
  3. 仅对新增或哈希变更的文件进行解析、切片、编码。
  4. 新向量追加到已有 FAISS 索引末尾。
  5. 全量 JSONL 覆盖写入（保证 row_id 连续性）。

- **SQLite 表操作**:
  - `documents` 表: 记录每个源文件的 source、hash、类型等。
  - `chunks` 表: 记录每个切片的元数据，关联 document_id。
  - `index_meta` 表: 存储版本号、维度等索引元信息。

#### `load_vectorstore(db_dir: Path, model_dir: Path) -> dict | None`

加载已有的向量库。

- **返回**: 包含 `index` (FAISS), `metadata` (list[dict]), `model` (SentenceTransformer), `total_chunks` (int) 的字典；若不存在返回 None。

#### `get_kb_stats(db_dir: Path) -> dict`

获取知识库统计信息。

- **返回**:
  ```python
  {
      "has_vectorstore": bool,
      "has_metadata": bool,
      "file_count": int,
      "chunk_count": int,
      "version": str | None,
  }
  ```

---

### 3.3 rag_engine — RAG 问答引擎

**文件**: `personal_kb/rag_engine.py`

#### `class KnowledgeBaseRAG`

核心 RAG 问答类。

**构造函数**:
```python
KnowledgeBaseRAG(
    db_dir: Path,                   # 向量库目录
    model_dir: Path,                # BGE-m3 模型目录
    api_key: str = "",              # DeepSeek API Key
    base_url: str = "https://api.deepseek.com",
    llm_model: str = "deepseek-chat",
    temperature: float = 0.3,
    max_tokens: int = 1024,
    top_k: int = 5,                 # 检索数量
)
```

#### `load() -> bool`

加载向量库和编码模型。返回是否成功。

#### `search(query: str, top_k: int | None = None) -> list[dict]`

向量检索。

- **参数**:
  - `query` (`str`): 查询文本。
  - `top_k` (`int | None`): 返回数量，默认使用构造函数中的值。
- **返回**: 检索结果列表，每项:
  ```python
  {
      "title": str,
      "source": str,
      "content": str,
      "score": float,    # 余弦相似度 (0~1)
      "page": int,
      "file_ext": str,
      ...
  }
  ```

#### `ask_stream(question, history=None, top_k=None) -> Generator[str, None, None]`

流式问答（生成器）。

- **参数**:
  - `question` (`str`): 用户问题。
  - `history` (`list[dict] | None`): 历史对话消息，格式为 `[{"role":"user","content":"..."}, ...]`。
  - `top_k` (`int | None`): 检索数量。
- **Yields**: `str` — 每次 yield 一个 LLM 输出的文本片段。

#### `ask(question, history=None, top_k=None) -> str`

非流式问答，返回完整答案字符串。参数同 `ask_stream`。

#### `get_stats() -> dict`

获取知识库统计信息，等同于 `vectorstore.get_kb_stats()`。

**检索 + 生成流程**:
```
用户问题
    │
    ▼
BGE-m3 编码 → FAISS.search() → Top-K 文档
    │
    ▼
构建 Prompt:
    system: "你是个人知识库助手..."
    user: "参考资料：\n[1] 标题 (来源, 相关度)\n内容...\n\n问题：..."
    │
    ▼
DeepSeek API (stream=True) → 流式输出
```

---

### 3.4 web_ui — Web 界面

**文件**: `personal_kb/web_ui.py`

#### `create_app() -> Flask`

创建 Flask 应用实例，注册所有路由。返回可运行的 Flask app 对象。

- **全局单例**: `rag` 对象懒加载，`_build_lock` 保证更新线程安全。
- **模板目录**: `personal_kb/templates/index.html`

---

## 4. Web API 端点

### `GET /`

聊天界面主页。返回 HTML 页面（含内嵌 CSS/JS）。

### `POST /api/chat`

流式问答接口（Server-Sent Events）。

- **Request Body** (JSON):
  ```json
  {
    "question": "什么是候补申请？",
    "history": [
      {"role": "user", "content": "你好"},
      {"role": "assistant", "content": "你好！有什么可以帮助你的？"}
    ]
  }
  ```
- **Response**: SSE 事件流
  ```
  data: {"type":"token","token":"候补申请"}
  data: {"type":"token","token":"下单"}
  ...
  data: {"type":"done","sources":[{...}]}
  ```
  - `type: "token"` — LLM 输出的文本片段。
  - `type: "done"` — 生成完成，附带 `sources` 数组（检索到的文档来源）。

- **Content-Type**: `text/event-stream`

### `POST /api/update`

触发向量数据库构建/更新（异步后台线程）。

- **Response** (JSON):
  ```json
  {"status": "started", "message": "数据库更新已启动。"}
  ```
- **注意**: 如果已有更新任务在运行，返回 `{"status": "running", "message": "..."}`。

### `GET /api/update/status`

查询当前更新任务状态。

- **Response** (JSON):
  ```json
  {
    "running": false,
    "result": {
      "status": "ok",
      "total_files": 3,
      "new_files": 1,
      "skipped_files": 2,
      "total_chunks": 48,
      "vector_dim": 1024,
      "build_time_ms": 12345.6
    }
  }
  ```

### `GET /api/status`

获取知识库整体状态。

- **Response** (JSON):
  ```json
  {
    "has_vectorstore": true,
    "has_metadata": true,
    "file_count": 3,
    "chunk_count": 48,
    "version": "1.0",
    "loaded": true,
    "model_available": true,
    "knowledge_dir_exists": true
  }
  ```

---

## 5. 数据流图

### 初始化流程

```
1. 用户将 PDF/MD 文件放入 knowledge/
2. 在 Web 界面点击「更新数据库」→ POST /api/update
3. build_vectorstore() 执行:
   a. collect_and_chunk(knowledge/)  → 遍历解析 + 切片
   b. MD5 检测变更（通过 SQLite documents 表）
   c. SentenceTransformer(BGE-m3).encode()  → 向量化
   d. faiss.IndexFlatIP.add()  → FAISS 索引
   e. SQLite 写入 documents/chunks/index_meta
   f. JSONL + index_meta.json 覆盖写入
```

### 问答流程

```
1. 用户在 Web 界面输入问题 → POST /api/chat
2. KnowledgeBaseRAG.load()  → 加载 FAISS + 元数据 + BGE-m3
3. search(question) → 向量检索 Top-K
4. ask_stream(question)  → 流式 LLM 生成
   ├─ _build_prompt() → 拼接参考资料
   └─ requests.post(deepseek API, stream=True) → 逐 token 返回
5. SSE 推送到浏览器 → 前端渲染
```

### 增量更新流程

```
build_vectorstore(force_rebuild=False)
  │
  ├─ 遍历 knowledge/ 所有 .pdf/.md
  ├─ 计算每个文件的 MD5
  ├─ 从 SQLite documents 表读取已入库 hash
  │
  ├─ hash 相同 → 跳过 (skipped_files++)
  │
  ├─ hash 不同或新文件 → 解析 + 切片 + 编码 (new_files++)
  │     ├─ SQLite: INSERT/UPDATE documents + DELETE旧chunks + INSERT新chunks
  │     └─ FAISS: 追加向量到现有索引
  │
  └─ 保存 FAISS + NPY + JSONL（全量覆盖）
```

---

## 6. 数据库设计 (SQLite)

**文件**: `kb_data/metadata.db`

### 表 `documents`

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INTEGER PK | 自增主键 |
| `source` | TEXT UNIQUE | 文件相对路径（如 `knowledge/notes.pdf`） |
| `file_hash` | TEXT | 文件 MD5 哈希，用于增量检测 |
| `file_ext` | TEXT | 文件扩展名（`.pdf` 或 `.md`） |
| `title` | TEXT | 文件名（不含扩展名） |
| `updated_at` | TEXT | 最后更新时间（ISO 格式） |

### 表 `chunks`

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INTEGER PK | 自增主键 |
| `document_id` | INTEGER FK | 关联 documents.id |
| `chunk_index` | INTEGER | 切片在文档内的序号（从 1 开始） |
| `content` | TEXT | 切片文本内容 |
| `content_length` | INTEGER | 文本字符数 |
| `page` | INTEGER | 页码（PDF）或 1（Markdown） |
| `vector_rowid` | INTEGER | FAISS 索引中的行号 |

### 表 `index_meta`

| 字段 | 类型 | 说明 |
|------|------|------|
| `key` | TEXT PK | 元信息键名 |
| `value` | TEXT | 元信息值 |

当前存储的键: `version`, `vector_dim`, `total_chunks`, `total_documents`

### ER 图

```
documents 1 ──── * chunks
  │                 │
  │ id              │ document_id (FK)
  │ source          │ chunk_index
  │ file_hash       │ content
  │ file_ext        │ vector_rowid → FAISS index row
  │ title           │
  │ updated_at      │
```

---

## 7. 配置说明

### 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `DEEPSEEK_API_KEY` | 是 | — | DeepSeek API 密钥 |
| `DEEPSEEK_BASE_URL` | 否 | `https://api.deepseek.com` | API 基础 URL |
| `DEEPSEEK_MODEL` | 否 | `deepseek-chat` | LLM 模型名 (`deepseek-chat` / `deepseek-reasoner`) |
| `QA_TEMPERATURE` | 否 | `0.3` | LLM 采样温度 (0~2) |
| `QA_MAX_TOKENS` | 否 | `1024` | 最大输出 token 数 |
| `RAG_TOP_K` | 否 | `5` | 检索返回的文档数量 |

### 配置方式

**方式一** — 环境变量:
```bash
export DEEPSEEK_API_KEY=your_api_key_herexxxxx
```

**方式二** — `.env` 文件（在 `code/.env`）:
```
DEEPSEEK_API_KEY=your_api_key_herexxxxx
DEEPSEEK_MODEL=deepseek-chat
QA_TEMPERATURE=0.3
```

### 依赖项

```
faiss-cpu>=1.7.4
sentence-transformers>=2.2.0
flask>=2.3.0
requests>=2.28.0
numpy>=1.24.0
pypdf>=3.0.0          # PDF 解析（三选一）
# 或 PyPDF2>=3.0.0
# 或 PyMuPDF>=1.23.0
```

### 模型要求

BGE-m3 模型必须预先放在 `open_models/bge-m3/` 目录下，该目录应包含:
- `config.json`
- `model.safetensors` 或 `pytorch_model.bin`
- `tokenizer.json`
- `sentence_bert_config.json`

---

## 8. 使用指南

### 快速开始

```bash
# 1. 进入项目目录
cd rag-tool-agent-course

# 2. 设置 API Key
export DEEPSEEK_API_KEY=your_api_key_herexxxxx

# 3. 放入文档到 knowledge/ 目录
cp your_document.pdf knowledge/
cp your_notes.md knowledge/

# 4. 构建向量数据库
python run_kb.py --build

# 5. 启动 Web 服务
python run_kb.py

# 6. 浏览器打开 http://localhost:7860
```

### 命令行模式

```bash
# 查看状态
python run_kb.py --status

# 单次问答
python run_kb.py --ask "什么是候补申请？"

# 仅更新数据库
python run_kb.py --build
```

### Web 界面使用

1. 访问 `http://localhost:7860`
2. 首次使用前，点击右上角 **🔄 更新数据库** 按钮
3. 等待更新完成（页面会显示 Toast 提示）
4. 在输入框输入问题，按 Enter 发送
5. 每个回答下方会显示参考来源文档和相似度评分

### 增量更新

将新的 PDF/MD 文件放入 `knowledge/` 目录后:
- **方式一**: 在 Web 界面点击「更新数据库」
- **方式二**: 运行 `python run_kb.py --build`

系统会自动检测变更，仅处理新增或修改的文件（通过 MD5 哈希对比），不会重复编码未变更的文件。

### 强制全量重建

如需完全重建数据库（例如模型更新后）:

```python
from personal_kb.vectorstore import build_vectorstore
from pathlib import Path

result = build_vectorstore(
    knowledge_dir=Path("knowledge"),
    db_dir=Path("kb_data"),
    model_dir=Path("open_models/bge-m3"),
    force_rebuild=True,
)
```

---

## 附录: 与参考代码的对应关系

| 本系统模块 | 参考来源 | 改进点 |
|-----------|---------|--------|
| `parser.parse_pdf()` | `lesson_01/02_PDF文档解析.py` | 提取为可复用函数 |
| `parser.chunk_text()` | `make_database.py:split_text()` | 保持相同算法 |
| `vectorstore.build_vectorstore()` | `lesson_02/03_向量库写入.py` + `make_database.py` | 新增 SQLite 元数据、增量更新 |
| `rag_engine.KnowledgeBaseRAG` | `qa_bot_rag.py` | 提取为类，支持流式/同步、Web API |
| `web_ui` | 全新 | Flask + SSE + 原生 HTML/CSS/JS |