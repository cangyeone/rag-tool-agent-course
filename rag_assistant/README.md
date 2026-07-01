# RAG 工具公开课 RAG 问答机器人

这是一个本地可运行的 RAG 问答机器人示例。它会先从 `rag-tool-agent-course` 课程资料里检索相关片段，再把检索结果交给 DeepSeek 生成回答。

整体链路：

```text
课程资料 -> 文档读取 -> 文本切片 -> BGE-m3 向量化 -> 向量+关键词混合检索 -> DeepSeek 回答
```

课堂中还有一个 可视化工具 版助教入口，适合参与者直接使用：

```text
http://localhost/chatbot/8vwiuzRC3UI1EUhI
```

本目录里的本地 RAG 示例更适合讲代码原理；可视化工具 版课堂助教更适合上课时查资料、问步骤、排查实操问题。

## 一、目录说明

```text
RAG_问答机器人/
├── app.py                    # Streamlit Web 页面
├── build_index.py            # 构建本地知识索引
├── rag_core.py               # RAG 核心逻辑
├── start_macos_linux.sh      # macOS / Linux 一键启动
├── start_windows.ps1         # Windows 一键启动
├── .env.example              # 环境变量示例，不要写真实密钥
└── storage/                  # 自动生成，保存 chunks 和 embeddings
```

## 二、准备 BGE-m3 模型

建议把 BGE-m3 模型放到课程目录下的相对路径：

```text
rag-tool-agent-course/open_models/bge-m3
```

启动脚本会优先读取上面的课程统一模型目录。如果你只单独拷贝 `RAG_问答机器人/` 文件夹，也可以把模型放到：

```text
RAG_问答机器人/models/bge-m3
```

需要手动指定时，可以设置环境变量：

macOS / Linux：

```bash
export BGE_M3_MODEL_PATH="../open_models/bge-m3"
```

Windows PowerShell：

```powershell
$env:BGE_M3_MODEL_PATH="..\open_models\bge-m3"
```

## 三、设置 DeepSeek API Key

不要把 API Key 写进代码。运行前在终端设置：

macOS / Linux：

```bash
export DEEPSEEK_API_KEY="你的 DeepSeek API Key"
```

Windows PowerShell：

```powershell
$env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"
```

## 四、首次构建索引

进入目录：

```bash
cd rag-tool-agent-course/RAG_问答机器人
```

构建索引：

```bash
python build_index.py
```

默认会读取整个 `rag-tool-agent-course` 目录下的 `.md`、`.txt`、`.docx`、`.md` 文件，并跳过离线镜像包。

常用参数：

```bash
python build_index.py --chunk-size 650 --overlap 120
```

- `chunk-size`：每个文本片段的大致长度。
- `overlap`：相邻片段保留多少重叠文本，用来减少上下文被切断的问题。

## 五、启动问答机器人

macOS / Linux：

```bash
chmod +x start_macos_linux.sh
./start_macos_linux.sh
```

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File .\start_windows.ps1
```

打开：

```text
http://localhost:8501
```

## 六、可以怎么提问

示例问题：

- RAG 问答机器人一般由哪些步骤组成？
- 切片时为什么要设置 overlap？
- 向量检索和关键词检索有什么区别？
- 可视化工具 如何接入 DeepSeek？
- 工具调用和 RAG 的区别是什么？

## 七、如何换成自己的资料

方法一：把新资料放进 `rag-tool-agent-course`，重新构建索引：

```bash
python build_index.py
```

方法二：指定资料目录：

```bash
python build_index.py --source-dir ./my_docs
```

## 八、课堂讲解建议

可以按这条线讲：

1. 先看 `build_index.py`：资料如何进入知识库。
2. 再看 `rag_core.py` 里的 `split_text`：为什么需要切片和 overlap。
3. 再看 `BGEEmbedder`：Embedding 把文本变成可比较的向量。
4. 再看 `search`：向量检索负责语义相似，关键词检索负责精确命中。
5. 最后看 `ask_deepseek`：把检索结果放进输入上下文，让模型基于资料回答。

这就是一个最小但完整的 RAG 闭环。

## 九、和 可视化工具 课堂助教的关系

建议这样安排：

- 先用本地 RAG 示例说明“资料读取、切片、向量化、检索、生成”的代码链路。
- 再打开 可视化工具 课堂助教，可看到同一批课程资料可以被放进可视化工作流。
- 最后可提问 可视化工具 助教：“这个本地 RAG 示例和 可视化工具 知识库节点有什么对应关系？”

这样参与者既能理解代码细节，也能理解可视化工具里的节点不是凭空出现的，而是对同一套 RAG 流程的封装。