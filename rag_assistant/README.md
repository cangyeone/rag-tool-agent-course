# RAG 问答机器人

这个目录放了一个很小的本地 RAG 示例。它做的事情很直接：

1. 读取一批文档。
2. 把长文本切成小段。
3. 用 BGE-m3 算向量。
4. 用关键词和向量一起检索。
5. 把检索到的内容交给模型回答。

对应代码主要是这三个文件：

```text
rag_assistant/
├── build_index.py    # 读取资料，切片，建立索引
├── rag_core.py       # 检索和问答逻辑
└── app.py            # Streamlit 页面
```

## 先准备模型

默认读取这个位置的 BGE-m3：

```text
open_models/bge-m3
```

如果模型放在别处，可以设置环境变量：

```bash
export BGE_M3_MODEL_PATH="../open_models/bge-m3"
```

Windows PowerShell：

```powershell
$env:BGE_M3_MODEL_PATH="..\open_models\bge-m3"
```

## 设置 API Key

不要把 Key 写进代码。运行前在终端设置：

```bash
export DEEPSEEK_API_KEY="你的 DeepSeek API Key"
```

Windows PowerShell：

```powershell
$env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"
```

## 构建索引

在仓库根目录运行：

```bash
python rag_assistant/build_index.py
```

也可以指定自己的资料目录：

```bash
python rag_assistant/build_index.py --source-dir ./my_docs
```

两个常用参数：

- `chunk-size`：每个切片大概多长。
- `overlap`：相邻切片之间保留多少重复内容。

例如：

```bash
python rag_assistant/build_index.py --chunk-size 650 --overlap 120
```

## 启动页面

macOS / Linux：

```bash
chmod +x rag_assistant/start_macos_linux.sh
./rag_assistant/start_macos_linux.sh
```

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File .\rag_assistant\start_windows.ps1
```

浏览器打开：

```text
http://localhost:8501
```

## 可以问什么

- RAG 一般分成哪几步？
- 切片时为什么要设置 overlap？
- 向量检索和关键词检索有什么区别？
- 工具调用和 RAG 有什么区别？
- 我想换成自己的资料，应该改哪里？

## 可以改哪里

- 想换资料：改 `--source-dir`。
- 想看切片效果：改 `chunk-size` 和 `overlap`。
- 想换模型：改 `BGE_M3_MODEL_PATH`。
- 想看检索结果：打开 `rag_core.py` 里的检索部分，打印 top-k 文本。
- 想改页面：看 `app.py`。

## 几个基本概念

- 知识库：可以被检索的一批文档片段。
- 切片：把长文档拆成适合检索的小段。
- Embedding：把文本变成向量。
- 向量检索：找语义相近的内容。
- 关键词检索：找字面上命中的内容。
- 混合检索：把向量检索和关键词检索放在一起用。
- 上下文：模型回答前看到的参考资料。
