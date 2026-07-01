# RAG 加工具调用

本节用两种方式讲工具调用：

1. 先看示意性流程，理解工具调用不是模型自己执行，而是程序执行。
2. 再看 DeepSeek 真实 API，观察模型如何通过 `tools` 参数选择工具。

## Demo 1：工具调用的基本流程

脚本：`02_工具调用的基本流程.py`

特点：

- 不调用大模型
- 代码短，适合先讲流程
- 展示“看问题 → 选工具 → 执行工具 → 整理回答”

运行：

```bash
python 02_工具调用的基本流程.py
```

## Demo 0：DeepSeek 直接选择工具

脚本：`01_DeepSeek直接选择工具.py`

特点：

- 使用真实 DeepSeek API
- 只演示模型返回 `tool_calls`
- 不执行工具，便于单独观察“模型能不能自己选工具”

运行：

```bash
export DEEPSEEK_API_KEY=your_api_key_here
python 01_DeepSeek直接选择工具.py
```

## Demo 2：DeepSeek 真实工具调用

脚本：`03_DeepSeek真实工具调用.py`

特点：

- 使用 DeepSeek Chat Completions 的 `tools` 参数
- 第一次请求：模型选择工具并给出参数
- Python 本地执行工具
- 第二次请求：模型根据工具结果生成最终回答

运行：

```bash
export DEEPSEEK_API_KEY=your_api_key_here
python 03_DeepSeek真实工具调用.py
```

Windows PowerShell：

```powershell
$env:DEEPSEEK_API_KEY="your_api_key_here"
python 03_DeepSeek真实工具调用.py
```

## 课堂讲解重点

- 工具调用不是模型真的查系统。
- 模型负责判断“要不要调用工具、调用哪个工具、参数怎么填”。
- 程序负责真正执行工具。
- 工具结果必须再传回模型，模型才能基于结果组织最终回答。
