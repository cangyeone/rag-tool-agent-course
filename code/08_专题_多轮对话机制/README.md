# 08_专题_多轮对话机制

本专题专门讲一件事：多轮对话时，历史消息到底应该怎么传。

大模型 API 通常是无状态的。它不会自动记住上一轮说过什么，程序需要把必要的历史消息重新放进 `messages`，再发送给模型。

## 目录

```text
08_专题_多轮对话机制/
├── 多轮对话_其他模型与国际模型做法说明.md
├── 多轮对话_缓存命中与计费说明.md
└── code/
    ├── 01_DeepSeek多轮对话_纯对话模式.py
    ├── 02_DeepSeek多轮对话_tool_calls_thinking.py
    ├── 03_Qwen本地模型多轮对话.py
    ├── 04_OpenAI_SDK调用DeepSeek多轮对话.py
    └── 05_Qwen本地模型最原始多轮对话.py
```

## 核心结论

| 场景 | 历史消息怎么传 | thinking / reasoning 怎么处理 |
| --- | --- | --- |
| DeepSeek 普通多轮对话 | 传回 `user` 和 `assistant.content` | 没有工具调用时，`reasoning_content` 不需要传回 |
| DeepSeek tool calls + thinking | 传回完整 assistant 消息和 tool 消息 | 发生 tool call 的那轮，`reasoning_content` 需要继续传回 |
| 本地 Qwen transformers | 自己维护 `messages` 列表 | 用 JSON 协议手搓 tool calls，再用 `apply_chat_template` 渲染输入 |
| 本地 Qwen 原始多轮 | 自己维护 `messages` 列表 | 不加工具，不加 JSON，只展示 chat template + generate |
| OpenAI SDK 调 DeepSeek | 写法与 OpenAI SDK 基本一致 | 关键是 `base_url="https://api.deepseek.com"` |

## 缓存与计费

多轮对话的费用不只来自用户刚输入的一句话。每轮请求里重新传入的历史消息、RAG 片段、工具返回结果，以及模型本轮生成的内容都会影响成本。

建议配合阅读：

```text
多轮对话_缓存命中与计费说明.md
```

这份文档重点解释：

- 缓存命中和未命中的区别。
- 输入 token、缓存 token、输出 token 分别是什么。
- 为什么长输出经常比长输入更贵。
- 为什么完整历史、过多检索片段、长回答会让费用差距变大。
- 如何通过历史摘要、Top K 控制、固定前缀来降低成本。

## 运行方式

从 `rag-tool-agent-course` 根目录运行：

```bash
python code/08_专题_多轮对话机制/code/01_DeepSeek多轮对话_纯对话模式.py
```

DeepSeek 示例需要设置环境变量：

```bash
export DEEPSEEK_API_KEY="YOUR_DEEPSEEK_API_KEY"
```

Windows PowerShell：

```powershell
$env:DEEPSEEK_API_KEY="your_api_key_here"
```

本地 Qwen 示例需要模型目录存在：

```text
rag-tool-agent-course/open_models/Qwen3.5-0.8B
```
