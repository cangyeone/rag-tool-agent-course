# 多轮对话代码说明

## 课程大纲

1. 先看 DeepSeek 纯对话多轮：理解 API 无状态，历史消息要自己带回。
2. 再看 DeepSeek tool calls：理解工具调用会让消息结构多出 `tool_calls` 和 `role=tool`。
3. 接着看百炼 Qwen 在线模型：确认普通多轮和保留思考过程的区别。
4. 再看本地 Qwen：用 transformers 手工维护 `messages`。
5. 最后看 OpenAI SDK：同一个 SDK 可以调用 DeepSeek，只要换 `base_url`。

## 脚本顺序

| 文件 | 作用 |
| --- | --- |
| `01_DeepSeek多轮对话_纯对话模式.py` | DeepSeek 普通多轮对话，不把 `reasoning_content` 输入回去 |
| `02_DeepSeek多轮对话_tool_calls_thinking.py` | DeepSeek 工具调用模式，保留并传回 `reasoning_content` |
| `03_百炼Qwen多轮对话_thinking传回规则.py` | 百炼 Qwen 在线模型：默认不读历史思考；需要时用 `preserve_thinking=True` |
| `04_Qwen本地模型多轮对话.py` | 本地 Qwen3.5-0.8B：手搓 JSON tool calls |
| `05_OpenAI_SDK调用DeepSeek多轮对话.py` | 使用 OpenAI Python SDK 调 DeepSeek 多轮对话 |
| `06_百炼Qwen多轮对话_tool_calls_thinking.py` | 百炼 Qwen 在线模型：多轮对话中的 tool calls + thinking |
| `07_Qwen本地模型最原始多轮对话.py` | 本地 Qwen3.5-0.8B：最原始 messages 多轮对话 |

## 课堂观察点

- `messages` 是多轮对话的核心。
- `system` 通常放规则，`user` 放用户问题，`assistant` 放模型回答。
- 工具调用时，assistant 消息里会出现 `tool_calls`。
- 工具执行结果要用 `role="tool"` 放回消息列表。
- 本地 transformers 没有原生 `tool_calls` 字段，可以用 JSON 协议手搓工具调用。
- 百炼 Function Calling 总结工具结果时，不再传 `tool_choice`。
- 普通多轮对话通常只保留最终回答即可。
- 需要连续推理或工具调用时，才特别关注 `reasoning_content`。