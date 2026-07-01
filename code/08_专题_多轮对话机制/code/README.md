# 多轮对话机制代码

## 主要讲什么

- 这一组脚本讲多轮对话中消息如何保存、如何传回模型，以及普通对话、工具调用和本地模型对话的差异。

## 基本概念

- Messages：多轮对话的消息列表。
- Role：system、user、assistant、tool 等消息身份。
- Tool calls：模型请求调用工具的结构化信息。
- Reasoning content：部分模型返回的推理内容。
- 本地多轮对话：用 chat 模板把历史消息拼接后生成。
