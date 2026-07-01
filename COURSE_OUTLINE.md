# 课程大纲

## 第 1 讲：AI 基础与模型发展

- 传统机器学习与深度学习的发展脉络
- Token、Embedding、Attention 与 Transformer
- 原始文本生成、Chat 模板、流式输出
- 本地模型、Ollama 和在线 API 的区别

## 第 2 讲：大模型接口与业务指令

- Chat Completions 请求结构
- system / user / assistant 三类消息
- temperature、max tokens、stream 等常见参数
- JSON 输出、错误处理、边界控制

## 第 3 讲：RAG 知识库与检索策略

- 文档解析：Markdown、PDF、Word、HTML
- 文本清洗、切片、chunk size、overlap
- BM25 关键词检索、Embedding 向量检索
- 混合检索、RRF、Rerank 与检索评估

## 第 4 讲：工具调用与 Agent

- 工具名称、描述、参数 schema、返回值
- 模型直接选择工具与程序侧执行
- Router：规则过滤、向量召回、LLM rerank、schema 注入
- Agent harness：计划、执行、日志、权限、检查

## 第 5 讲：上下文工程与多轮对话

- 短期上下文、长期记忆、检索上下文、工具返回上下文
- Token 预算、历史裁剪、记忆压缩
- 多轮对话中 reasoning / tool calls 的传回规则

## 第 6 讲：OpenCode 辅助脚本学习与修改

- 用 OpenCode 阅读脚本、解释报错、修改参数
- 用固定任务练习 RAG、工具调用、Agent 代码
- 把课堂脚本改成自己的个人知识库或业务助手原型

## 第 7 讲：专题案例

- 极简 Coding Agent
- 多智能体游戏示例
- 辩论机器人与文章润色
- 个人知识库问答助手