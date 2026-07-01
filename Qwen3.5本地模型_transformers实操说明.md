# 本地模型 transformers 实操说明

## 主要讲什么

- 本说明用于解释本地模型调用的基本流程：加载 tokenizer、加载模型、准备输入、生成输出、解码文本。
- 公开课代码以简单可读为主，适合学员对照脚本逐步修改。

## 基本概念

- Tokenizer：把文本转成模型可处理的 token id。
- Model：加载本地权重并完成推理。
- Generate：根据输入继续生成 token。
- Decode：把生成的 token id 转回文本。
- Chat template：把多轮对话整理成模型熟悉的输入格式。
