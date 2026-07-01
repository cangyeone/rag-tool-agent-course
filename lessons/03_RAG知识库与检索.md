# RAG 知识库与检索
## 学习目标
- 文档解析、清洗、切片和元数据
- chunk size 与 overlap 的作用
- BM25 关键词检索与 Embedding 向量检索
- 混合检索、RRF、Rerank 和检索评估

## 对应代码
代码目录：`code/03_RAG知识库与检索`

建议从目录中的 `README.md` 开始读，再逐个运行脚本。

## OpenCode 练习
- 替换示例文档为自己的 Markdown
- 修改 chunk_size 和 overlap，对比命中片段
- 让 OpenCode 把关键词检索脚本扩展为混合检索

## 建议对 OpenCode 这样提问
```text
请先阅读本章节 README 和脚本旁边的 Markdown，帮我按课程顺序列出应该运行哪些脚本。
请解释第一个脚本的输入、处理过程和输出。
请帮我修改一个参数，并说明为什么这样改会影响结果。
请根据同名 Markdown 的任务说明重新实现这个脚本，要求代码简单、注释清楚。
```

## 学习检查
- 能说出本章关键概念。
- 能运行至少 2 个脚本。
- 能让 OpenCode 按 Markdown 说明改写一个脚本。