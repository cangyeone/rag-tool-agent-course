# 11 专题：辩论与文章润色

这个专题用于演示一个最小辩论系统。它包含两个辩手 Agent 和一个裁判 Agent，并逐步加入 memory 与压缩记忆。

核心链路：

```text
辩题 -> 正方发言 -> 反方反驳 -> 多轮记忆 -> 压缩记忆 -> 裁判评分
```

扩展链路：

```text
题目/文章 -> 两个评审 Agent 找问题 -> 判断 Agent 筛新增问题 -> 修改 Agent 润色 -> 无新增问题后停止
```

代码放在：

```text
code/
```

## 课程安排

| 环节 | 建议时长 | 脚本 | 重点 |
| --- | --- | --- | --- |
| 最小 API 辩论 | 20 分钟 | `01_最小辩论_单轮.py` | chat、角色、正反方 |
| 双 Agent 互辩 | 40 分钟 | `02_双Agent辩论_多轮.py` | 多智能体、多轮对话 |
| 加入记忆 | 40 分钟 | `03_加入Memory_辩手记忆.py` | public memory、private memory |
| 压缩记忆 | 40 分钟 | `04_压缩Memory_保留要点.py` | memory summary、上下文裁剪 |
| 裁判评分 | 50 分钟 | `05_裁判Agent_评分判断.py` | judge agent、JSON 评分、胜负判断 |
| 文章润色循环 | 60 分钟 | `06_文章润色_辩论式评审循环.py` | 双评审、判断新增问题、自动修改、停止条件 |

## 能展示什么

- Role Playing：正方、反方、裁判分别有不同角色。
- Multi-Agent：两个辩手轮流发言，裁判最后判断。
- Memory：保存每轮发言、对手观点、自己的策略。
- Memory Compression：把长对话压缩成短摘要。
- Structured Output：裁判输出 JSON，程序读取 winner、score、reason。
- Reasoning：围绕论点、证据、反驳质量进行判断。
- Article Polishing：把辩论结构改造成文章评审与润色流程。
- Stop Condition：判断 Agent 认为没有新增问题时自动停止。

## 运行方式

进入课程根目录：

```bash
cd rag-tool-agent-course
```

运行：

```bash
python code/11_专题_辩论与文章润色/code/01_最小辩论_单轮.py
```

运行文章润色循环：

```bash
python code/11_专题_辩论与文章润色/code/06_文章润色_辩论式评审循环.py
```

脚本会读取系统环境变量 `DEEPSEEK_API_KEY`。公开仓库不提供任何真实 Key。

## 文章润色脚本的可改参数

只给题目，让脚本先生成初稿：

```bash
export ARTICLE_TITLE="RAG 在企业知识库中的价值"
python code/11_专题_辩论与文章润色/code/06_文章润色_辩论式评审循环.py
```

直接给文章：

```bash
export ARTICLE_TEXT="这里放需要润色的文章正文"
python code/11_专题_辩论与文章润色/code/06_文章润色_辩论式评审循环.py
```

控制最大循环轮数：

```bash
export ARTICLE_POLISH_MAX_ROUNDS=3
```