# 极简辩论机器代码目录

## 文件说明

```text
01_最小辩论_单轮.py
02_双Agent辩论_多轮.py
03_加入Memory_辩手记忆.py
04_压缩Memory_保留要点.py
05_裁判Agent_评分判断.py
06_文章润色_辩论式评审循环.py
```

## 推荐运行顺序

1. `01_最小辩论_单轮.py`：看懂正方和反方如何用不同 system message 发言。
2. `02_双Agent辩论_多轮.py`：看懂多 Agent 轮流发言。
3. `03_加入Memory_辩手记忆.py`：看懂 public memory 和 private memory。
4. `04_压缩Memory_保留要点.py`：看懂为什么长上下文需要压缩。
5. `05_裁判Agent_评分判断.py`：看懂裁判如何用 JSON 输出评分和胜负。
6. `06_文章润色_辩论式评审循环.py`：看懂如何把辩论结构改成文章评审、修改和停止循环。

## 06 脚本流程

```text
题目或文章
→ 结构评审 Agent 找逻辑问题
→ 表达评审 Agent 找语言问题
→ 判断 Agent 判断是否有新增问题
→ 修改 Agent 根据新增问题润色
→ 下一轮继续评审
→ 没有新增问题则停止
```

## 06 脚本可改参数

```text
ARTICLE_TITLE              文章题目；没有 ARTICLE_TEXT 时，会先生成初稿
ARTICLE_TEXT               需要润色的文章正文
ARTICLE_POLISH_MAX_ROUNDS  最大评审修改轮数，默认 4
DEEPSEEK_MODEL             默认 deepseek-v4-flash
```

## API Key

本目录已经提供本地演示用 `.env`。如果要换 Key，修改：

```text
.env
```