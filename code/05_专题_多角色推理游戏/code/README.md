# LLM 狼人杀代码目录

## 运行顺序

```text
01_调API_最小chat.py
02_单角色狼人_角色扮演_JSON.py
03_完整狼人杀_多智能体状态机.py
04_加记忆系统_玩家记忆.py
05_加工具调用_投票统计.py
06_加裁判Agent_状态裁决.py
07_多模型对战_不同策略.py
08_完整游戏直到胜负.py
09_加入女巫_完整游戏直到胜负.py
```

## 能展示什么

| 脚本 | 展示能力 |
| --- | --- |
| 01 | chat messages、system/user/assistant |
| 02 | role playing、JSON 输出 |
| 03 | multi-agent、game state、reasoning |
| 04 | memory、公开记忆、私有记忆 |
| 05 | tool call、工具执行、工具结果回传 |
| 06 | judge agent、状态裁决、胜负判断 |
| 07 | 多模型配置、不同策略对战 |
| 08 | 完整游戏循环、夜晚行动、白天发言、投票出局、最终胜负 |
| 09 | 女巫解药/毒药、夜晚多角色结算、完整游戏胜负 |

## API Key

本目录已经提供本地演示用 `.env`：

```text
.env
```

脚本读取顺序：

1. 先读取系统环境变量 `DEEPSEEK_API_KEY`。
2. 如果环境变量不存在，再读取同目录 `.env`。

如果要临时更换演示 Key，修改 `.env` 即可。