# 10 专题：极简 Coding Agent

这个专题用于给学生演示“最小 coding agent”是怎么工作的。它不做复杂框架，只保留最关键的链路：

```text
任务 -> 计划 plan -> 读文件 -> 改代码 -> 运行测试 -> 根据结果继续修复
```

代码放在：

```text
code/
```

## 课程安排

| 环节 | 建议时长 | 脚本 | 重点 |
| --- | --- | --- | --- |
| 代码接口最小示例 | 20 分钟 | `00_DeepSeek代码接口最小示例.py` | Chat 生成代码、FIM 补全代码 |
| 认识 Coding Agent | 20 分钟 | `01_生成Plan_JSON.py` | 任务拆解、plan、JSON |
| 工具调用读文件 | 30 分钟 | `02_工具调用_读取文件.py` | tool call、list_files、read_file |
| 让模型改代码 | 40 分钟 | `03_生成代码并写入文件.py` | 读取上下文、生成完整文件、写入 |
| 运行测试再修复 | 60 分钟 | `04_运行测试并自动修复.py` | run tests、错误反馈、二次修复 |
| 最小 Agent 循环 | 90 分钟 | `05_最小CodingAgent循环.py` | plan、act、observe、memory |
| 专用代码补全接口 | 30 分钟 | `06_FIM代码补全接口.py` | prefix、suffix、补中间代码 |
| 代码块前缀补全 | 30 分钟 | `07_ChatPrefix代码块补全.py` | assistant prefix、Python 代码块前缀、stop |

## 能展示什么

- Plan：模型先把任务拆成步骤。
- Tool Call：模型选择读文件、列目录等工具。
- Memory：保存已经观察到的文件内容和测试结果。
- Agent Loop：每轮执行一个动作，再根据结果进入下一轮。
- Test Feedback：把报错交给模型，让模型修复。
- FIM Completion：用已有代码前后文补中间缺失代码。
- Chat Prefix：用代码块前缀让模型稳定输出代码片段。

## 接口选择

```text
写完整程序、修 bug、做 Agent：优先用 Chat Completions。
补一段缺失代码：可以用 FIM Completion。
只想让模型继续输出代码块：可以用 Chat Prefix Completion。
```

写入 `.py` 文件时保存纯代码；课堂展示时再用 Markdown 代码块标注语言。

## 运行方式

进入课程根目录：

```bash
cd rag-tool-agent-course
```

运行：

```bash
python code/06_专题_极简代码智能体/code/01_生成Plan_JSON.py
```

从最基础的代码接口开始：

```bash
python code/06_专题_极简代码智能体/code/00_DeepSeek代码接口最小示例.py
```

脚本会读取系统环境变量 `DEEPSEEK_API_KEY`。公开仓库不提供任何真实 Key。

## 课堂修改点

- 修改 `goal`，观察 plan 怎么变化。
- 修改 `demo_project/calculator.py` 里的 bug，观察测试结果。
- 修改工具列表，观察 agent 能力边界。
- 修改循环轮数，观察 agent 是否能完成任务。