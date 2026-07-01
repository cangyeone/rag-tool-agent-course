# 极简 Coding Agent 代码目录

## 文件说明

```text
01_DeepSeek代码接口最小示例.py
02_生成Plan_JSON.py
03_工具调用_读取文件.py
04_生成代码并写入文件.py
05_运行测试并自动修复.py
06_最小CodingAgent循环.py
07_FIM代码补全接口.py
08_ChatPrefix代码块补全.py
demo_project/
├── calculator.py
└── test_calculator.py
```

## 学习顺序

1. 先看 `01_DeepSeek代码接口最小示例.py`，理解代码类 API 的最小调用方式。
2. 再看 `02_生成Plan_JSON.py`，理解 plan。
3. 再看 `03_工具调用_读取文件.py`，理解 tool call。
4. 再看 `04_生成代码并写入文件.py`，理解模型如何产出完整代码。
5. 再看 `05_运行测试并自动修复.py`，理解测试反馈。
6. 再看 `06_最小CodingAgent循环.py`，理解最小 agent loop。
7. 补充看 `07_FIM代码补全接口.py`，理解专门的代码补全接口。
8. 补充看 `08_ChatPrefix代码块补全.py`，理解如何用代码块前缀约束输出。

## DeepSeek 接口说明

本目录统一使用新版模型名：

```text
deepseek-v4-flash
deepseek-v4-pro
```

主要有三种写法：

```text
Chat Completions
适合：代码解释、生成完整文件、修复 bug、工具调用、Coding Agent 循环。

FIM Completion
适合：已有代码前缀和后缀，让模型补中间缺失代码。

Chat Prefix Completion
适合：希望模型严格接着某个 assistant 前缀输出，例如只输出 Python 代码块。
```

代码写入文件时，不要把 Markdown 代码块标记写进去。  
课堂展示代码时，可以打印成：

````text
```python
这里是代码
```
````

也就是说：

```text
写文件：纯代码
给人看：可以加 ```python 代码块
```

## API Key

本目录已经提供本地演示用 `.env`。如果要换 Key，修改：

```text
.env
```

脚本读取顺序：

1. 先读取系统环境变量 `DEEPSEEK_API_KEY`。
2. 如果没有，再读取同目录 `.env`。
3. 多人共用一个 Key 时，可设置 `DEEPSEEK_USER_ID` 区分不同使用者。