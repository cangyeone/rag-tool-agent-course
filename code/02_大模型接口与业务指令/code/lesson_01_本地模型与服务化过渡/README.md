# 本地模型与服务化过渡


## 对应课程大纲

| 时间 | 主题 | 课程要点 | 对应代码 |
| --- | --- | --- | --- |
| 10:20-10:45 | 大模型基础：从 Transformer 到大语言模型 | token、embedding、位置编码、attention、训练范式、能力边界 | tokenizer 观察、next token 循环、Qwen 本地加载 |
| 10:45-11:15 | 大模型使用方式：从原始生成到对话 API | completion、messages、role、system、user、assistant、temperature、max tokens、stream | Qwen chat 模板、HTTP 服务草图、DeepSeek 最小请求体 |
| 11:15-11:40 | 业务可用的任务表达方式 | 任务边界、输入材料、输出格式、结构化 JSON、错误兜底 | 系统消息控制边界、结构化 JSON 输出、多案例批量测试 |

本目录代码用于配合上午下半场课程材料，重点展示本地模型调用、在线模型 API、流式输出、结构化输出和异常处理。

所属半天：6月22日上午_大模型基础与DeepSeek接入
课次：第 02 节

## 本节课目标

本节课按 9 个 demo 展开。每个 demo 都能单独运行，适合先讲 3-5 分钟，再可改一个参数观察输出。

## 演示脚本

### Demo 1: tokenizer 观察

脚本：`01_tokenizer_观察.py`

演示重点：用轻量规则模拟 tokenizer，把文本变成可计算的 id 序列。

运行方式：

```bash
python 01_tokenizer_观察.py
```

### Demo 2: 最小 next-token 循环

脚本：`02_最小_next_token_循环.py`

演示重点：不用黑盒接口，手写一个可解释的下一个 token 选择循环。

运行方式：

```bash
python 02_最小_next_token_循环.py
```

### Demo 3: Qwen 本地模型加载检查

脚本：`03_Qwen_本地模型加载检查.py`

演示重点：检查 Qwen3.5-0.8B 是否已经下载到课程相对路径，并读取 tokenizer 配置。

运行方式：

```bash
python 03_Qwen_本地模型加载检查.py
```

### Demo 4: Qwen 原始逐步输出

脚本：`04_Qwen_原始逐步输出.py`

演示重点：不用 chat template，直接给一段普通输入文本，观察本地模型如何逐步生成文本。

运行方式：

```bash
python 04_Qwen_原始逐步输出.py
```

### Demo 5: Qwen chat模板与API

脚本：`05_Qwen_chat模板与API.py`

演示重点：把 system/user 消息转换成模型能读的 chat 模板，再对比 Chat API 的请求格式。

运行方式：

```bash
python 05_Qwen_chat模板与API.py
```

### Demo 6: Ollama 调用兜底

脚本：`06_Ollama_调用兜底.py`

演示重点：真实尝试调用本机 Ollama；不可用时给出排查提示。

运行方式：

```bash
python 06_Ollama_调用兜底.py
```

### Demo 7: HTTP 服务接口草图

脚本：`07_HTTP_服务接口草图.py`

演示重点：说明命令行模型能力如何变成 /generate 接口。

运行方式：

```bash
python 07_HTTP_服务接口草图.py
```

### Demo 8: 模型后端切换

脚本：`08_模型后端切换.py`

演示重点：在 DeepSeek、Ollama、本地模型之间做清晰的后端选择。

运行方式：

```bash
python 08_模型后端切换.py
```

### Demo 9: GPR 与多模态安全样例

脚本：`09_GPR_与多模态安全样例.py`

演示重点：用模拟 GPR 异常、现场记录和规则依据，演示多模态行业安全辅助链路。

运行方式：

```bash
python 09_GPR_与多模态安全样例.py
```

## 推荐节奏

- Demo 1：建立本节课概念入口。
- Demo 2-3：展开核心代码，可改输入或参数。
- Demo 4：把代码能力接到真实应用视角，例如服务、检索、工具或工作流。
- 最后一个 Demo：做小复盘，检查输出是否符合业务边界。

## 统一修改点

- 改用户问题，观察检索、路由或质检输出是否变化。
- 改规则资料，观察命中片段和回答依据是否变化。
- 改工具参数，观察结构化结果和最终回答是否变化。
- 把终端输出发给课堂助教，让它解释这个 demo 在知识地图中的位置。

## 本节课产出

- 能说清本节课 9 个 demo 的先后关系。
- 至少完成 1 次参数修改，并解释输出为什么变化。
- 记录 1 个可以带到下一节课的问题。

## 下一节衔接

下一节进入：DeepSeek 调用与回答边界。
本节课最后建议留 3 分钟，可把当前 demo 的输出和下一节的输入对应起来。