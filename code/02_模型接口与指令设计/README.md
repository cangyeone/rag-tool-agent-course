# 02 课程章节：模型接口与指令设计

## 对应课程大纲

| 时间 | 主题 | 课程要点 | 对应代码 |
| --- | --- | --- | --- |
| 10:20-10:45 | 从本地到云端：服务化过渡 | 本地模型局限、HTTP API 范式、环境变量与密钥管理、多后端对比 | 本地模型回顾、服务化过渡、HTTP 协议调用、多模型后端对比 |
| 10:45-11:15 | DeepSeek API 调用实战 | 请求体结构、系统消息控制边界、流式输出、结构化 JSON、错误兜底 | DeepSeek 最小请求体、系统消息边界、结构化输出、多案例测试 |
| 11:15-11:40 | 业务可用的任务表达方式 | 任务边界、输入材料、输出格式、结构化 JSON、AI Native 应用画布 | 错误与兜底、AI Native 应用画布 |

本目录代码用于配合上午下半场课程材料，重点展示从本地模型到云端 API 的过渡，以及 DeepSeek API 的完整调用链路。

对应课程材料：

```text
lessons/02_6月22日_上午下半场_模型接口与指令设计_正式授课版.md
```

## 讲解主线

这一部分的核心是 **TRANSITION（过渡）**：从本地模型到云端服务化。先讲清为何本地模型不够用，再逐步过渡到 HTTP API、环境变量管理、多后端对比，最后聚焦 DeepSeek 调用实战。

## 推荐运行顺序

```text
code/lesson_01_本地模型与服务化过渡/
├── 01_本地模型回顾.py              → 回顾第一章本地模型运行方式
├── 04_HTTP协议与模型调用.py        → HTTP 请求/响应结构，模型 API 调用
├── 05_从transformers到API.py       → transformers 库调用 vs REST API 调用
├── 06_环境变量与密钥管理.py        → 安全存储密钥，dotenv 使用
├── 07_请求与响应结构.py            → 完整请求体字段解析（messages、temperature 等）
├── 09_从本地到云端总结.py          → 过渡小结，为下一课 DeepSeek 调用做准备
└── 10_vLLM_生产级推理服务.py      → 高性能推理引擎，生产级自建部署（PagedAttention / 连续批处理）

code/lesson_02_DeepSeek调用与回答边界/
├── 01_最小_DeepSeek_请求体.py      → 最简可运行的 DeepSeek API 调用
├── 02_系统消息_控制边界.py         → system prompt 控制回答风格与范围
├── 03_结构化_JSON_输出.py          → response_format 强制 JSON 输出
├── 04_多案例批量测试.py            → 批量测试不同 prompt 效果
└── 05_错误与兜底.py                → API 异常处理、重试、降级策略
```

## 运行提示

调用 DeepSeek 前必须设置环境变量：

macOS / Linux：
```bash
export DEEPSEEK_API_KEY="客户自己的 API Key"
```

Windows PowerShell：
```powershell
$env:DEEPSEEK_API_KEY="客户自己的 API Key"
```

代码中不包含真实 API Key。建议先运行 lesson_01 理解服务化概念，再运行 lesson_02 进行实际 API 调用。