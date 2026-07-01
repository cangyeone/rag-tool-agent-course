# 阿里系模型在线调用

本文件夹演示阿里百炼 OpenAI 兼容接口的基本用法。课堂主线从最简文本调用开始，再加入参数控制、thinking 控制，最后演示多模态图像理解。

## 运行前准备

请先进入课程根目录：

```bash
cd rag-tool-agent-course
```

脚本中统一使用以下配置：

```python
API_BASE = os.getenv("BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = "qwen3.7-plus"
```

为避免交付资料中携带真实密钥，脚本顶部保留了 `API_KEY` 变量。课堂演示时在本机填入即可，交付前请保持为空。

## 脚本顺序

1. `01_阿里百炼_最简文本调用.py`
   - 最少字段完成一次文本问答
   - 说明 `model`、`messages`、`stream`

2. `02_阿里百炼_参数控制.py`
   - 加入 `system`、`temperature`、`top_p`、`max_tokens`
   - 对比稳定回答、发散回答、短回答

3. `03_阿里百炼_thinking控制.py`
   - 演示 `enable_thinking`
   - 演示非流式请求中的完整 `reasoning_content`
   - 使用流式输出读取 `reasoning_content`
   - 提供 quick / standard / deep 三档 `thinking_budget`
   - 打印原始 SSE 响应，展示 `data:`、`[DONE]`、转义字符等接口符号
   - 说明普通回答与深度思考的适用差异
   - 增加 JSON 输出示例

4. `04_阿里百炼_多模态图像解释.py`
   - 生成一张课堂演示图片
   - 使用 OpenAI 兼容 `image_url` 格式输入图像
   - 让模型读取图中订单编号、服务点、办理窗口、时间等信息

## 参数简表

| 参数 | 作用 |
| --- | --- |
| `model` | 选择模型，本课程使用 `qwen3.7-plus` |
| `messages` | 对话消息，按 `system`、`user`、`assistant` 顺序组织 |
| `temperature` | 控制输出随机性，越低越稳定 |
| `top_p` | 控制候选 token 范围，通常不和 temperature 同时大幅调整 |
| `max_tokens` | 控制最大输出长度 |
| `stream` | 是否流式返回 |
| `enable_thinking` | 是否启用思考模式，属于百炼/Qwen 扩展参数；直接 HTTP 请求时放在请求体顶层 |
| `thinking_budget` | 控制思考过程最多消耗多少 token；脚本中提供 quick / standard / deep 三档 |
| `stream_options.include_usage` | 流式输出时在最后返回 token 用量 |
| `response_format` | 要求模型输出 JSON 等结构化结果 |

## thinking 层级

`03_阿里百炼_thinking控制.py` 顶部可修改：

```python
SELECTED_THINKING_LEVEL = "standard"
RUN_ALL_THINKING_LEVELS = False
SHOW_RAW_RESPONSE = True
```

可选层级：

| 层级 | thinking_budget | 适用情况 |
| --- | ---: | --- |
| quick | 512 | 简单判断、课堂快速演示 |
| standard | 2048 | 常规分析，推荐默认 |
| deep | 4096 | 复杂判断，耗时和 token 更多 |