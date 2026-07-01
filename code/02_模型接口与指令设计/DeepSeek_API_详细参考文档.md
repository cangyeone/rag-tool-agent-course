# DeepSeek API 详细参考文档

> 基于 [DeepSeek API 官方文档](https://api-docs.deepseek.com/zh-cn/) 整理，用于 示例业务系统 AI 培训课程第 02 章配套参考。
> 文档更新时间：2026-06

---

## 目录

1. [快速开始](#1-快速开始)
2. [模型与价格](#2-模型与价格)
3. [Chat Completions API](#3-chat-completions-api)
4. [思考模式](#4-思考模式)
5. [Tool Calls（函数调用）](#5-tool-calls函数调用)
6. [JSON Output](#6-json-output)
7. [错误码](#7-错误码)
8. [限速与并发](#8-限速与并发)
9. [示例业务系统 场景最佳实践](#9-示例业务系统-场景最佳实践)

---

## 1. 快速开始

### 1.1 获取 API Key

1. 访问 [DeepSeek Platform](https://platform.deepseek.com/)
2. 注册/登录账号
3. 进入 [API Keys](https://platform.deepseek.com/api_keys) 页面
4. 点击「创建 API Key」，保存生成的 key（只显示一次）

### 1.2 设置环境变量

```bash
# macOS / Linux
export DEEPSEEK_API_KEY="YOUR_API_KEY_HERE"

# Windows PowerShell
$env:DEEPSEEK_API_KEY="YOUR_API_KEY_HERE"
```

### 1.3 最小可运行示例

```python
from openai import OpenAI

client = OpenAI(
    api_key="<your api key>",
    base_url="https://api.deepseek.com",
)

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "system", "content": "你是 通用客服助手。"},
        {"role": "user", "content": "候补申请能保证成功吗？"},
    ],
    max_tokens=200,
    temperature=0.2,
)

print(response.choices[0].message.content)
```

---

## 2. 模型与价格

### 2.1 当前可用模型

| 项目 | deepseek-v4-flash | deepseek-v4-pro |
|------|-------------------|-----------------|
| 模型版本 | DeepSeek-V4-Flash | DeepSeek-V4-Pro |
| 上下文长度 | 1M tokens | 1M tokens |
| 输出长度 | 最大 384K | 最大 384K |
| 思考模式 | 支持（默认） | 支持（默认） |
| JSON Output | ✓ | ✓ |
| Tool Calls | ✓ | ✓ |
| FIM 补全 | 仅非思考模式 | 仅非思考模式 |
| 并发限制 | 2500 | 500 |

### 2.2 价格（元 / 百万 tokens）

| 计费项 | deepseek-v4-flash | deepseek-v4-pro |
|--------|-------------------|-----------------|
| 输入（缓存命中） | ¥0.02 | ¥0.025 |
| 输入（缓存未命中） | ¥1.00 | ¥3.00 |
| 输出 | ¥2.00 | ¥6.00 |

> **价格极低**：1 元 ≈ 100 万 token 输入，约等于处理一本《三体》级别的文本量。
> 教学使用几乎零成本。

### 2.3 推荐模型名

| 推荐模型 | 适用方式 | 说明 |
|---|---|---|
| `deepseek-v4-flash` | 非思考 / 思考均可 | 默认演示模型，速度快、成本低 |
| `deepseek-v4-pro` | 复杂推理、复杂工具调用 | 质量更高，适合代码、规划和复杂分析 |

---

## 3. Chat Completions API

### 3.1 接口地址

```
base_url = https://api.deepseek.com
POST /chat/completions
```

与 OpenAI API 格式兼容。使用 `requests` 时，实际请求地址是 `base_url + "/chat/completions"`；使用 OpenAI SDK 时，只需要把 `base_url` 设置为 `https://api.deepseek.com`。

### 3.2 请求参数详解

#### 必填参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | string | 模型 ID：`deepseek-v4-pro` 或 `deepseek-v4-flash` |
| `messages` | object[] | 对话消息列表，至少 1 条 |

#### messages 结构

```json
[
  {
    "role": "system",
    "content": "你是 通用客服助手，回答要简洁..."
  },
  {
    "role": "user",
    "content": "ORD-1001 没票了怎么办？"
  }
]
```

**四种角色：**

| role | 用途 | 何时出现 |
|------|------|---------|
| `system` | 设定助手身份、行为边界、输出风格 | 对话开始时（建议放在最前面） |
| `user` | 用户提问 | 每轮用户输入 |
| `assistant` | 模型回答 | 上一轮的模型回答，回传给 API 保持上下文 |
| `tool` | 工具执行结果 | 调用完外部工具后，把结果回传给模型 |

**每条消息的完整字段：**

```json
// System message
{
  "role": "system",
  "content": "系统提示词内容",
  "name": "可选：参与者名称"
}

// User message
{
  "role": "user",
  "content": "用户问题",
  "name": "可选：参与者名称"
}

// Assistant message
{
  "role": "assistant",
  "content": "模型回答内容",
  "name": "可选：参与者名称",
  "prefix": false,            // Beta: 强制以此前缀开始回答
  "reasoning_content": "思维链内容"  // 思考模式下的推理过程
}

// Tool message
{
  "role": "tool",
  "tool_call_id": "call_xxxxxx",
  "content": "工具执行结果"
}
```

#### 常用可选参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temperature` | number | 1 | 0~2，越高越随机，越低越确定 |
| `top_p` | number | 1 | 核采样概率阈值，与 temperature 二选一调整 |
| `max_tokens` | integer | - | 限制输出最大 token 数 |
| `stream` | boolean | false | true 时以 SSE 流式返回 |
| `stop` | string/string[] | - | 遇到这些词时停止生成，最多 16 个 |

#### 思考模式参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `thinking.type` | `"enabled"` / `"disabled"` | 思考模式开关，默认 enabled |
| `reasoning_effort` | `"high"` / `"max"` | 推理强度，普通请求默认 high |

> **注意**：思考模式下 `temperature`、`top_p`、`frequency_penalty`、`presence_penalty` 不会生效（设置不报错但无效果）。

#### Function Calling 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `tools` | object[] | 可用工具列表，最多 128 个 function |
| `tool_choice` | string/object | `"none"` / `"auto"` / `"required"` 或指定具体 function |

详见 [第 5 节](#5-tool-calls函数调用)。

#### 其它参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `response_format` | object | `{"type": "json_object"}` 启用 JSON 模式 |
| `stream_options.include_usage` | boolean | 流式末尾返回 token 用量统计 |
| `logprobs` | boolean | 是否返回每个 token 的对数概率 |
| `top_logprobs` | integer | 返回 top N 个概率最高的 token（0~20） |
| `user_id` | string | 用户标识，用于安全隔离和并发管理（最大 512 字符，正则 `[a-zA-Z0-9\-_]+`） |

### 3.3 响应结构

#### 非流式响应

```json
{
  "id": "930c60df-bf64-41c9-a88e-3ec75f81e00e",
  "object": "chat.completion",
  "created": 1705651092,
  "model": "deepseek-v4-pro",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "可以尝试候补申请，但不能保证一定成功...",
        "reasoning_content": "先分析候补申请规则...",
        "tool_calls": null
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 120,
    "total_tokens": 165,
    "prompt_cache_hit_tokens": 0,
    "prompt_cache_miss_tokens": 45,
    "completion_tokens_details": {
      "reasoning_tokens": 80
    }
  },
  "system_fingerprint": "fp_xxxxx"
}
```

#### finish_reason 含义

| 值 | 含义 |
|----|------|
| `stop` | 正常结束或遇到 stop 词 |
| `length` | 达到 max_tokens 或上下文长度上限 |
| `content_filter` | 内容触发过滤策略 |
| `tool_calls` | 模型决定调用工具 |
| `insufficient_system_resource` | 推理资源不足，被打断 |

#### 流式响应的 SSE 格式

```
data: {"id":"...","choices":[{"delta":{"role":"assistant","content":""},"index":0}],...}
data: {"id":"...","choices":[{"delta":{"content":"可以"},"index":0}],...}
data: {"id":"...","choices":[{"delta":{"content":"尝试"},"index":0}],...}
...
data: {"id":"...","choices":[{"delta":{"content":""},"finish_reason":"stop","index":0}],...,"usage":{...}}
data: [DONE]
```

---

## 4. 思考模式

### 4.1 什么是思考模式

模型在输出最终回答前，先输出一段**思维链**（reasoning_content），用于：
- 多步推理（数学、逻辑、复杂规则分析）
- 自我验证和纠错
- 工具调用的决策规划

### 4.2 开关控制

```python
# 开启思考模式（默认就是开启的）
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    extra_body={"thinking": {"type": "enabled"}},
    reasoning_effort="high",   # 推理强度
)

# 关闭思考模式（适合简单问答）
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    extra_body={"thinking": {"type": "disabled"}},
)
```

### 4.3 推理强度选择

| 值 | 适用场景 |
|----|---------|
| `high` | 普通推理（默认），适合大部分客服问答 |
| `max` | 复杂推理，适合多步骤规则分析、工单诊断 |

### 4.4 多轮对话中的思维链处理

**无工具调用时**：reasoning_content 在下一轮中会被 API 自动忽略，不需要手动处理。

```
Turn 1: user -> assistant (含 reasoning) -> 回答
Turn 2: user -> 只需要回传 assistant 的 content，reasoning 会被忽略
```

**有工具调用时**：reasoning_content 必须在后续所有轮次中**完整回传**。

```python
# 正确做法：直接 append 整个 message 对象
messages.append(response.choices[0].message)

# 这行等价于：
messages.append({
    "role": "assistant",
    "content": response.choices[0].message.content,
    "reasoning_content": response.choices[0].message.reasoning_content,
    "tool_calls": response.choices[0].message.tool_calls,
})
```

> **常见错误**：手动构建 assistant message 时忘了带 `reasoning_content`，导致 API 返回 400 错误。

### 4.5 流式读取思考内容

```python
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    stream=True,
    extra_body={"thinking": {"type": "enabled"}},
)

reasoning_content = ""
content = ""

for chunk in response:
    delta = chunk.choices[0].delta
    if delta.reasoning_content:
        reasoning_content += delta.reasoning_content
        print(delta.reasoning_content, end="", flush=True)  # 实时显示思考过程
    elif delta.content:
        content += delta.content
        print(delta.content, end="", flush=True)  # 实时显示回答
```

---

## 5. Tool Calls（函数调用）

### 5.1 工作原理

```
用户提问 → 模型分析 → 决定调用工具 → 返回 tool_calls
    → 你的代码执行工具 → 结果回传给模型 → 模型生成最终回答
```

### 5.2 工具定义（JSON Schema）

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_service_point_code",
            "description": "根据中文服务点名称查询服务点编码，用于后续服务编号查询",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_point_name": {
                        "type": "string",
                        "description": "中文服务点名称，例如 服务点A、服务点B"
                    }
                },
                "required": ["service_point_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_waiting_rules",
            "description": "查询候补申请规则，返回候补申请成功的条件和限制",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "服务编号号，例如 ORD-1001"
                    },
                    "date": {
                        "type": "string",
                        "description": "日期，格式 YYYY-MM-DD"
                    }
                },
                "required": ["order_id"]
            }
        }
    }
]
```

### 5.3 tool_choice 控制

| 值 | 行为 |
|----|------|
| `"none"` | 不调用任何工具，直接回答 |
| `"auto"` | 模型自行决定（默认） |
| `"required"` | 必须调用至少一个工具 |
| `{"type": "function", "function": {"name": "xxx"}}` | 强制调用指定工具 |

### 5.4 完整调用流程示例

```python
from openai import OpenAI
import json

client = OpenAI(
    api_key="<your api key>",
    base_url="https://api.deepseek.com",
)

def query_service_point_code(service_point_name: str) -> str:
    """模拟服务点编码查询"""
    codes = {"服务点A": "SP-A", "服务点B": "SP-B", "杭州东": "HZH"}
    return codes.get(service_point_name, "未知服务点")

def check_waiting_rules(order_id: str, date: str = None) -> str:
    """模拟候补申请规则查询"""
    return json.dumps({
        "can_wait": True,
        "note": "候补申请不能保证成功，兑现取决于库存变化，最终以示例业务系统官方页面为准。"
    }, ensure_ascii=False)

# 工具注册表
TOOL_MAP = {
    "query_service_point_code": query_service_point_code,
    "check_waiting_rules": check_waiting_rules,
}

def run_agent(question: str):
    messages = [{"role": "user", "content": question}]

    while True:
        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        messages.append(msg)

        # 如果没有 tool_calls，说明模型已经给出了最终回答
        if not msg.tool_calls:
            return msg.content

        # 执行工具调用
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            result = TOOL_MAP[func_name](**func_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })

# 使用
answer = run_agent("服务点A的服务点编码是什么？ORD-1001 能候补申请吗？")
print(answer)
```

### 5.5 strict 模式（Beta）

确保模型输出的 function 参数严格符合 JSON Schema。

```python
# 需要用 beta base_url
client = OpenAI(
    base_url="https://api.deepseek.com/beta",
)

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "strict": True,  # 启用严格模式
        "description": "...",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "城市名"}
            },
            "required": ["location"],
            "additionalProperties": False  # strict 模式必须设为 false
        }
    }
}]
```

---

## 6. JSON Output

### 6.1 启用方式

```python
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "system", "content": "返回 JSON 格式。"},
        {"role": "user", "content": "分析候补申请规则"},
    ],
    response_format={"type": "json_object"},
)
```

> **重要**：必须在 system 或 user 消息中明确指示模型输出 JSON，否则可能生成空白字符直到 token 耗尽。

### 6.2 示例业务系统 场景示例

```python
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {
            "role": "system",
            "content": (
                "你是 示例业务系统 客服分析助手。"
                "分析用户问题，输出如下 JSON：\n"
                '{"intent": "咨询类型", "risk_level": "低/中/高", '
                '"answer": "回答内容", "need_escalation": true/false}'
            )
        },
        {"role": "user", "content": "候补申请能不能保证成功？"}
    ],
    response_format={"type": "json_object"},
    temperature=0.2,
)

result = json.loads(response.choices[0].message.content)
print(f"意图：{result['intent']}")
print(f"风险：{result['risk_level']}")
print(f"需要升级：{result['need_escalation']}")
```

---

## 7. 错误码

| 状态码 | 原因 | 解决方法 |
|--------|------|---------|
| **400** | 请求体格式错误 | 检查 JSON 结构，参考本文档第 3 节 |
| **401** | API Key 错误 | 检查 `DEEPSEEK_API_KEY` 环境变量是否正确 |
| **402** | 余额不足 | 前往 [充值页面](https://platform.deepseek.com/top_up) |
| **422** | 请求体参数错误 | 检查 model/messages 等参数名是否正确 |
| **429** | 并发超限 | 降低请求频率，或提交扩容申请 |
| **500** | 服务器内部错误 | 等待后重试 |
| **503** | 服务器繁忙 | 稍后重试 |

### 7.1 常见错误排查

```python
# 课堂常见 401 错误
# 原因：环境变量未设置或拼写错误
import os
key = os.getenv("DEEPSEEK_API_KEY")
if not key:
    print("请设置 DEEPSEEK_API_KEY 环境变量")
elif not key.startswith("sk-"):
    print("API Key 格式不对，应该以 sk- 开头")

# 课堂常见 422 错误
# 原因：model 名称写错
# 正确：deepseek-v4-pro 或 deepseek-v4-flash
```

---

## 8. 限速与并发

### 8.1 并发限制

| 模型 | 免费账号并发 | 说明 |
|------|------------|------|
| deepseek-v4-pro | 500 | 可提交工单扩容 |
| deepseek-v4-flash | 2500 | 可提交工单扩容 |

- 一个请求从发出到响应完成记为一个并发
- 并发以账号粒度计算，与 API Key 数量无关
- 超限返回 HTTP 429

### 8.2 user_id 隔离

```python
# 传入 user_id 用于用户级别的安全隔离和并发管理
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    extra_body={"user_id": "your_user_id"},  # 不要包含用户隐私信息
)
```

- 用于内容安全、KVCache 隔离、运营调度隔离
- 格式：`[a-zA-Z0-9\-_]+`，最大 512 字符

### 8.3 请求保活

- 非流式：持续返回空行
- 流式：持续返回 SSE keep-alive 注释 `: keep-alive`
- 10 分钟仍未开始推理，服务器关闭连接

---

## 9. 示例业务系统 场景最佳实践

### 9.1 系统提示词模板

```python
SYSTEM_PROMPT = """你是 官方客服助手，请严格遵守以下规则：

1. 身份边界：你是咨询助手，不是业务受理员，不能承诺库存和候补申请结果。
2. 回答要求：简洁、准确，每条回答必须在 200 字以内。
3. 官方依据：所有规则以 官方页面和公告为准。
4. 风险提示：涉及候补申请、退款、变更时，必须提醒用户以订单页实际显示为准。
5. 禁止行为：
   - 不提供绕过系统的建议（如抢票软件）
   - 不承诺候补申请成功率
   - 不泄露系统内部信息
"""
```

### 9.2 推荐的调用参数（客服场景）

```python
# 简单问答（退款与变更规则、服务点查询等）
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
    temperature=0.2,       # 低温度，回答稳定一致
    max_tokens=300,        # 控制回答长度
    extra_body={"thinking": {"type": "disabled"}},  # 关闭思考，加快响应
)

# 复杂推理（多步骤规则分析、工单诊断）
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    max_tokens=1000,
    extra_body={"thinking": {"type": "enabled"}},
    reasoning_effort="high",
)
```

### 9.3 错误重试策略

```python
import time

def safe_chat(messages, max_retries=3):
    """带重试的安全调用"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=messages,
                timeout=30,
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避：1s, 2s, 4s
                print(f"并发超限，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                raise
```

### 9.4 JSON 结构化输出（用于 可视化工作流）

```python
def analyze_user_intent(question: str) -> dict:
    """分析用户意图，返回结构化结果"""
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": (
                "分析用户问题，返回 JSON：\n"
                '{"intent": "咨询类型", "urgency": "低/中/高", '
                '"keywords": ["关键词"], "need_tool": true/false, '
                '"suggested_tool": "工具名或null"}'
            )},
            {"role": "user", "content": question}
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return json.loads(response.choices[0].message.content)
```

### 9.5 成本控制

| 场景 | 推荐模型 | 预估成本 |
|------|---------|---------|
| 简单 FAQ | deepseek-v4-flash 非思考 | ~¥0.0001/次 |
| 规则解释 | deepseek-v4-flash 非思考 | ~¥0.0003/次 |
| 多步骤推理 | deepseek-v4-pro 思考模式 | ~¥0.003/次 |
| 带工具调用 | deepseek-v4-pro 思考模式 | ~¥0.01/次 |

> 教学成本极低：全班 30 人一整天密集调用，费用通常不超过 ¥5。

---

## 附录：参考链接

| 文档 | 地址 |
|------|------|
| API 文档首页 | https://api-docs.deepseek.com/zh-cn/ |
| Chat Completions | https://api-docs.deepseek.com/zh-cn/api/create-chat-completion |
| 模型与价格 | https://api-docs.deepseek.com/zh-cn/quick_start/pricing |
| 思考模式 | https://api-docs.deepseek.com/zh-cn/guides/thinking_mode |
| Tool Calls | https://api-docs.deepseek.com/zh-cn/guides/tool_calls |
| JSON Output | https://api-docs.deepseek.com/zh-cn/guides/json_mode |
| 错误码 | https://api-docs.deepseek.com/zh-cn/quick_start/error_codes |
| 限速与隔离 | https://api-docs.deepseek.com/zh-cn/quick_start/rate_limit |
| DeepSeek Platform | https://platform.deepseek.com/ |