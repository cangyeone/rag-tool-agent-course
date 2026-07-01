# Qwen3.5 本地模型 transformers 实操说明

本说明用于 6 月 22 日上午课程中的“本地模型与服务化过渡”部分。它的目标不是训练模型，也不是追求复杂效果，而是可看清楚三件事：

1. 一段文字进入模型前会先变成 token。
2. 生成不是一次性变出整段话，而是一步一步往后接 token。
3. `messages` 这种 Chat API 格式，最终也要通过 chat template 变成模型能读的文本。

## 1. 模型放在哪里

本课程把 Qwen 模型放在课程目录的相对路径下：

```bash
rag-tool-agent-course/open_models/Qwen3.5-0.8B
```

下载命令：

```bash
modelscope download --model Qwen/Qwen3.5-0.8B --local_dir rag-tool-agent-course/open_models/Qwen3.5-0.8B
```

不要把模型路径写成 `个人电脑绝对路径` 这样的个人电脑绝对路径。课堂代码会从当前脚本目录向上查找 `open_models/Qwen3.5-0.8B`，这样课程文件夹整体复制到 Windows、macOS、Linux 后仍然能讲清路径逻辑。

## 2. 为什么先用 transformers

DeepSeek、可视化工具、Ollama 这类接口都很方便，但它们把很多底层细节封装掉了。上课时如果一开始只讲 API，参与者容易以为模型就是“发一句话，收一句话”。

用 `transformers` 本地跑一次 Qwen，可以把这几个环节摊开：

```text
输入文本
  -> AutoTokenizer
  -> input_ids
  -> AutoModelForCausalLM.generate()
  -> TextIteratorStreamer
  -> 逐步输出文本
```

这条链路能帮助参与者理解：大模型并不是直接理解“整句话”，而是把文本切成 token 后继续预测后面的 token。

注意：这个 Qwen3.5 模型使用较新的 `qwen3_5` 架构。若运行时看到 `Transformers does not recognize this architecture`，说明 `transformers` 版本还不够新。课堂环境已验证可用版本为开发版：

```bash
python -m pip install --upgrade "git+https://github.com/huggingface/transformers.git"
python -c "import transformers; print(transformers.__version__)"
```

验证时版本显示为 `5.13.0.dev0`。如果后续官方稳定版已经支持 `qwen3_5`，也可以改用稳定版。

## 3. 第一步：检查本地模型和 tokenizer

对应脚本：

```bash
cd rag-tool-agent-course/半天三节课_演示脚本/01_6月22日上午_大模型基础与DeepSeek接入/lesson_02_本地模型与服务化过渡
python 03_Qwen_本地模型加载检查.py
```

这段脚本主要看四件事：

- 本地模型目录是否存在。
- `config.json`、`tokenizer_config.json`、`model.safetensors` 是否在。
- `AutoTokenizer.from_pretrained()` 是否能读取模型。
- 一句 示例业务系统 咨询问题会被拆成哪些 token 和 token id。

课堂讲法可以很直接：

> 现在还没让模型回答问题，只是先看“模型读文字之前，文字长什么样”。如果 tokenizer 都加载不了，后面的生成一定跑不起来。

## 4. 第二步：最原始的逐步输出

对应脚本：

```bash
python 04_Qwen_原始逐步输出.py
```

这段脚本故意不使用 `system/user/assistant`，只给模型一个普通输入文本：

```python
input_text = "请用两句话解释：为什么 通用咨询助手不能承诺候补申请一定成功？"
```

然后代码会做几步：

```python
inputs = tokenizer(input_text, return_tensors="pt").to(device)
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
model.generate(..., streamer=streamer)
```

课堂上重点讲：

- `tokenizer(...)` 把输入文本变成 `input_ids`。
- `model.generate()` 是继续往后生成 token。
- `TextIteratorStreamer` 会把生成结果一点点吐出来。
- 逐步输出不等于模型在“思考过程可见”，它只是生成结果被分段返回。

这一页最好可观察终端：文字不是最后一次性出现，而是逐步出来。这就是很多聊天页面“正在打字”的基础。

## 5. 第三步：chat template 是什么

对应脚本：

```bash
python 05_Qwen_chat模板与API.py
```

Chat API 里我们常写这样的结构：

```python
messages = [
    {"role": "system", "content": "你是 通用咨询助手，只解释规则，不承诺真实库存。"},
    {"role": "user", "content": "ORD-1001 没票了，候补申请是不是一定能成功？"},
]
```

但本地模型真正接收的不是 Python 里的 list/dict，而是一段经过模板渲染后的文本，再进一步变成 token id。

所以脚本会调用：

```python
rendered_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

可以这样讲：

> `messages` 是程序和 API 看的结构，`chat template` 是模型看的格式。不同模型训练时用的角色标签可能不一样，所以不要手写标签，优先用 tokenizer 自带的模板。

## 6. Chat API 和本地 transformers 的关系

可以把二者对照着讲：

| 位置 | 本地 transformers | Chat API |
| --- | --- | --- |
| 输入结构 | 输入文本或 messages | messages |
| 角色标签 | `apply_chat_template()` 生成 | 服务端内部处理 |
| 生成函数 | `model.generate()` | `/chat/completions` |
| 逐步输出 | `TextIteratorStreamer` | `stream=True` |
| 模型位置 | 本地模型目录 | 云端或服务端 |
| 适合课堂讲解 | 能看清底层过程 | 能快速接业务应用 |

课堂顺序建议：

1. 先跑 tokenizer 检查。
2. 再跑原始输入文本 逐步输出。
3. 再展示 chat template。
4. 最后回到 DeepSeek API，说明 Chat API 帮我们封装了模板、生成和流式返回。

## 7. 常见问题

### 找不到模型目录

先确认模型是否下载到：

```bash
rag-tool-agent-course/open_models/Qwen3.5-0.8B
```

如果没有，重新运行：

```bash
modelscope download --model Qwen/Qwen3.5-0.8B --local_dir rag-tool-agent-course/open_models/Qwen3.5-0.8B
```

### 第一次运行很慢

正常。第一次要从磁盘读取模型权重，也可能要初始化后端。课堂上可以提前运行一次，避免现场等待太久。

### Apple 芯片能不能跑

脚本会优先尝试 `mps`，不可用就回到 `cpu`。0.8B 模型通常适合课堂演示，但不要把 `max_new_tokens` 设置太大。

### 输出内容不稳定怎么办

课堂演示可以先用：

```python
do_sample=False
max_new_tokens=60
```

这样输出更稳定。要演示创造性时，再讲 `temperature`、`top_p` 等参数。

## 8. 这一节和后续课程怎么衔接

本地 Qwen 可看见“模型输入输出的底层过程”。

后面的 DeepSeek API 负责说明“如何把模型能力接成稳定服务”。

RAG 负责说明“模型回答前应该先查资料”。

工具调用负责说明“实时状态不能靠模型猜，要查系统接口”。

可视化工具 负责说明“这些能力最后如何变成可视化工作流和课堂辅助助手”。