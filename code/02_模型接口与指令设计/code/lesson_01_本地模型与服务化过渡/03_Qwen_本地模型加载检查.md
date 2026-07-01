# Qwen 本地模型加载检查：代码说明

> 对应脚本：同名 Python 脚本。

## 1. 这段代码对应的行业需求

业务知识库问答：围绕订单服务规则、安全制度、应急预案、巡检记录和培训材料做可追溯检索。

## 2. 课堂目标

03 Qwen 本地模型加载检查

## 3. 讲解顺序

- 先用业务问题开场：业务现场为什么需要这段能力。
- 再讲输入是什么：用户问题、规则资料、图片/记录、工具参数或环境信息。
- 然后讲处理过程：每一步生成什么中间结果。
- 最后讲输出怎么看：哪些结果可信，哪些地方要人工确认或回到官方系统。

## 4. 代码结构

1. 先写脚本顶部说明：这段代码解决哪个业务系统问题，运行后应该看到什么。
2. 再写 import 和常量：优先使用标准库，路径使用 `Path(__file__).resolve().parent`。
3. 准备课堂模拟数据：例如订单服务规则、服务点、服务编号、巡检记录、工单状态。
4. 按“准备数据 -> 核心处理 -> 打印结果 -> 练习修改”的顺序写代码。
5. 最后写 `if __name__ == "__main__":` 入口，串起课堂演示流程。
6. 运行一次，确认输出里有中间结果、最终结果和可修改点。

## 5. 主要函数或代码块

- 这个脚本没有明显函数拆分，可以按代码段讲解。

## 6. 可修改点

- 把示例问题换成在线客服、安全检查、设备运维或培训答疑中的真实表达。
- 改一条模拟规则或模拟工具返回值，观察最终输出怎么变化。
- 故意制造一个缺字段、空结果或接口不可用的情况，说明兜底逻辑。
- 补充一条测试用例，检查输出是否仍然符合业务边界。

## 7. 运行方式

在脚本所在目录运行：

```bash
python 03_Qwen_本地模型加载检查.py
```

涉及 DeepSeek 时，只使用环境变量 `DEEPSEEK_API_KEY`，不要把 Key 写入文件。

## 8. 完整参考代码

```python
"""03 Qwen 本地模型加载检查。

按顺序检查：目录 -> 文件 -> tokenizer -> token id。
"""

from pathlib import Path

print("03 Qwen 本地模型加载检查")
print("=" * 72)

model_dir = None
for parent in Path(__file__).resolve().parents:
    candidate = parent / "open_models" / "Qwen3.5-0.8B"
    if candidate.exists():
        model_dir = candidate
        break

if model_dir is None:
    print("未找到模型目录：open_models/Qwen3.5-0.8B")
    raise SystemExit("请先下载 Qwen3.5-0.8B。")

print("一、模型目录")
print(model_dir)

print("\n二、关键文件")
for name in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
    path = model_dir / name
    print(f"{name:<24} {'存在' if path.exists() else '缺失'}")

print("\n三、加载 tokenizer")
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_dir)
text = "服务点A到服务点B的候补申请怎么解释？"
ids = tokenizer.encode(text, add_special_tokens=False)
print("输入：", text)
print("token ids：", ids[:30])
print("token 数量：", len(ids))
print("词表大小：", tokenizer.vocab_size)
print("是否有 chat_template：", bool(tokenizer.chat_template))
```