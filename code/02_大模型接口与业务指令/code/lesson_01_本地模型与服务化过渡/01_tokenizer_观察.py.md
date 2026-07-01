# tokenizer 观察：代码说明

> 对应代码：`半天三节课_演示脚本/01_6月22日上午_大模型基础与DeepSeek接入/lesson_02_本地模型与服务化过渡/01_tokenizer_观察.py`

## 1. 这段代码对应的行业需求

业务知识库问答：围绕订单服务规则、安全制度、应急预案、巡检记录和培训材料做可追溯检索。

## 2. 课堂目标

01 tokenizer 观察

## 3. 讲解顺序

- 先用业务问题开场：业务现场为什么需要这段能力。
- 再讲输入是什么：用户问题、规则资料、图片/记录、工具参数或环境信息。
- 然后讲处理过程：每一步生成什么中间结果。
- 最后讲输出怎么看：哪些结果可信，哪些地方要人工确认或回到官方系统。

## 4. 代码结构

1. 先写脚本顶部说明：这段代码解决哪个业务系统问题，运行后应该看到什么。
2. 再写 import 和常量：优先使用标准库，路径使用 `Path(__file__).resolve().parent`。
3. 准备课堂模拟数据：例如订单服务规则、服务点、订单编号、巡检记录、工单状态。
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
python 01_tokenizer_观察.py
```

涉及 DeepSeek 时，只使用环境变量 `DEEPSEEK_API_KEY`，不要把 Key 写入文件。

## 8. 完整参考代码

```python
"""01 tokenizer 观察。

优先读取本地 Qwen tokenizer；如果环境不支持，就用简化切分做课堂说明。
"""

from pathlib import Path

print("01 tokenizer 观察")
print("=" * 72)

text = "北京南到上海虹桥的候补申请怎么解释？"
model_dir = None
for parent in Path(__file__).resolve().parents:
    candidate = parent / "open_models" / "Qwen3.5-0.8B"
    if candidate.exists():
        model_dir = candidate
        break

print("输入文本：", text)
print("模型目录：", model_dir if model_dir else "未找到 open_models/Qwen3.5-0.8B")

try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text, add_special_tokens=False)
    print("\n一、真实 tokenizer 输出")
    print("tokens:", tokens[:20])
    print("ids   :", ids[:20])
except Exception as exc:
    print("\n无法加载真实 tokenizer，使用课堂简化版。原因：", exc)
    tokens = list(text)
    ids = [abs(hash(ch)) % 10000 for ch in tokens]
    print("tokens:", tokens)
    print("ids   :", ids)

print("\n要点：tokenizer 负责把文字变成模型能计算的数字。")
```