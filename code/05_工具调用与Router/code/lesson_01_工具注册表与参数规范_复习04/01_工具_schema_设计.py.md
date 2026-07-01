# 工具 schema 设计：代码说明

> 对应代码：`半天三节课_演示脚本/03_6月23日上午_工具注册表与多工具Agent/lesson_01_工具注册表与参数规范/01_工具_schema_设计.py`

## 1. 这段代码对应的行业需求

业务系统工具调用：把库存、订单、服务点、设备、工单状态等确定性查询交给工具，而不是让模型猜。

## 2. 课堂目标

01_工具_schema_设计

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
python 01_工具_schema_设计.py
```

涉及 DeepSeek 时，只使用环境变量 `DEEPSEEK_API_KEY`，不要把 Key 写入文件。

## 8. 完整参考代码

```python
"""01_工具_schema_设计。

脚本说明：
- 本脚本展示一个独立的业务演示环节。
- 输出包含输入、处理中间结果和最终结果。
- 数据全部为模拟数据，不连接真实 示例业务系统 或生产系统。
"""

from pathlib import Path
from collections import Counter
import json
import math
import os
import re
import time

print("01_工具_schema_设计")
print("=" * 72)
print("知识地图位置：03_6月23日上午_工具注册表与多工具Agent / lesson_01_工具注册表与参数规范")
print("演示目标：把工具、参数、路由、Agent 执行过程拆开讲，避免模型凭空猜业务事实。")
print()

# 第 1 步：准备工具注册表。真实系统会把这些信息给模型或工作流节点。
tools = {
    "station_code": {"desc": "查询服务点编码", "required": ["station_name"]},
    "ticket_advice": {"desc": "根据订单编号和场景给下单建议", "required": ["train_no", "scene"]},
    "refund_rule": {"desc": "解释退变更注意事项", "required": ["ticket_type"]},
}
print("一、工具注册表")
print(json.dumps(tools, ensure_ascii=False, indent=2))

question = "北京南到上海虹桥的 G107 没票了，还能怎么办？"
print("\n二、用户问题：", question)

# 第 2 步：抽取参数。课堂用规则演示，真实系统可让模型输出 JSON。
station_name = "北京南" if "北京南" in question else "未知服务点"
train_no = "G107" if "G107" in question else "未知订单编号"
scene = "候补申请咨询" if "没票" in question or "候补申请" in question else "普通咨询"
params = {"station_name": station_name, "train_no": train_no, "scene": scene}
print("\n三、抽取到的参数")
print(json.dumps(params, ensure_ascii=False, indent=2))

# 第 3 步：校验 required 字段。工具调用失败，很多时候不是模型差，而是参数不完整。
for tool_name, schema in tools.items():
    missing = [name for name in schema["required"] if not params.get(name)]
    print(f"工具 {tool_name} 缺失参数：{missing if missing else '无'}")

print("\n课堂可修改点：")
print("1. 修改 question，观察检索、路由或输出是否变化。")
print("2. 修改 docs / tools / workflow 中的一条模拟数据，看结果如何变化。")
print("3. 故意删掉一个字段，讨论真实系统为什么需要兜底和日志。")
print("4. 把当前脚本输出截图或记录下来，作为课堂复盘材料。")
```