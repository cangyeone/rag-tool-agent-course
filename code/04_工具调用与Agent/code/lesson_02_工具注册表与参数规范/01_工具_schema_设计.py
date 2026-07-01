"""01_工具_schema_设计。

用一个最小例子说明工具 schema。
schema 的作用：告诉模型“这个工具叫什么、什么时候用、参数怎么填”。
"""

from __future__ import annotations

import json
from pathlib import Path

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("01_工具_schema_设计（简化版）")
print("=" * 72)

tool_schema = {
    "type": "function",
    "function": {
        "name": "search_policy",
        "description": "查询订单服务政策、候补申请、退款变更等规则资料。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要检索的政策问题，例如：候补申请是否保证成功",
                }
            },
            "required": ["query"],
        },
    },
}

print("\n一个最小工具 schema：")
print(json.dumps(tool_schema, ensure_ascii=False, indent=2))

print("\n逐项解释")
print("1. name：工具名，程序会按这个名字找到函数。")
print("2. description：工具用途，模型主要靠它判断该不该调用。")
print("3. parameters：参数结构，写成 JSON Schema。")
print("4. required：必填参数，缺了就不能执行。")

print("\n模型可能生成的工具调用：")
tool_call = {
    "name": "search_policy",
    "arguments": {
        "query": "候补申请是否保证成功",
    },
}
print(json.dumps(tool_call, ensure_ascii=False, indent=2))

print("\n课堂可修改点")
print("1. 把工具名改成 query_ticket，思考 description 应该怎么写。")
print("2. 给工具增加 train_no 参数，观察 required 应该怎么改。")
print("3. description 写得越清楚，模型越容易选对工具。")
