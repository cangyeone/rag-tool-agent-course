"""02_注册表校验。

工具调用前要先检查：
1. 工具是否存在
2. 必填参数是否齐全
3. 参数值是否在允许范围内
"""

from __future__ import annotations

import json
from pathlib import Path

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("02_注册表校验（简化版）")
print("=" * 72)

registry = {
    "search_policy": {
        "required": ["query"],
    },
    "query_ticket": {
        "required": ["train_no"],
        "enum": {
            "seat_type": ["标准服务", "高级服务", "商务座", "全部"],
        },
    },
}


def validate(tool_name: str, args: dict) -> tuple[bool, list[str]]:
    errors = []

    if tool_name not in registry:
        return False, [f"工具不存在：{tool_name}"]

    rule = registry[tool_name]

    for field in rule.get("required", []):
        if not args.get(field):
            errors.append(f"缺少必填参数：{field}")

    for field, allowed_values in rule.get("enum", {}).items():
        if field in args and args[field] not in allowed_values:
            errors.append(f"{field}={args[field]} 不在允许范围：{allowed_values}")

    return len(errors) == 0, errors


test_calls = [
    ("search_policy", {"query": "候补申请是否保证成功"}),
    ("search_policy", {}),
    ("query_ticket", {"train_no": "G107", "seat_type": "标准服务"}),
    ("query_ticket", {"train_no": "G107", "seat_type": "硬座"}),
    ("unknown_tool", {"query": "候补申请"}),
]

print("\n工具注册表：")
print(json.dumps(registry, ensure_ascii=False, indent=2))

print("\n校验测试：")
for tool_name, args in test_calls:
    ok, errors = validate(tool_name, args)
    print("\n工具：", tool_name)
    print("参数：", json.dumps(args, ensure_ascii=False))
    print("结果：", "通过" if ok else "拒绝")
    for error in errors:
        print("原因：", error)

print("\n课堂可修改点")
print("1. 给 query_ticket 增加必填参数 date。")
print("2. 给 search_policy 增加 query 最小长度检查。")
print("3. 讨论：为什么模型输出的参数不能直接交给业务系统执行。")
