"""02_工具调用的基本流程。

这是示意性版本，不调用大模型。
目标是把工具调用拆成 4 个最容易看懂的步骤：
1. 看问题
2. 选工具
3. 执行工具
4. 根据工具结果回答
"""

from __future__ import annotations

import json
from pathlib import Path

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("02_工具调用的基本流程（示意版）")
print("=" * 72)


def search_policy(query: str) -> dict:
    """模拟知识库检索工具。"""
    return {
        "tool": "search_policy",
        "query": query,
        "title": "候补申请规则",
        "content": "候补申请不能保证一定成功，兑现结果取决于退款、变更、新增席位和排队顺序。",
        "source": "课堂业务样例",
    }


def query_order_status(order_id: str) -> dict:
    """模拟库存查询工具。"""
    return {
        "tool": "query_order_status",
        "order_id": order_id,
        "date": "2026-06-22",
        "remaining": 0,
        "suggestion": "可尝试候补申请，或查看同方向其他服务编号。",
    }


question = "ORD-1001 没票了，候补申请一定能成功吗？"
print("用户问题：", question)

# 第一步：人工写一个非常简单的规则，判断要用哪些工具。
need_order_tool = "ORD-1001" in question or "没票" in question
need_policy_tool = "候补申请" in question or "成功" in question

print("\n一、判断需要哪些工具")
print("需要库存工具：", need_order_tool)
print("需要政策检索：", need_policy_tool)

# 第二步：执行工具。
tool_results = []

if need_order_tool:
    order_result = query_order_status("ORD-1001")
    tool_results.append(order_result)

if need_policy_tool:
    policy_result = search_policy("候补申请是否保证成功")
    tool_results.append(policy_result)

print("\n二、工具返回结果")
for result in tool_results:
    print(json.dumps(result, ensure_ascii=False, indent=2))

# 第三步：把工具结果整理成人能看懂的答复。
answer = (
    "ORD-1001 当前示例数据中显示无库存。"
    "候补申请不能保证一定成功，是否兑现取决于退款、变更、新增席位和排队顺序。"
    "建议提交候补申请后继续关注订单状态，也可以查看同方向其他服务编号。"
    "最终以官方页面实时显示为准。"
)

print("\n三、最终回答")
print(answer)

print("\n课堂可修改点")
print("1. 把问题改成“退款费怎么算”，观察需要换成什么工具。")
print("2. 修改 query_order_status 的 remaining，观察最终回答应该怎么改。")
print("3. 讨论：工具只返回数据，真正负责执行工具的是程序，不是模型。")
