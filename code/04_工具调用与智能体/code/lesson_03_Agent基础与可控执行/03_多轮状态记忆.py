"""03_多轮状态记忆 — 跨轮次的状态追踪、上下文管理、会话生命周期。

学习目标：管理 Agent 在多轮对话中的状态，让 Agent 记住「聊到哪了」和「前面做了什么」。
"""

import json
import time
from typing import Dict, List, Optional

print("03_多轮状态记忆")
print("=" * 72)
print("无状态 Agent = 每次对话从零开始。有状态 Agent = 记得上下文和已做的事。")
print()

# ========== 会话状态管理器 ==========
class SessionState:
    """管理一个用户会话的完整状态。"""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.history: List[Dict] = []       # 对话历史
        self.tool_calls: List[Dict] = []    # 已调用的工具记录
        self.context: Dict = {}             # 当前上下文（用户信息、上次查询结果等）
        self.status = "active"

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content, "time": time.time()})

    def record_tool_call(self, tool_name: str, args: dict, result: dict):
        self.tool_calls.append({"tool": tool_name, "args": args, "result": result,
                                "time": time.time()})

    def set_context(self, key: str, value):
        self.context[key] = value

    def get_context(self, key: str) -> Optional[any]:
        return self.context.get(key)

    def summarize(self):
        return {
            "session_id": self.session_id,
            "turns": len([m for m in self.history if m["role"] == "user"]),
            "tool_calls": len(self.tool_calls),
            "context_keys": list(self.context.keys()),
            "age_seconds": round(time.time() - self.created_at),
        }

# ========== 模拟多轮对话 ==========
session = SessionState("sess_20260619_001")

# 第一轮：用户查库存
print("【第 1 轮：用户查库存】")
q1 = "帮我查明天 ORD-1001 从服务点A出发的票"
session.add_message("user", q1)
# 模拟 Agent 调用工具
session.record_tool_call("query_order_statuss",
                         {"order_id": "ORD-1001", "date": "2026-06-20"},
                         {"remaining": 0, "status": "无库存"})
session.set_context("last_query_train", "ORD-1001")
session.set_context("last_query_date", "2026-06-20")
session.set_context("last_query_result", "无库存")
session.add_message("assistant", "ORD-1001 明日已无库存，需要帮您查看其他服务编号吗？")
print(f"  用户: {q1}")
print(f"  [工具调用] query_order_statuss(ORD-1001) → 无库存")
print(f"  状态: {json.dumps(session.summarize(), ensure_ascii=False)}")

# 第二轮：用户追问（Agent 必须"记得"上一轮在聊 ORD-1001）
print("\n【第 2 轮：用户追问（依赖状态）】")
q2 = "那帮我候补申请吧"
session.add_message("user", q2)
# Agent 从状态中恢复上下文
order_id = session.get_context("last_query_train")  # "ORD-1001"
date = session.get_context("last_query_date")        # "2026-06-20"
print(f"  用户: {q2}")
print(f"  [状态恢复] 上一轮查询的是 {order_id}，日期 {date}")
# Agent 现在知道候补申请的是 ORD-1001，而不是问"哪趟车"
session.record_tool_call("submit_waitlist",
                         {"order_id": order_id, "date": date},
                         {"waitlist_id": "WL001", "position": 5})
session.set_context("waitlist_active", True)
session.add_message("assistant",
                    f"已为您提交 {order_id} 候补申请订单，当前排队第 5 位。")
print(f"  [工具调用] submit_waitlist({order_id}) → 排队第 5 位")
print(f"  Agent: 已为您提交 {order_id} 的候补申请，排队第 5 位")

# 第三轮：用户问状态
print("\n【第 3 轮：用户询问候补申请状态】")
q3 = "我的候补申请排到第几了？"
session.add_message("user", q3)
# Agent 从状态中获取已有信息，无需再次调用工具
has_waitlist = session.get_context("waitlist_active")
print(f"  用户: {q3}")
print(f"  [状态查询] waitlist_active = {has_waitlist}")
if has_waitlist:
    last_call = [tc for tc in session.tool_calls
                 if tc["tool"] == "submit_waitlist"][-1]
    print(f"  [无需再次调用工具] 直接从状态获取：排队第 {last_call['result']['position']} 位")
    session.add_message("assistant",
                        f"您候补申请的是 {order_id}，当前排队第 {last_call['result']['position']} 位。")

# ========== 会话总结 ==========
print(f"\n{'─' * 60}")
print(f"【会话总结】")
print(json.dumps(session.summarize(), ensure_ascii=False, indent=2))
print(f"\n对话历史（{len(session.history)} 条消息）：")
for msg in session.history:
    print(f"  [{msg['role']}] {msg['content']}")

print("\n状态记忆设计要点：")
print("1. context 存关键信息（上次查询的服务编号、日期、结果），避免重复调用工具")
print("2. tool_calls 记录每次调用，方便溯源和审计")
print("3. 会话超时后状态应归档或过期清理，避免内存泄漏")
print("4. 多轮中若用户切换话题，应能检测并重置相关 context")