"""06_Agent_harness_与未来软件形态 — Harness 模式、Agent 组合、未来趋势。

学习目标：理解 Agent harness（运行外壳）的概念以及 Agent 如何组合成更大的系统。
"""

import json

print("06_Agent_harness_与未来软件形态")
print("=" * 72)
print("Harness = 给 Agent 套一个'运行外壳'：管理输入输出、工具、日志、权限、评测。")
print()

# ========== Agent Harness 的结构 ==========
print("【Agent Harness 的六层结构】")
harness_layers = {
    "L1 输入层": "接收用户输入，做 guardrail 检查，格式化后传给 Agent",
    "L2 上下文层": "管理对话历史、会话状态、用户画像",
    "L3 工具层": "工具注册表 + 执行器 + 超时 + 重试 + 限流",
    "L4 Agent 核心": "ReAct 循环：Thought → Action → Observation → Answer",
    "L5 输出层": "结果格式化 + output guardrail + 脱敏",
    "L6 观测层": "日志、监控、评测指标、成本追踪",
}
for layer, desc in harness_layers.items():
    print(f"  {layer}：{desc}")

# ========== Harness 代码骨架 ==========
print("\n【Harness 代码骨架（以 示例业务系统 场景为例）】")

class AgentHarness:
    """Agent 的运行外壳，统一管理 Agent 的所有生命周期。"""

    def __init__(self, agent_name: str, tools: dict):
        self.agent_name = agent_name
        self.tools = tools
        self.sessions = {}
        self.metrics = {"total_calls": 0, "total_tokens": 0, "errors": 0}
        self.output_guard_enabled = True

    def create_session(self, user_id: str) -> str:
        session_id = f"{user_id}_{len(self.sessions)}"
        self.sessions[session_id] = {
            "user_id": user_id,
            "history": [],
            "context": {},
            "created_at": "2026-06-19",
        }
        return session_id

    def check_input(self, user_input: str) -> bool:
        """输入 guardrail。"""
        blocked = ["忽略规则", "忘记指令", "越狱"]
        return not any(b in user_input for b in blocked)

    def run(self, session_id: str, user_input: str) -> dict:
        """Agent 执行入口。"""
        if not self.check_input(user_input):
            return {"answer": "输入不合规，已拦截。", "blocked": True}

        session = self.sessions[session_id]
        session["history"].append({"role": "user", "content": user_input})

        # 简化的 Agent 循环
        tool_result = self._maybe_call_tool(user_input)
        answer = self._generate_answer(user_input, tool_result)

        if self.output_guard_enabled:
            answer = self._sanitize_output(answer)

        session["history"].append({"role": "assistant", "content": answer})
        self.metrics["total_calls"] += 1

        return {"answer": answer, "session_id": session_id,
                "metrics": self.metrics}

    def _maybe_call_tool(self, user_input: str) -> dict:
        """判断是否需要调用工具。"""
        if "票" in user_input or "G" in user_input:
            return {"tool": "query_order_statuss", "result": {"ORD-1001": "无库存", "ORD-1002": "有库存"}}
        if "站" in user_input or "编码" in user_input:
            return {"tool": "service_point_code", "result": {"服务点A": "SP-A"}}
        return {}

    def _generate_answer(self, user_input: str, tool_result: dict) -> str:
        if not tool_result:
            return "请描述您的问题。"
        tool_name = tool_result["tool"]
        if tool_name == "query_order_statuss":
            return (f"查询结果：ORD-1001 无库存，ORD-1002 有库存。"
                    f"最终以 官方页面为准。")
        if tool_name == "service_point_code":
            return f"服务点编码：{tool_result['result']}。"
        return "已处理。"

    def _sanitize_output(self, answer: str) -> str:
        """输出脱敏和合规检查。"""
        # 确保有免责声明
        if "示例业务系统" not in answer and len(answer) > 20:
            answer += "（以 官方信息为准）"
        return answer

# ========== 运行演示 ==========
harness = AgentHarness("示例业务系统_Agent", {
    "query_order_statuss": "查库存",
    "service_point_code": "查服务点编码",
    "create_order": "创建订单",
})

sid = harness.create_session("user_001")

print("【单 Agent 运行】")
r1 = harness.run(sid, "查 ORD-1001 明天有库存吗")
print(f"  Q: 查 ORD-1001 明天有库存吗")
print(f"  A: {r1['answer']}")

r2 = harness.run(sid, "服务点A的编码")
print(f"  Q: 服务点A的编码")
print(f"  A: {r2['answer']}")

print(f"\n  运行指标: {json.dumps(harness.metrics, ensure_ascii=False)}")

# ========== Multi-Agent 组合 ==========
print("\n【Multi-Agent 组合：把 Agent 当积木搭更大的系统】")

class MultiAgentSystem:
    """多个 Agent 编排成一个更大的系统。"""

    def __init__(self):
        self.service_agent = AgentHarness("service_agent", {"query": "查库存", "order": "创建订单"})
        self.refund_agent = AgentHarness("refund_agent", {"calc": "算退款", "apply": "申请退款"})
        self.router_agent = AgentHarness("router_agent", {})

    def dispatch(self, user_input: str, user_id: str) -> str:
        """路由到对应的子 Agent。"""
        if any(w in user_input for w in ["退款", "退款", "变更"]):
            return self.refund_agent.run(
                self.refund_agent.create_session(user_id), user_input)["answer"]
        elif any(w in user_input for w in ["票", "服务编号", "G", "D"]):
            return self.service_agent.run(
                self.service_agent.create_session(user_id), user_input)["answer"]
        else:
            return "请描述与 示例业务系统 相关的问题。"

system = MultiAgentSystem()
print(f"  创建订单请求: {system.dispatch('帮我查 ORD-1001 明天有库存吗', 'user_001')}")
print(f"  退款请求: {system.dispatch('我要退款，能退多少钱', 'user_001')}")

# ========== 未来软件形态 ==========
print("\n【Agent 驱动的未来软件形态】")
future_trends = [
    ("从 GUI 到 CUI", "传统软件 = 按钮+表单；Agent 软件 = 对话+工具调用"),
    ("从单体到组合", "一个 Agent 做一件事，多个 Agent 通过编排组成复杂系统"),
    ("从硬编码到动态规划", "Planner 根据任务自动决定调用哪些工具、按什么顺序"),
    ("从人工操作到自主执行", "不只是'回答问题'，而是'帮你把事办了'（创建订单、退款、变更）"),
    ("从封闭到生态", "工具可以来自不同的服务商，Agent 就像一个浏览器调用各种 API"),
]
for title, desc in future_trends:
    print(f"  {title}：{desc}")

print(f"\nHarness + Multi-Agent 总结：")
print(f"1. Harness 是 Agent 的基础设施：输入/输出/工具/日志/安全")
print(f"2. 多 Agent 协作 = 把复杂任务拆给专业 Agent 各司其职")
print(f"3. 未来软件 = AI 理解需求 → 规划步骤 → 调工具 → 完成操作 → 反馈结果")