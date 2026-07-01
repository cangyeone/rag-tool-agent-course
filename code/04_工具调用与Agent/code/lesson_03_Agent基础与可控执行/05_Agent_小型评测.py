"""05_Agent_小型评测 — 评估 Agent 的成功率、工具选择准确性、回答质量。

学习目标：建立评测体系，用指标衡量 Agent 质量，而不是凭感觉。
"""


import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("05_Agent_小型评测")
print("=" * 72)
print("没有评测的 Agent = 不知道它到底行不行。")
print("评测三维度：工具选择是否正确 + 回答是否有依据 + 是否满足业务约束")
print()


def _find_qwen_path() -> str:
    candidate = COURSE_ROOT / "open_models" / "Qwen3.5-0.8B"
    if candidate.is_dir():
        return str(candidate)
    raise FileNotFoundError("未找到 open_models/Qwen3.5-0.8B。请确认已经从 rag-tool-agent-course 根目录运行，并且模型已下载。")


# ========== 加载本地 Qwen 模型 ==========
QWEN_PATH = _find_qwen_path()
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device != "cpu" else torch.float32
print(f"加载 Qwen 模型: {QWEN_PATH} (device={device})")
qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH)
qwen_model = AutoModelForCausalLM.from_pretrained(QWEN_PATH, torch_dtype=dtype).to(device).eval()


# ========== 真实 Agent（使用本地 Qwen） ==========
class QwenAgent:
    """基于本地 Qwen3.5-0.8B 的简单 Agent。"""

    AGENT_PROMPT = (
        "你是 示例业务系统 在线客服 Agent。根据用户问题：\n"
        "1. 先判断需要调用哪个工具（query_tickets / calc_refund / query_station / order_ticket）\n"
        "2. 提取工具所需参数\n"
        "3. 生成回答，回答中必须提及官方渠道，不能做虚假承诺。\n\n"
        "可用的工具：\n"
        "- query_tickets: 查询订单编号库存，参数 train_no, date\n"
        "- calc_refund: 计算退款手续费，参数 ticket_price, hours_before\n"
        "- query_station: 查询服务点编码，参数 station_name\n"
        "- order_ticket: 提交下单订单，参数 train_no, seat\n\n"
        "请严格按以下 JSON 格式输出（只输出 JSON）：\n"
        '{"tool_called": "工具名或null", "tool_args": {...}, "answer": "回答文本"}'
    )

    def process(self, question: str) -> dict:
        messages = [
            {"role": "system", "content": self.AGENT_PROMPT},
            {"role": "user", "content": question},
        ]
        text = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = qwen_tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = qwen_model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=True,
                                             pad_token_id=qwen_tokenizer.eos_token_id)
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        response = qwen_tokenizer.decode(generated, skip_special_tokens=True).strip()

        try:
            result = json.loads(response)
            if "tool_called" not in result:
                result["tool_called"] = None
            if "answer" not in result:
                result["answer"] = response
            return result
        except json.JSONDecodeError:
            return {"tool_called": None, "answer": response}

# ========== 评测维度 ==========
class EvalMetrics:
    @staticmethod
    def tool_selection_accuracy(agent_result: dict, expected_tool: str) -> float:
        """工具选择是否准确。"""
        return 1.0 if agent_result["tool_called"] == expected_tool else 0.0

    @staticmethod
    def answer_has_evidence(answer: str) -> bool:
        """回答是否有依据（不含臆测）。"""
        evidence_indicators = ["根据", "查询结果", "当前", "目前", "示例业务系统", "官方"]
        return any(ind in answer for ind in evidence_indicators)

    @staticmethod
    def answer_has_disclaimer(answer: str) -> bool:
        """回答是否包含免责声明。"""
        return "以" in answer and "为准" in answer

    @staticmethod
    def no_false_promise(answer: str) -> bool:
        """回答不含虚假承诺。"""
        false_promises = ["保证", "100%", "肯定能", "绝对", "包你"]
        return not any(fp in answer for fp in false_promises)

# ========== 评测用例 ==========
eval_cases = [
    {"question": "G107 没票了还能怎么办？",
     "expected_tool": "query_tickets",
     "checks": ["answer_has_evidence", "answer_has_disclaimer", "no_false_promise"]},
    {"question": "退款要扣多少钱？",
     "expected_tool": "calc_refund",
     "checks": ["answer_has_evidence", "no_false_promise"]},
    {"question": "北京南的服务点编码是多少？",
     "expected_tool": "query_station",
     "checks": ["answer_has_evidence", "no_false_promise"]},
    {"question": "帮我订一张明天的 G107 标准服务",
     "expected_tool": "order_ticket",
     "checks": ["answer_has_evidence", "answer_has_disclaimer", "no_false_promise"]},
]

# ========== 执行评测 ==========
agent = QwenAgent()
metrics = EvalMetrics()
results = []

print("【评测执行】")
for case in eval_cases:
    q = case["question"]
    agent_result = agent.process(q)

    eval_result = {
        "question": q,
        "expected_tool": case["expected_tool"],
        "actual_tool": agent_result["tool_called"],
        "answer": agent_result["answer"][:50] + "...",
    }

    # 1. 工具选择
    eval_result["tool_accuracy"] = metrics.tool_selection_accuracy(
        agent_result, case["expected_tool"])

    # 2. 各检查项
    check_results = {}
    for check_name in case["checks"]:
        method = getattr(metrics, check_name)
        check_results[check_name] = method(agent_result["answer"])
    eval_result["checks"] = check_results

    # 3. 综合评分
    scores = [eval_result["tool_accuracy"]] + list(check_results.values())
    eval_result["overall"] = sum(scores) / len(scores)

    results.append(eval_result)

# ========== 输出报告 ==========
for r in results:
    status = "✓" if r["overall"] >= 0.8 else "⚠" if r["overall"] >= 0.5 else "✗"
    print(f"\n  [{status}] {r['question']}")
    print(f"    工具: 期望={r['expected_tool']}, 实际={r['actual_tool']} "
          f"({'✓' if r['tool_accuracy'] else '✗'})")
    for ck, cv in r["checks"].items():
        print(f"    {ck}: {'✓' if cv else '✗'}")
    print(f"    综合评分: {r['overall']:.2f}")

# ========== 汇总统计 ==========
total = len(results)
tool_acc = sum(r["tool_accuracy"] for r in results) / total
avg_overall = sum(r["overall"] for r in results) / total

print(f"\n{'─' * 40}")
print(f"【汇总统计】")
print(f"  总用例: {total}")
print(f"  工具选择准确率: {tool_acc:.0%}")
print(f"  综合均分: {avg_overall:.2f}")
print(f"  通过率(≥0.8): {sum(1 for r in results if r['overall'] >= 0.8)}/{total}")

print("\n评测体系建议：")
print("1. 工具选择准确性 → 反映 Agent '知道该调哪个工具'的能力")
print("2. 回答质量（有依据/不瞎编/有免责）→ 反映生成质量")
print("3. 综合评分 = 工具选择 + 多项回答质量检查的加权平均")