"""04 本地 Qwen 多轮对话：手搓 tool calls。

本脚本使用 rag-tool-agent-course/open_models/Qwen3.5-0.8B。

DeepSeek 这类在线 API 有原生 tools 字段，模型会返回 tool_calls。
本地 transformers 直接 generate 时，没有这个 HTTP 协议层。

所以本脚本手搓一个最小 tool call 协议：
1. 把可用工具写进 system message。
2. 要求本地 Qwen 只输出 JSON。
3. Python 解析 JSON。
4. 如果 JSON 表示 tool_call，就执行本地工具。
5. 把工具结果再放回 messages。
6. 再让模型输出 final_answer。

运行方式：
    cd rag-tool-agent-course
    python code/08_专题_多轮对话机制/code/04_Qwen本地模型多轮对话.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("04 本地 Qwen 多轮对话：手搓 tool calls")
print("=" * 72)

MODEL_DIR = COURSE_ROOT / "open_models" / "Qwen3.5-0.8B"
if not MODEL_DIR.is_dir():
    raise SystemExit("未找到 open_models/Qwen3.5-0.8B，请先确认本地模型已经下载。")

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device != "cpu" else torch.float32

print(f"加载本地模型：{MODEL_DIR}")
print(f"运行设备：{device}")

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), torch_dtype=dtype).to(device).eval()


def search_policy(query: str) -> dict:
    """模拟政策知识库查询。真实项目里这里会接 RAG 或数据库。"""
    return {
        "tool": "search_policy",
        "query": query,
        "title": "候补申请规则",
        "content": "候补申请是排队兑现机制，不能保证一定成功。是否兑现取决于库存释放、退款变更、候补申请顺序和截止时间。",
        "source": "课堂示例政策库",
    }


def query_order_status(order_id: str, date: str = "2026-06-22") -> dict:
    """模拟服务编号库存查询。真实项目里这里会接业务系统接口。"""
    return {
        "tool": "query_order_status",
        "order_id": order_id,
        "date": date,
        "from": "服务点A",
        "to": "服务点B",
        "seat_type": "标准服务",
        "remaining": 0,
        "status": "无库存",
        "can_submit_waitlist": True,
        "risk_note": "候补申请可以提交，但不保证一定兑现。",
        "suggestion": "可提交候补申请，同时关注临近服务编号或其他服务类型。",
    }


tool_map = {
    "search_policy": search_policy,
    "query_order_status": query_order_status,
}

tools_schema = [
    {
        "name": "search_policy",
        "description": "查询候补申请、退款变更等政策资料。",
        "arguments": {
            "query": "string，要查询的政策问题，例如：候补申请是否保证成功"
        },
    },
    {
        "name": "query_order_status",
        "description": "查询指定服务编号在指定日期的示例库存状态。",
        "arguments": {
            "order_id": "string，服务编号号，例如 ORD-1001",
            "date": "string，使用服务日期，格式 YYYY-MM-DD；不知道时使用 2026-06-22",
        },
    },
]

json_protocol = {
    "tool_call": {
        "action": "tool_call",
        "tool_name": "query_order_status",
        "arguments": {"order_id": "ORD-1001", "date": "2026-06-22"},
    },
    "final_answer": {
        "action": "final_answer",
        "answer": "根据工具结果组织给用户看的最终回答。",
    },
}

system_text = f"""
你是客服辅助助手。你不能直接编造服务编号状态或订单服务规则。

可用工具如下：
{json.dumps(tools_schema, ensure_ascii=False, indent=2)}

你每次只能输出一个 JSON 对象，不能输出 Markdown，不能输出解释文字。

如果需要调用工具，输出：
{json.dumps(json_protocol["tool_call"], ensure_ascii=False, indent=2)}

如果已经拿到工具结果，可以回答用户，输出：
{json.dumps(json_protocol["final_answer"], ensure_ascii=False, indent=2)}

要求：
1. action 只能是 tool_call 或 final_answer。
2. tool_call 时必须填写 tool_name 和 arguments。
3. final_answer 时必须填写 answer。
4. 用户问题里只要出现服务编号、库存、没票、标准服务、日期，必须先调用 query_order_status。
5. 没有看到工具执行结果前，不能输出 final_answer。
6. 不要输出 <think> 内容。
7. 不要输出 JSON 之外的文字。
""".strip()

messages = [
    {"role": "system", "content": system_text}
]


def generate_json_from_qwen() -> str:
    """把当前 messages 渲染成 chat prompt，然后让本地模型生成 JSON 文本。"""
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("\n模型实际看到的 prompt 片段：")
    print(rendered_prompt[:1200])
    if len(rendered_prompt) > 1200:
        print("...（后面省略）")

    inputs = tokenizer(rendered_prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=260,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def extract_json(text: str) -> dict:
    """从模型输出中提取 JSON。

    小模型有时会包一层 ```json，或者在 JSON 前后多输出几个字。
    这里用 JSONDecoder.raw_decode 从第一个 { 开始解析。
    """
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    start = cleaned.find("{")
    if start == -1:
        raise ValueError("模型输出里没有找到 JSON 对象。")

    candidate = cleaned[start:]

    try:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(candidate)
        if not isinstance(obj, dict):
            raise ValueError("解析结果不是 JSON 对象。")
        return obj
    except Exception:
        # 本地小模型有时会少输出最后一个 }。
        # 如果只是末尾缺括号，补一个再试。
        if candidate.count("{") > candidate.count("}"):
            repaired = candidate + "}" * (candidate.count("{") - candidate.count("}"))
            obj = json.loads(repaired)
            if isinstance(obj, dict):
                return obj

        # 有时 answer 字段里会出现未转义换行，严格 JSON 解析会失败。
        # 课堂里保留这个容错，让脚本能继续演示 tool call 主线。
        if '"action"' in candidate and '"final_answer"' in candidate and '"answer"' in candidate:
            marker = '"answer"'
            pos = candidate.find(marker)
            colon = candidate.find(":", pos)
            answer_text = candidate[colon + 1:].strip()
            answer_text = answer_text.strip().strip("{}").strip().strip('"')
            return {
                "action": "final_answer",
                "answer": answer_text,
            }

        raise


def question_needs_tool(question: str) -> bool:
    """判断这类问题是否应该先查工具。"""
    keywords = ["ORD-1001", "服务编号", "库存", "没票", "标准服务", "日期", "候补申请"]
    return any(word in question for word in keywords)


def repair_with_simple_rule(question: str, raw_text: str) -> dict:
    """课堂兜底：本地小模型如果没有稳定输出 JSON，就用简单规则保证流程能跑完。

    这不是生产做法，只是为了让课堂演示不中断。
    生产系统应该让更强模型输出结构化结果，或加重试、校验和人工兜底。
    """
    print("\nJSON 解析失败，进入课堂兜底。原始输出：")
    print(raw_text)

    if "ORD-1001" in question or "服务编号" in question or "标准服务" in question:
        return {
            "action": "tool_call",
            "tool_name": "query_order_status",
            "arguments": {"order_id": "ORD-1001", "date": "2026-06-22"},
        }

    if "候补申请" in question or "规则" in question:
        return {
            "action": "tool_call",
            "tool_name": "search_policy",
            "arguments": {"query": question},
        }

    return {
        "action": "final_answer",
        "answer": "这个问题不需要调用工具，可以直接回答：请提供更具体的服务编号、日期或政策问题。",
    }


def run_user_turn(question: str) -> None:
    print(f"\n{'=' * 24} 用户新问题 {'=' * 24}")
    print("用户：", question)

    messages.append({"role": "user", "content": question + " /no_think"})

    sub_turn = 1
    while True:
        print(f"\n--- 本地模型子步骤 {sub_turn}：生成 JSON 指令 ---")

        raw_text = generate_json_from_qwen()
        print("\n模型原始输出：")
        print(raw_text)

        try:
            parsed = extract_json(raw_text)
        except Exception:
            parsed = repair_with_simple_rule(question, raw_text)

        if (
            sub_turn == 1
            and parsed.get("action") == "final_answer"
            and question_needs_tool(question)
        ):
            print("\n校验发现：这类问题需要先查工具，不能直接 final_answer。")
            parsed = {
                "action": "tool_call",
                "tool_name": "query_order_status",
                "arguments": {"order_id": "ORD-1001", "date": "2026-06-22"},
            }

        print("\n解析并校验后的 JSON：")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))

        # 把模型的 JSON 指令也放回历史，下一轮模型能看到自己刚才做过什么。
        messages.append({
            "role": "assistant",
            "content": json.dumps(parsed, ensure_ascii=False),
        })

        action = parsed.get("action")

        if action == "final_answer":
            print("\n模型最终回答：")
            print(parsed.get("answer", ""))
            print("\n当前完整 messages：")
            print(json.dumps(messages, ensure_ascii=False, indent=2))
            break

        if action != "tool_call":
            print("\n未知 action，结束本轮。")
            break

        tool_name = parsed.get("tool_name")
        arguments = parsed.get("arguments") or {}

        print("\n--- Python 本地执行工具 ---")
        print("工具名：", tool_name)
        print("参数：", json.dumps(arguments, ensure_ascii=False))

        if tool_name not in tool_map:
            tool_result = {"error": f"未知工具：{tool_name}"}
        else:
            try:
                tool_result = tool_map[tool_name](**arguments)
            except TypeError as exc:
                tool_result = {"error": f"工具参数错误：{exc}"}

        print("工具结果：")
        print(json.dumps(tool_result, ensure_ascii=False, indent=2))

        # 本地 transformers 的 chat_template 不一定支持 role=tool。
        # 所以这里把工具结果作为普通 user 消息放回去。
        # 内容里明确写“工具执行结果”，模型仍然能基于它继续生成 final_answer。
        messages.append({
            "role": "user",
            "content": (
                "工具执行结果如下。请基于工具结果回答用户，只输出 final_answer JSON：\n"
                + json.dumps(tool_result, ensure_ascii=False, indent=2)
                + "\n注意：如果 can_submit_waitlist 为 true，应表达为“可以提交候补申请，但不保证兑现”，不要说“候补申请机制无法覆盖”。"
                + "\n/no_think"
            ),
        })

        sub_turn += 1
        if sub_turn > 4:
            print("\n达到最大工具循环次数，结束本轮。")
            break


run_user_turn("ORD-1001 在 2026-06-22 标准服务没票了，候补申请一定能成功吗？请给一个稳妥答复。")
run_user_turn("如果用户很着急，下一句应该怎么补充建议？")

print("\n结论")
print("1. 本地 transformers 没有原生 tool_calls 字段，需要自己约定 JSON 协议。")
print("2. 模型负责输出 JSON 指令，Python 负责解析 JSON 和执行工具。")
print("3. 工具结果要重新放回 messages，模型才能基于结果继续回答。")
print("4. 这和 DeepSeek 等在线 API 的 tool call 主线一致，只是协议层由我们手搓。")
