"""01_上下文窗口管理。

学习目标：理解 LLM 上下文窗口的 token 计数、截断策略，以及滑动窗口机制。
焦点：token 计数、first-k/last-k/summary 截断、滑动窗口。

示例业务系统 场景：多轮客服对话的上下文管理。
"""

import json

print("01 上下文窗口管理 —— Token 计数与截断策略")
print("=" * 72)

# ── 1. 模拟一场较长的 示例业务系统 客服对话 ──
conversation = [
    {"role": "user", "content": "你好，我想查一下服务点A到服务点B的业务服务"},
    {"role": "assistant", "content": "您好！服务点A到服务点B每天有多趟业务服务，如 G1、G3、G5、ORD-1001 等，请问您想查哪一趟？"},
    {"role": "user", "content": "ORD-1001 标准服务还有库存吗"},
    {"role": "assistant", "content": "ORD-1001 当前标准服务已无库存，高级服务还有少量。需要帮您查别的服务编号吗？"},
    {"role": "user", "content": "那 订单B 呢"},
    {"role": "assistant", "content": "订单B 标准服务还有 12 张，价格 553 元。需要帮您创建订单吗？"},
    {"role": "user", "content": "我再想想。对了，我之前买的 订单A 能退款吗？"},
    {"role": "assistant", "content": "退款需要在服务开始前办理。请问您 订单A 的服务开始的日期和您的订单类型是什么？"},
    {"role": "user", "content": "明天的，标准服务，服务开始前 3 小时还能退吗"},
    {"role": "assistant", "content": "服务开始前 2-24 小时退款，收取 10% 手续费。您距离服务开始 3 小时，可以办理。请登录 示例业务系统 订单页操作。"},
    {"role": "user", "content": "手续费具体多少钱"},
    {"role": "assistant", "content": "手续费按票面价格计算。订单A 标准服务价格 553 元，手续费约 55.3 元，具体以订单页显示为准。"},
    {"role": "user", "content": "那我先不退款了，帮我把 订单B 的标准服务创建订单吧"},
    {"role": "assistant", "content": "好的，已为您选择 订单B 标准服务。请确认：\n- 服务编号：订单B\n- 日期：2026年6月20日\n- 服务类型：标准服务\n- 价格：553 元\n确认后将跳转支付页面。"},
    {"role": "user", "content": "还有一个问题，候补申请是什么意思？能保证买到吗"},
    {"role": "assistant", "content": "候补申请是一种排队等待业务受理的机制。当有人退款时，系统按顺序兑现。但不能保证一定成功，最终以 官方页面通知为准。"},
    {"role": "user", "content": "好的明白了，谢谢"},
]

print("\n【对话总览】共 {} 轮对话".format(len(conversation) // 2))
for i, msg in enumerate(conversation):
    tag = "👤" if msg["role"] == "user" else "🤖"
    print(f"  {tag} [{i+1}] {msg['content'][:60]}{'...' if len(msg['content'])>60 else ''}")

# ── 2. Token 估算（简易版：中文约 2 字符 = 1 token，英文约 4 字符 = 1 token） ──
print("\n【Token 估算】")

def estimate_tokens(text):
    """简易 token 估算：中文按 1.5 字/token，英文/数字按 4 字/token"""
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)

total_tokens = 0
token_per_msg = []
for msg in conversation:
    t = estimate_tokens(msg["content"])
    token_per_msg.append(t)
    total_tokens += t

print(f"   对话总 token 估算: {total_tokens}")
print(f"   系统提示预留: 200 tokens")
print(f"   可用窗口假设: 4096 tokens")
available_for_history = 4096 - 200 - estimate_tokens(conversation[-1]["content"])
print(f"   留给历史消息的空间: {available_for_history} tokens")

# ── 3. 截断策略演示 ──
print("\n【截断策略对比】假设用户刚刚问了最后一个问题，我们需要给 LLM 构建上下文：")

# 策略 A：first-k（保留最早的消息）
def first_k(msgs, k):
    return msgs[:k]

# 策略 B：last-k（保留最近的消息）
def last_k(msgs, k):
    return msgs[-k:]

# 策略 C：滑动窗口（保持一定 token 预算）
def sliding_window(msgs, token_budget):
    result = []
    token_count = 0
    for msg in reversed(msgs):
        t = estimate_tokens(msg["content"])
        if token_count + t > token_budget:
            break
        result.insert(0, msg)
        token_count += t
    return result

print("\n   ┌────────────────────┬──────────────┬─────────────────────────────────┐")
print("   │ 策略               │ 截取条数     │ 内容示例                        │")
print("   ├────────────────────┼──────────────┼─────────────────────────────────┤")

for strategy_name, strategy_fn, arg in [
    ("first-k (k=4)", first_k, 4),
    ("last-k (k=6)", last_k, 6),
    ("sliding_window (budget=800)", lambda msgs: sliding_window(msgs, 800), None),
]:
    subset = strategy_fn(conversation) if arg is None else strategy_fn(conversation, arg)
    tokens_in = sum(estimate_tokens(m["content"]) for m in subset)
    preview = " → ".join(m["content"][:20] for m in subset[:3])
    print(f"   │ {strategy_name:<18} │ {len(subset):>4} 条 ({tokens_in:>3}t) │ {preview:<47} │")

print("   └────────────────────┴──────────────┴─────────────────────────────────┘")

# ── 4. 实践：对当前对话做滑动窗口 ──
print("\n【滑动窗口实践】token_budget=500，构建给 LLM 的上下文：")

# 最新一条是用户的提问
current_query = conversation[-1]["content"]
window_msgs = sliding_window(conversation[:-1], 500)  # 保留 500 tokens 给历史
window_tokens = sum(estimate_tokens(m["content"]) for m in window_msgs)

print(f"   窗口内消息: {len(window_msgs)} 条（共 {window_tokens} tokens）")
for m in window_msgs:
    role_label = "用户" if m["role"] == "user" else "客服"
    print(f"   [{role_label}] {m['content'][:70]}{'...' if len(m['content'])>70 else ''}")

# ── 5. 上下文溢出检测 ──
print("\n【溢出检测与降级策略】")

def check_overflow(history_tokens, system_tokens, model_limit):
    usage = history_tokens + system_tokens
    pct = usage / model_limit * 100
    if pct > 90:
        level = "RED"
        action = "立即截断至最近 2 轮，或触发摘要压缩"
    elif pct > 70:
        level = "YELLOW"
        action = "启动滑动窗口，仅保留最近 500 tokens"
    else:
        level = "GREEN"
        action = "正常使用，无需要操作"
    return {"usage_tokens": usage, "usage_pct": f"{pct:.1f}%", "alert_level": level, "action": action}

test_scenarios = [
    (300, 200, 4096, "正常对话"),
    (3000, 200, 4096, "长对话"),
    (3800, 200, 4096, "超长对话"),
]
for hist_tok, sys_tok, limit, label in test_scenarios:
    report = check_overflow(hist_tok, sys_tok, limit)
    print(f"   [{label}] 历史={hist_tok}t 系统={sys_tok}t 限额={limit}t → {report['alert_level']}: {report['action']}")

# ── 6. 总结 ──
print(f"\n{'=' * 72}")
print("上下文窗口管理要点:")
print("  1. 始终为系统提示和当前问题预留 token 空间")
print("  2. first-k 适合固定开场白的场景，last-k 适合持续对话")
print("  3. 滑动窗口是最通用的策略，建议生产环境优先使用")
print("  4. 超过 70% 阈值时启动截断，超过 90% 时触发摘要/压缩")
print("  5. 截断后丢失的关键信息可通过长期记忆（向量检索）补回")