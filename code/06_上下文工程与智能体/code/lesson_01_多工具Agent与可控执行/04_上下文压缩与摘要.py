"""04_上下文压缩与摘要。

学习目标：压缩长对话历史，提取关键事实，减少 token 消耗。
焦点：摘要压缩、关键事实提取、压缩前后对比、分层压缩。

示例业务系统 场景：客服处理超长多轮对话，压缩历史后再给 LLM。
"""

import json
import re

print("04 上下文压缩与摘要 —— 长对话压缩实战")
print("=" * 72)

# ── 1. 模拟一段很长的客服对话（20 轮） ──
long_conversation = [
    {"role": "user", "content": "你好，查一下服务点A到服务点B的业务服务"},
    {"role": "assistant", "content": "您好，有多趟业务服务可选：G1、G3、G5、G7、ORD-1001、订单B 等。请问您想查哪天的？"},
    {"role": "user", "content": "明天的，标准服务"},
    {"role": "assistant", "content": "明天服务点A到服务点B标准服务有 G1(8:00)、G5(9:00)、ORD-1001(12:00)、订单B(14:00)。价格 553 元。"},
    {"role": "user", "content": "ORD-1001 怎么样，靠窗好选吗"},
    {"role": "assistant", "content": "ORD-1001 是复兴号，标准服务支持在线选座。当前库存 15 张，靠窗可选 A/F 座。"},
    {"role": "user", "content": "好的。顺便问一下我要退一张之前的票"},
    {"role": "assistant", "content": "请问要退哪趟车的票？麻烦提供服务编号和日期。"},
    {"role": "user", "content": "订单A，上周买的，6月18号的"},
    {"role": "assistant", "content": "订单A 6月18日距离现在超过 24 小时，退款按 5% 收取手续费。价格 553 元，手续费约 27.65 元。"},
    {"role": "user", "content": "手续费能免吗"},
    {"role": "assistant", "content": "退款费用按规定收取，无法免除。如果因行业原因停运，可全额退款。"},
    {"role": "user", "content": "行吧。那 ORD-1001 帮我创建订单标准服务靠窗"},
    {"role": "assistant", "content": "好的，已为您选择 ORD-1001 明天服务点A→服务点B 标准服务。请确认：\n- 服务编号：ORD-1001\n- 日期：6月20日 12:00\n- 区间：服务点A→服务点B\n- 服务类型：标准服务\n- 价格：553 元"},
    {"role": "user", "content": "等一下，我要靠窗的，帮我选 A 座"},
    {"role": "assistant", "content": "已选择 A 座（靠窗）。确认后将跳转支付。"},
    {"role": "user", "content": "支付用什么方式"},
    {"role": "assistant", "content": "支持微信、支付宝、银联和 示例业务系统 钱包余额支付。"},
    {"role": "user", "content": "用微信。对了，如果到时候想变更怎么办"},
    {"role": "assistant", "content": "变更需要在服务开始前办理。距服务开始 48 小时以上可免费变更一次，48 小时内需支付差额。"},
    {"role": "user", "content": "好的，先支付吧，变更的事后面再说"},
    {"role": "assistant", "content": "好的，已为您生成支付订单，请在 30 分钟内完成支付。"},
    {"role": "user", "content": "还有一件事，我要报销，怎么开电子发票"},
    {"role": "assistant", "content": "使用服务后 180 天内可在 业务 APP 的「订单详情」中申请电子发票，填写抬头后即可开具。"},
    {"role": "user", "content": "OK。还有个问题，我朋友也想买同一趟车，能坐一起吗"},
    {"role": "assistant", "content": "可以尝试在创建订单时选择相邻座位。如果系统分配后不在一起，可以通过「选座」功能手动调整。但无法保证绝对相邻。"},
    {"role": "user", "content": "好的，那就这样，谢谢"},
]

print(f"【原始对话】共 {len(long_conversation)} 条消息")

# 估算 token
def est_tokens(text):
    zh = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return int(zh / 1.5 + (len(text) - zh) / 4)

original_tokens = sum(est_tokens(m["content"]) for m in long_conversation)
print(f"   原始 token 估算: {original_tokens}")

# ── 2. 分层压缩策略 ──
print("\n【分层压缩策略】")
print("   Level 1: 完整保留 → 高 token 成本，信息无损")
print("   Level 2: 去重降噪 → 合并重复信息，减少冗余")
print("   Level 3: 关键事实提取 → 仅保留用户意图 + 业务结果")
print("   Level 4: 摘要压缩 → 一段摘要 + 最近 2 轮对话")

# ── 3. Level 2: 去重降噪 ──
print("\n── Level 2: 去重降噪 ──")
# 合并连续的 assistant 消息（保留最新）和合并同一话题的 user 消息
seen_facts = set()
deduped = []
for msg in long_conversation:
    # 简单去重：相同的 assistant 回复只保留一次
    content_key = msg["content"][:50]
    if msg["role"] == "assistant" and content_key in seen_facts:
        continue  # 跳过重复
    seen_facts.add(content_key)
    deduped.append(msg)

dedup_tokens = sum(est_tokens(m["content"]) for m in deduped)
print(f"   去重后: {len(deduped)} 条 ({dedup_tokens}t), 缩减 {original_tokens - dedup_tokens}t ({(1-dedup_tokens/original_tokens)*100:.0f}%)")

# ── 4. Level 3: 关键事实提取 ──
print("\n── Level 3: 关键事实提取 ──")

def extract_key_facts(conversation):
    """从对话中提取结构化事实"""
    facts = {
        "用户意图": [],
        "查询的服务编号": [],
        "业务结果": [],
        "用户偏好": [],
        "待办事项": [],
    }
    for msg in conversation:
        content = msg["content"]
        role = "用户" if msg["role"] == "user" else "客服"
        # 提取服务编号
        train_matches = re.findall(r'[GCD]\d+', content)
        for t in train_matches:
            if t not in str(facts["查询的服务编号"]):
                facts["查询的服务编号"].append(t)
        # 提取意图关键词
        if msg["role"] == "user":
            if any(w in content for w in ["查", "业务服务", "服务编号"]):
                facts["用户意图"].append("查服务编号")
            if any(w in content for w in ["退", "退款"]):
                facts["用户意图"].append("退款")
            if any(w in content for w in ["创建订单", "买", "订"]):
                facts["用户意图"].append("创建订单创建订单")
            if any(w in content for w in ["变更"]):
                facts["用户意图"].append("变更咨询")
            if any(w in content for w in ["发票", "报销"]):
                facts["用户意图"].append("发票咨询")
            if any(w in content for w in ["靠窗", "A座", "A 座"]):
                facts["用户偏好"].append("靠窗座位(A座)")
            if any(w in content for w in ["微信"]):
                facts["用户偏好"].append("微信支付")
        # 提取业务结果
        if msg["role"] == "assistant":
            if "已为您选择" in content or "已为您生成" in content:
                facts["业务结果"].append(content[:60])
    # 去重
    for key in facts:
        facts[key] = list(dict.fromkeys(facts[key]))
    return facts

facts = extract_key_facts(long_conversation)
print("   提取的关键事实:")
print(json.dumps(facts, ensure_ascii=False, indent=2))

# ── 5. Level 4: 摘要压缩 ──
print("\n── Level 4: 摘要压缩 ──")

def summarize_conversation(conversation, facts):
    """生成对话摘要"""
    user_msgs = [m["content"] for m in conversation if m["role"] == "user"]
    asst_msgs = [m["content"] for m in conversation if m["role"] == "assistant"]
    # 规则式摘要（实际生产用 LLM 生成）
    summary_parts = []
    if facts["用户意图"]:
        summary_parts.append(f"用户意图: {' → '.join(facts['用户意图'])}")
    if facts["查询的服务编号"]:
        summary_parts.append(f"涉及服务编号: {', '.join(facts['查询的服务编号'])}")
    if facts["用户偏好"]:
        summary_parts.append(f"偏好: {', '.join(facts['用户偏好'])}")
    if facts["业务结果"]:
        summary_parts.append(f"业务结果: {'; '.join(facts['业务结果'][:2])}")
    summary = " | ".join(summary_parts)
    return summary

summary = summarize_conversation(long_conversation, facts)

# 构建压缩后的上下文：摘要 + 最近 4 条消息
recent_msgs = long_conversation[-4:]
compressed_context = [
    {"role": "system", "content": f"[对话历史摘要] {summary}"},
    *recent_msgs,
]

compressed_tokens = sum(est_tokens(m["content"]) for m in compressed_context)
print(f"   摘要: {summary}")
print(f"   压缩后上下文:")
for m in compressed_context:
    role_label = "SYSTEM" if m["role"] == "system" else ("用户" if m["role"] == "user" else "客服")
    print(f"   [{role_label}] {m['content'][:80]}{'...' if len(m['content'])>80 else ''}")
print(f"   压缩后 token: {compressed_tokens}")

# ── 6. 压缩效果对比 ──
print(f"\n{'=' * 72}")
print("压缩效果对比:")
print(f"   ┌─────────────────────┬──────────┬──────────┐")
print(f"   │ 级别                │  消息数  │  Token   │")
print(f"   ├─────────────────────┼──────────┼──────────┤")
print(f"   │ Level 1 原始        │  {len(long_conversation):>6}  │  {original_tokens:>6}  │")
print(f"   │ Level 2 去重降噪    │  {len(deduped):>6}  │  {dedup_tokens:>6}  │")
print(f"   │ Level 3 事实提取    │    1*    │  {est_tokens(json.dumps(facts,ensure_ascii=False)):>6}  │")
print(f"   │ Level 4 摘要+最近   │  {len(compressed_context):>6}  │  {compressed_tokens:>6}  │")
print(f"   └─────────────────────┴──────────┴──────────┘")
print(f"   * Level 3 为结构化 JSON，非消息列表")
print(f"   最大压缩率: {(1 - compressed_tokens/original_tokens)*100:.0f}%")

# ── 7. 生产建议 ──
print(f"\n上下文压缩最佳实践:")
print("   1. 分层压缩：根据 token 压力自动选择压缩级别")
print("   2. 关键事实不可丢：用户意图、服务编号、订单号始终保留")
print("   3. 摘要 + 最近 N 轮：最实用的混合策略")
print("   4. 压缩时机：token 用量 > 70% 目标窗口时触发")
print("   5. 可逆性：压缩后的关键事实应能通过 LTM 检索完整还原")