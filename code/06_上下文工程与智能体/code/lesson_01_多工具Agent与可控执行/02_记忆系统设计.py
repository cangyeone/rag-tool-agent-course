"""02_记忆系统设计。

学习目标：理解 Agent 的三层记忆体系 —— 短期记忆、长期记忆、工作记忆。
焦点：消息列表（短期）、向量检索（长期）、scratchpad（工作记忆）。

示例业务系统 场景：跨会话的用户偏好和历史订单记忆。
"""

import json
import hashlib
from collections import OrderedDict

print("02 记忆系统设计 —— 三层记忆体系")
print("=" * 72)

# ── 1. 三层记忆架构 ──
print("\n【三层记忆架构】")
print("   短期记忆(STM):  当前会话的消息列表，存在于上下文窗口中，会话结束即销毁")
print("   长期记忆(LTM):  向量数据库存储的用户画像、历史行为，跨会话持久化")
print("   工作记忆(WM):  当前任务执行中的中间变量、工具结果、草稿，类似 scratchpad")
print()

# ── 2. 短期记忆：当前会话消息列表 ──
print("一、短期记忆（STM）—— 当前会话消息")
stm = [
    {"role": "user",   "content": "帮我查北京西到服务点C的业务服务", "timestamp": "10:00:01"},
    {"role": "assistant", "content": "好的，请问您想查哪一天的服务编号？", "timestamp": "10:00:02"},
    {"role": "user",   "content": "明天，标准服务", "timestamp": "10:00:05"},
    {"role": "assistant", "content": "明天北京西到服务点C的标准服务有 G79、G81、G65 等服务编号，价格约 862 元。需要帮您筛选吗？", "timestamp": "10:00:06"},
    {"role": "user",   "content": "我比较在意靠窗座位和充电插座，这些车哪个好？", "timestamp": "10:01:00"},
]
print("   当前会话消息数:", len(stm))
for i, msg in enumerate(stm):
    label = "用户" if msg["role"] == "user" else "客服"
    print(f"   [{label} @{msg['timestamp']}] {msg['content'][:60]}{'...' if len(msg['content'])>60 else ''}")

# STM 容量管理
stm_max = 20  # 最多保留 20 条消息
if len(stm) > stm_max:
    overflow = len(stm) - stm_max
    stm = stm[-stm_max:]  # 丢弃最早的
    print(f"   ⚠️ STM 溢出，丢弃了 {overflow} 条最早的消息")
else:
    print(f"   STM 用量: {len(stm)}/{stm_max}，安全")

# ── 3. 长期记忆：用户画像与历史行为（向量检索模拟） ──
print("\n二、长期记忆（LTM）—— 向量检索模拟")

# 模拟向量存储：以用户 ID 为 key
user_profile_db = {
    "user_88321": {
        "基本信息": {"会员等级": "白金会员", "常用起点": ["服务点A", "北京西"], "偏好服务类型": "标准服务"},
        "偏好画像": {"靠窗": 0.9, "充电插座": 0.8, "安静车厢": 0.6, "快速": 0.7},
        "历史订单": [
            {"服务编号": "订单A", "日期": "2026-06-10", "区间": "服务点A→服务点B", "服务类型": "标准服务", "满意度": 5},
            {"服务编号": "G79",  "日期": "2026-06-15", "区间": "北京西→服务点C", "服务类型": "标准服务", "满意度": 4},
            {"服务编号": "ORD-2001", "日期": "2026-05-20", "区间": "服务点A→服务点B", "服务类型": "标准服务", "满意度": 3},
        ],
        "last_session_summary": "用户上次咨询了退款规则，对靠窗座位和车内设施比较关注。",
    }
}

# 模拟向量检索
def retrieve_from_ltm(user_id, query_embedding_simulated, top_k=3):
    """用规则模拟向量检索：从用户画像中找相关记忆"""
    profile = user_profile_db.get(user_id, {})
    history = profile.get("历史订单", [])
    # 模拟：按相似度排序（实际用 embedding cosine similarity）
    scored = []
    for order in history:
        score = 0.0
        if "北京西" in query_embedding_simulated and "北京西" in order.get("区间", ""):
            score += 0.3
        if "广州" in query_embedding_simulated and "广州" in order.get("区间", ""):
            score += 0.4
        if "标准服务" in query_embedding_simulated and order.get("服务类型") == "标准服务":
            score += 0.2
        if order.get("满意度", 0) >= 4:
            score += 0.1
        if score > 0:
            scored.append((score, order))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]

# 用户当前查询
current_query = "查北京西到服务点C的业务服务标准服务，要舒服的"
print(f"   当前查询: {current_query}")
print(f"   用户 ID: user_88321")
print(f"   用户画像: {json.dumps(user_profile_db['user_88321']['基本信息'], ensure_ascii=False)}")

related_memories = retrieve_from_ltm("user_88321", current_query, top_k=3)
print(f"   检索到 {len(related_memories)} 条相关历史:")
for mem in related_memories:
    print(f"   ├─ {mem['日期']} {mem['服务编号']} {mem['区间']} 满意度{mem['满意度']}★")

# ── 4. 工作记忆：Scratchpad ──
print("\n三、工作记忆（WM）—— Scratchpad")

wm = OrderedDict()
wm["task"] = "帮用户推荐北京西→服务点C的业务服务标准服务"
wm["intent"] = "服务编号推荐"
wm["extracted_slots"] = {"from": "北京西", "to": "服务点C", "seat": "标准服务", "preference": "舒适"}
wm["user_profile"] = {"会员": "白金", "偏好": ["靠窗", "充电", "安静"]}
wm["tool_query_train_result"] = [
    {"train": "G79",  "seat_available": 15, "has_window_seat": True, "has_outlet": True,  "duration": "8h12m"},
    {"train": "G81",  "seat_available": 8,  "has_window_seat": True, "has_outlet": False, "duration": "8h30m"},
    {"train": "G65",  "seat_available": 3,  "has_window_seat": False,"has_outlet": True,  "duration": "7h55m"},
]
wm["tool_policy_result"] = {"安静车厢": "部分业务服务提供静音车厢，需创建订单时勾选"}
wm["scoring"] = {}  # 待计算
wm["draft_answer"] = ""  # 待生成

print(f"   工作记忆中保存了 {len(wm)} 个键:")
for k in wm:
    v = wm[k]
    if isinstance(v, list):
        print(f"   ├─ {k}: [{len(v)} 项]")
    elif isinstance(v, dict):
        print(f"   ├─ {k}: [{len(v)} 个字段]")
    else:
        print(f"   ├─ {k}: {str(v)[:50]}")

# ── 5. 三记忆协同：工作记忆利用 STM 和 LTM 的信息做决策 ──
print("\n四、三记忆协同评分")

# 基于 LTM 的偏好和 WM 的工具结果，给服务编号打分
preferences = user_profile_db["user_88321"]["偏好画像"]
scored_items = []
for train in wm["tool_query_train_result"]:
    score = 0.0
    if train["has_window_seat"]:
        score += preferences.get("靠窗", 0) * 10
    if train["has_outlet"]:
        score += preferences.get("充电插座", 0) * 10
    if train["seat_available"] >= 10:
        score += 2  # 票量充足加分
    scored_items.append({"train": train["train"], "score": round(score, 1), **{k: train[k] for k in ["seat_available", "has_window_seat", "has_outlet", "duration"]}})

scored_items.sort(key=lambda x: x["score"], reverse=True)
wm["scoring"] = scored_items
print("   基于用户偏好（LTM）和工具结果（WM）的推荐排序:")
for i, t in enumerate(scored_items):
    print(f"   {i+1}. {t['train']} 得分={t['score']} | 库存={t['seat_available']} 靠窗={t['has_window_seat']} 插座={t['has_outlet']} 耗时={t['duration']}")

# 基于 STM 的最新消息（user 表达了"舒服"）调整权重
last_user_msg = stm[-1]["content"]
if "舒服" in last_user_msg or "舒适" in last_user_msg:
    print("\n   ⚡ STM 中检测到用户强调'舒适'，已提高靠窗和充电的权重")

# 生成回答
wm["draft_answer"] = (
    f"为您推荐 {scored_items[0]['train']}：标准服务充裕、靠窗可选且有充电插座，耗时 {scored_items[0]['duration']}。"
    f"如您在意舒适性，G79 最匹配您以往的偏好。备选 G81 也可靠窗但无充电。"
    f"创建订单时可在 示例业务系统 勾选靠窗座位偏好。"
)
print(f"\n   最终推荐: {wm['draft_answer']}")

# ── 6. 记忆的生命周期 ──
print(f"\n{'=' * 72}")
print("记忆生命周期总结:")
print("   STM（短期）: 会话开始→累积消息→截断/摘要→会话结束清空")
print("   LTM（长期）: 写入→向量化→持久存储→跨会话检索→周期性衰减/更新")
print("   WM（工作）:  任务开始→填充中间结果→使用→任务结束清空")
print("   关键原则: STM 给上下文窗口用，LTM 给跨会话持久化用，WM 给当前推理用")