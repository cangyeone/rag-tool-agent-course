"""07 HTTP 服务接口草图。

学习目标：理解将一个 LLM 调用能力封装为 HTTP 服务需要哪些核心组件——路由设计、请求解析、业务处理、响应构造。用模拟数据展示完整的请求→处理→响应链路，不依赖真实模型或外部 API。

运行方式：python 07_HTTP_服务接口草图.py
"""

import json
import time
from pathlib import Path

COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

print("07 HTTP 服务接口草图")
print("=" * 72)
print("演示目标：把命令行模型调用拆成 Web API 的三件事 —— 请求、处理、响应。")
print()


# ========== 一、为什么需要 HTTP 服务 ==========
print("=" * 60)
print("一、为什么要把 LLM 包装成 HTTP 服务")
print("=" * 60)

print("  本地调用（transformers）：")
print("    tokenizer(text) → model.generate(input_ids) → tokenizer.decode(output_ids)")
print("    问题：只能本机单进程用，前端、其他服务、多用户都没法调。")
print()
print("  HTTP 服务化之后：")
print("    任何能发 HTTP 请求的客户端（浏览器、curl、App）都能用同一个模型。")
print("    服务端负责加载模型、管理并发、记录日志，客户端只管发 JSON。")
print()
print("  课堂类比：自助业务受理机 = HTTP 接口，业务受理系统 = 后端模型。")
print("  买票的人不需要知道后端怎么查库存、怎么出票——他只需要按屏幕上的按钮。")
print()


# ========== 二、最小 HTTP 服务的三个核心问题 ==========
print("=" * 60)
print("二、最小 HTTP 服务的三件事：请求 → 处理 → 响应")
print("=" * 60)

api_flow = [
    ("请求 (Request)",   "客户端 POST JSON 到 /chat，body 包含 question 和可选参数"),
    ("处理 (Process)",   "服务端解析请求 → 拼装 prompt → 调用 LLM → 拿到 answer"),
    ("响应 (Response)",  "服务端返回 JSON：question、answer、note 和 token 用量"),
]
for name, desc in api_flow:
    print(f"  ■ {name}")
    print(f"    {desc}")
print()


# ========== 三、请求：客户端发送什么 ==========
print("=" * 60)
print("三、请求体设计 —— 客户端发什么 JSON")
print("=" * 60)

# 模拟一个 示例业务系统 客服场景的请求
example_request = {
    "question": "ORD-1001 标准服务没票了，候补申请能成功吗？",
    "context": {
        "user_level": "普通用户",
        "channel": "APP 在线客服",
    },
    "options": {
        "max_tokens": 300,
        "temperature": 0.3,
    },
}
print("  一个典型的客服请求 JSON：")
print(json.dumps(example_request, ensure_ascii=False, indent=2))
print()
print("  【请求字段说明】")
print(f"  {'字段':<18} {'类型':<10} {'必填':<6} {'说明'}")
print(f"  {'─'*18} {'─'*10} {'─'*6} {'─'*30}")
fields = [
    ("question", "string", "是", "用户输入的问题文本"),
    ("context",  "object", "否", "上下文信息：用户等级、渠道、历史等"),
    ("options",  "object", "否", "生成参数：max_tokens、temperature 等"),
]
for name, typ, required, desc in fields:
    print(f"  {name:<18} {typ:<10} {required:<6} {desc}")


# ========== 四、处理：服务端怎么做 ==========
print()
print("=" * 60)
print("四、服务端处理流程 —— 从收到请求到返回答案")
print("=" * 60)

# 模拟在线客服知识库
RAILWAY_KNOWLEDGE = {
    "候补申请规则": "候补申请按排队顺序兑现，不保证成功。截止时间为服务开始前 2 小时。",
    "退款规则": "服务开始前 8 天以上免手续费，48 小时-8 天收 5%，24-48 小时收 10%，不足 24 小时收 20%。",
    "变更规则": "服务开始前可免费变更一次，变更后订单日期在春运期间的不办理退款款。",
    "学生优惠": "每年 6 月 1 日至 9 月 30 日、12 月 1 日至 3 月 31 日可购买。需进行优惠资质核验。",
}

def simulate_llm_process(question: str, context: dict = None, options: dict = None) -> dict:
    """模拟服务端处理流程（实际应调用 LLM）。"""
    context = context or {}
    options = options or {}

    print(f"\n  >>> 第 1 步：解析请求")
    print(f"    question = {question[:50]}...")
    print(f"    context  = {json.dumps(context, ensure_ascii=False)}")
    print(f"    options  = {json.dumps(options, ensure_ascii=False)}")

    # 第 2 步：检索相关知识（模拟 RAG 流程）
    print(f"\n  >>> 第 2 步：检索相关知识")
    matched = []
    for keyword, answer in RAILWAY_KNOWLEDGE.items():
        if any(kw in question for kw in [keyword, keyword.replace("规则", "")]):
            matched.append((keyword, answer))
    if not matched:
        matched.append(("通用回复", "请以 官方页面实时信息为准。"))
    for kw, ans in matched:
        print(f"    命中「{kw}」: {ans[:50]}...")

    # 第 3 步：拼装 prompt（模拟 chat template）
    print(f"\n  >>> 第 3 步：拼装 prompt")
    system_prompt = "你是 通用客服助手，回答简洁专业。如涉及业务规则，请注明以官方页面为准。"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    print(f"    messages = {json.dumps(messages, ensure_ascii=False)[:100]}...")

    # 第 4 步：调用 LLM（模拟）
    print(f"\n  >>> 第 4 步：调用 LLM（模拟）")
    print(f"    参数: max_tokens={options.get('max_tokens', 300)}, temperature={options.get('temperature', 0.3)}")
    time.sleep(0.1)  # 模拟推理耗时

    # 第 5 步：构造结构化回答
    answer = f"候补申请按排队顺序兑现，不保证一定成功。建议同时关注同方向其他服务编号（如 G1/G3/G5 等）。截止时间为服务开始前 2 小时，请以 官方页面为准。"
    print(f"\n  >>> 第 5 步：构造回答")
    print(f"    answer = {answer[:60]}...")

    # 第 6 步：组装响应
    prompt_tokens = len(question) + len(system_prompt)
    completion_tokens = len(answer)
    result = {
        "question": question,
        "answer": answer,
        "knowledge_used": [kw for kw, _ in matched],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "note": "本回答由模拟引擎生成，实际应调用 LLM。以官方页面为准。",
        "timestamp": time.time(),
    }
    return result

# 运行模拟
result = simulate_llm_process(
    question="候补申请能不能保证成功？",
    context={"user_level": "普通用户"},
    options={"max_tokens": 300, "temperature": 0.3},
)


# ========== 五、响应：客户端收到什么 ==========
print()
print("=" * 60)
print("五、响应体设计 —— 客户端收到什么 JSON")
print("=" * 60)

print("  服务端返回的标准 JSON 响应：")
print(json.dumps(result, ensure_ascii=False, indent=2))
print()
print("  【响应字段说明】")
print(f"  {'字段':<18} {'类型':<10} {'说明'}")
print(f"  {'─'*18} {'─'*10} {'─'*30}")
resp_fields = [
    ("question",  "string", "原始问题（回显，方便核对）"),
    ("answer",    "string", "模型生成的回答"),
    ("knowledge_used", "list", "命中的知识条目（便于溯源）"),
    ("usage",     "object", "token 用量统计"),
    ("note",      "string", "免责/辅助说明"),
    ("timestamp", "float",  "响应时间戳"),
]
for name, typ, desc in resp_fields:
    print(f"  {name:<18} {typ:<10} {desc}")


# ========== 六、完整 HTTP 交互示例 ==========
print()
print("=" * 60)
print("六、完整 HTTP 交互示例（用 curl 模拟）")
print("=" * 60)

print("""
  客户端发送：
    curl -X POST http://localhost:8000/chat \\
      -H "Content-Type: application/json" \\
      -d '{"question": "候补申请能不能保证成功？"}'

  服务端响应：
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "question": "候补申请能不能保证成功？",
      "answer": "候补申请按排队顺序兑现...",
      "usage": {"prompt_tokens": 61, "completion_tokens": 65, "total_tokens": 126},
      "note": "以官方页面为准。"
    }
""")


# ========== 七、路由设计 ==========
print("=" * 60)
print("七、一个最小 LLM 服务需要哪些路由")
print("=" * 60)

routes = [
    ("GET  /health",    "健康检查",     "返回 {\"status\": \"ok\"}，负载均衡器用它判断服务是否存活"),
    ("GET  /models",    "模型列表",     "返回可用模型清单，前端切换模型时用"),
    ("POST /chat",      "对话接口",     "核心接口，接收 question 返回 answer"),
    ("POST /chat/stream","流式对话",    "SSE 流式输出，逐 token 返回，适合实时交互"),
]
print(f"  {'路由':<20} {'用途':<12} {'说明'}")
print(f"  {'─'*20} {'─'*12} {'─'*40}")
for route, usage, desc in routes:
    print(f"  {route:<20} {usage:<12} {desc}")


# ========== 八、同步 vs 流式 ==========
print()
print("=" * 60)
print("八、同步响应 vs 流式响应")
print("=" * 60)

comparison = [
    ("返回方式",   "一次性返回完整 JSON",             "SSE 事件流逐块推送"),
    ("Content-Type","application/json",              "text/event-stream"),
    ("客户端体验", "等待完成后一次性展示",            "逐字打出，打字机效果"),
    ("首字延迟",   "较高（需等全部生成完）",          "较低（第一个 token 就开始返回）"),
    ("适用场景",   "后台批处理、日志分析",            "实时客服、在线对话"),
    ("实现复杂度", "简单，一次 HTTP 请求即可",        "需处理 iter_lines 和 SSE 协议"),
]
print(f"  {'维度':<14} {'同步 (一次性返回)':<30} {'流式 (SSE)'}")
print(f"  {'─'*14} {'─'*30} {'─'*30}")
for dim, sync_val, stream_val in comparison:
    print(f"  {dim:<14} {sync_val:<30} {stream_val}")


# ========== 九、完整服务实现伪代码 ==========
print()
print("=" * 60)
print("九、服务端实现伪代码（Flask 示例）")
print("=" * 60)

pseudocode = '''
# ── 服务端伪代码 ──
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data["question"]

    # 1. 检索知识
    knowledge = search_knowledge(question)

    # 2. 拼装 prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question + knowledge},
    ]

    # 3. 调用 LLM
    answer = call_llm(messages)

    # 4. 返回响应
    return jsonify({
        "question": question,
        "answer": answer,
        "usage": {"total_tokens": len(question) + len(answer)},
    })

app.run(host="0.0.0.0", port=8000)
'''
print(pseudocode)


# ========== 十、示例业务系统 行业场景应用 ==========
print("=" * 60)
print("十、HTTP 服务在 示例业务系统 场景的典型应用")
print("=" * 60)

scenarios = [
    {
        "场景": "在线客服助手",
        "请求": "用户在 APP 里输入问题 → POST /chat",
        "处理": "检索订单服务规则 → 拼装 prompt → 调用 LLM",
        "响应": "返回专业回答 + 引用规则编号",
    },
    {
        "场景": "工单自动分类",
        "请求": "系统收到退款投诉 → POST /chat",
        "处理": "解析投诉内容 → 匹配退款规则 → 判断责任方",
        "响应": "返回分类标签 (退款/变更/投诉) + 建议处理方案",
    },
    {
        "场景": "安全检查辅助",
        "请求": "巡检员上传问题描述 → POST /chat",
        "处理": "检索安全规章 → 对比巡检描述 → 判断风险等级",
        "响应": "返回风险评估 + 引用规章条款",
    },
    {
        "场景": "培训知识问答",
        "请求": "新员工输入规章问题 → POST /chat",
        "处理": "检索培训材料 → 找到对应章节 → 生成解释",
        "响应": "返回易懂解释 + 原文出处链接",
    },
]

for s in scenarios:
    print(f"\n  【{s['场景']}】")
    print(f"    请求: {s['请求']}")
    print(f"    处理: {s['处理']}")
    print(f"    响应: {s['响应']}")


# ========== 十一、总结 ==========
print()
print("=" * 60)
print("十一、本节要点")
print("=" * 60)
print("  1. HTTP 服务化 = 把 LLM 调用包装成 Web API，让任何客户端都能调")
print("  2. 核心三件事：请求（JSON 入参）→ 处理（检索+prompt+推理）→ 响应（JSON 出参）")
print("  3. 最小路由：/health（探活）+ /chat（对话）+ /chat/stream（流式）")
print("  4. 同步 vs 流式：同步适合批量处理，流式适合实时交互")
print("  5. 示例业务系统 场景：在线客服、工单分类、巡检辅助、培训问答都适用这个模式")
print()
print("  课堂可修改点：")
print("    • 修改 question，观察模拟回答的变化")
print("    • 在 RAILWAY_KNOWLEDGE 中增加一条规则，再看回答是否引用")
print("    • 故意传入空 question，思考客户端应如何校验")
print("    • 修改 simulate_llm_process 中的 answer 生成逻辑，模拟不同的 LLM 回答风格")
