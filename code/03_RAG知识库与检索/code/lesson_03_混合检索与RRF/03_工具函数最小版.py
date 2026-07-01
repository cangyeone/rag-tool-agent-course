"""03_工具函数最小版。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
学习目标：理解工具函数（Tool/Function Calling）的核心概念，
         掌握工具定义 schema 的写法，区分检索（RAG）与工具调用的差异。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import time
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")

# ── 从课程目录加载真实文档 ──
def _load_real_docs():
    real_docs = []
    sample_files = [
        ("code/01_大模型基础/README.md", 350),
        ("code/02_模型接口与指令设计/README.md", 350),
        ("code/03_RAG知识库与检索/README.md", 350),
        ("README.md", 350),
    ]
    for rel_path, max_chars in sample_files:
        file_path = COURSE_ROOT / rel_path
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")[:max_chars]
            title = rel_path.split("/")[1] if rel_path.startswith("code/") else "课程总览"
            real_docs.append({"title": title, "source": rel_path, "content": content})
    return real_docs

docs = _load_real_docs()


question = "帮我查一下今天的ORD-1001次服务流程还有库存吗？"

print("=" * 72)
print("03_工具函数最小版 —— 工具 Schema 定义与调用")
print("=" * 72)

# ═══════════════════════════════════════════
# 一、什么是工具函数？
# ═══════════════════════════════════════════
print("\n一、检索（RAG） vs 工具调用（Tool Calling）")
print("""
  检索（RAG）：
    从知识库中「找」已有的文档/段落
    回答的是「知识性」问题：规章、FAQ、说明
    示例：「候补申请能保证成功吗？」→ 查规章文档

  工具调用（Tool Calling）：
    调用外部系统/API「获取」实时数据
    回答的是「事实性」问题：当前状态、查询结果
    示例：「ORD-1001次服务流程还有库存吗？」→ 调用库存查询接口

  关键区别：知识库是「静态的过去知识」，工具是「动态的当前数据」。
""")

# ═══════════════════════════════════════════
# 二、工具 Schema 定义（Function Calling 格式）
# ═══════════════════════════════════════════
print("━" * 60)
print("二、工具 Schema 定义")
print("━" * 60)

# 模拟 OpenAI / 通用 function calling 格式
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_inventory",
            "description": "查询指定服务编号的库存信息，返回各服务类型的剩库存数。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "服务编号号，如 ORD-1001、D310",
                    },
                    "date": {
                        "type": "string",
                        "description": "查询日期，格式 YYYY-MM-DD，默认今天",
                    },
                    "from_location": {
                        "type": "string",
                        "description": "起点中文名，如 服务点A",
                    },
                    "to_location": {
                        "type": "string",
                        "description": "终点中文名，如 服务点B",
                    },
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_service_point_code",
            "description": "根据服务点中文名查询服务点编码（电报码）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_point_name": {
                        "type": "string",
                        "description": "服务点中文名",
                    }
                },
                "required": ["service_point_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_service_refund_fee",
            "description": "根据价格和退款时间计算退款费用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {
                        "type": "number",
                        "description": "原价格（元）",
                    },
                    "hours_before_departure": {
                        "type": "number",
                        "description": "距离服务开始的小时数",
                    },
                },
                "required": ["order_amount", "hours_before_departure"]
            }
        }
    },
]

print(f"  已定义 {len(tools)} 个工具：")
for t in tools:
    func = t["function"]
    params = list(func["parameters"]["properties"].keys())
    required = func["parameters"].get("required", [])
    print(f"    ✓ {func['name']}")
    print(f"      参数：{params}，必需：{required}")
    print(f"      描述：{func['description']}")
    print()

# ═══════════════════════════════════════════
# 三、工具的实现（模拟）
# ═══════════════════════════════════════════
print("━" * 60)
print("三、工具实现（模拟数据）")
print("━" * 60)

def query_inventory(order_id, date="2025-06-19",
                               from_location=None, to_location=None):
    """模拟库存查询。"""
    return {
        "order_id": order_id,
        "date": date,
        "from": from_location or "服务点A",
        "to": to_location or "服务点B",
        "seats": {
            "商务座": 3,
            "高级服务": 12,
            "标准服务": 0,         # 已无库存
            "无座": 25,
        },
        "status": "部分服务类型已无库存",
        "query_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

def query_service_point_code(service_point_name):
    """模拟服务点编码查询。"""
    code_map = {
        "服务点A": "BJP",
        "服务点B": "SP-B",
        "杭州东": "HGH",
        "南京南": "NKH",
    }
    code = code_map.get(service_point_name, "未知")
    return {
        "service_point_name": service_point_name,
        "telegram_code": code,
        "status": "found" if code != "未知" else "not_found",
    }

def calculate_service_refund_fee(order_amount, hours_before_departure):
    """模拟退款费用计算。"""
    if hours_before_departure > 8 * 24:
        rate = 0.05
    elif hours_before_departure > 48:
        rate = 0.10
    elif hours_before_departure > 24:
        rate = 0.20
    else:
        rate = 0.20
    fee = order_amount * rate
    return {
        "order_amount": order_amount,
        "hours_before_departure": hours_before_departure,
        "fee_rate": f"{rate*100:.0f}%",
        "refund_fee": round(fee, 2),
        "refund_amount": round(order_amount - fee, 2),
    }

# 演示调用
print("\n  调用 query_inventory(\"ORD-1001\")：")
result1 = query_inventory("ORD-1001")
print(f"    {json.dumps(result1, ensure_ascii=False)}")

print("\n  调用 query_service_point_code(\"服务点A\")：")
result2 = query_service_point_code("服务点A")
print(f"    {json.dumps(result2, ensure_ascii=False)}")

print("\n  调用 calculate_service_refund_fee(553, 36)：")
result3 = calculate_service_refund_fee(553, 36)
print(f"    {json.dumps(result3, ensure_ascii=False)}")

# ═══════════════════════════════════════════
# 四、工具调用与 RAG 的协作流程
# ═══════════════════════════════════════════
print("\n" + "━" * 60)
print("四、工具调用与 RAG 的协作流程")
print("━" * 60)

print("""
  用户问题：{question}

  步骤 1：意图识别
    → 问题涉及两类信息：
      a) 「有没有库存」→ 需要调用工具查询实时数据
      b) 「候补申请规则」→ 需要 RAG 检索知识库

  步骤 2：并行执行
    → 工具调用：query_inventory("ORD-1001") → 标准服务已无库存
    → RAG 检索：「候补申请规则」文档 → 候补申请不保证成功

  步骤 3：结果融合（下一课详细讲解）
    → 将工具结果 + RAG 结果组装成完整回答
""".format(question=question))

# ═══════════════════════════════════════════
# 五、工具 Schema 的设计原则
# ═══════════════════════════════════════════
print("━" * 60)
print("五、工具 Schema 设计原则")
print("━" * 60)

print("""
  1. description 要精确
     → 好的：「查询指定服务编号的库存信息」
     → 差的：「查询」
     → 原因是 LLM 根据描述决定是否调用此工具。

  2. 参数约束要完整
     → 标明 required（必需参数），避免 LLM 遗漏
     → 使用 enum 限制可选值范围

  3. 返回结果要结构化
     → 统一使用 JSON 格式
     → 包含 status 字段（success/error/partial）

  4. 工具函数的职责要单一
     → 一个工具只做一件事
     → 不要 query_and_book_order（查询+创建订单混在一起）
""")

print("=" * 72)
print("学习要点：")
print("  1. RAG = 查知识库（静态），工具 = 调接口（动态）")
print("  2. 工具 Schema 决定了 LLM 能否正确理解和调用工具")
print("  3. description 是 LLM 判断是否调用的唯一依据")
print("  4. 返回 JSON 结构易于被 LLM 理解和组装回答")