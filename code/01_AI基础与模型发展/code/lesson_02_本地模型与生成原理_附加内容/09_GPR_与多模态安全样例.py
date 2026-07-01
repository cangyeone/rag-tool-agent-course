"""09 GPR 与多模态安全样例。

学习目标：理解行业安全场景中的多模态数据分析流程——如何将图像描述、里程文本、巡检记录三种模态的信息合并在一起，由大模型给出综合辅助判断。

运行方式：python 09_GPR_与多模态安全样例.py
"""


import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
import hashlib
import time

print("09 GPR 与多模态安全样例")
print("=" * 72)

print("一、什么是 GPR（探地雷达）？")
print("  GPR (Ground Penetrating Radar) 通过发射电磁波探测地下结构。")
print("  在行业养护中，GPR 用于检测路基含水、空洞、沉降等隐患。")
print("  但 GPR 图像解读需要专业经验，多模态分析可以辅助判断。")
print()

# ==================== 多模态数据模拟 ====================

print("二、多模态数据输入（模拟某业务服务区间的巡检记录）")

# 模态1：图像分析结果
image_analysis = {
    "modality": "visual",
    "source": "GPR 雷达剖面截图",
    "frame_id": "GPR_20260120_K128_450",
    "observations": [
        {"region": "0-2m 深度", "feature": "连续强反射带", "confidence": 0.92},
        {"region": "2-4m 深度", "feature": "反射相位反转", "confidence": 0.78},
        {"region": "4-6m 深度", "feature": "信号衰减增强（含水特征）", "confidence": 0.85},
    ],
    "interpretation": "雷达剖面显示 K128+450 处存在疑似含水异常区，2-4m 处相位反转提示可能存在空洞。",
}
print("\n  [模态1] GPR 图像分析：")
print(f"    源文件：{image_analysis['frame_id']}")
for obs in image_analysis["observations"]:
    print(f"    {obs['region']}: {obs['feature']} (置信度 {obs['confidence']})")

# 模态2：里程文本记录
text_record = {
    "modality": "text",
    "source": "巡检工单 #WO-2026-0119-0842",
    "section": {
        "line": "示例线路业务服务",
        "mileage": "K128+450",
        "direction": "下行",
        "track_type": "无砟轨道",
    },
    "context": [
        {"date": "2026-01-15", "event": "连续降雨 3 天，累计降水量 68mm"},
        {"date": "2026-01-18", "event": "雨后巡检发现 K128+440 处排水沟淤积"},
        {"date": "2026-01-19", "event": "K128+450 处路基边坡出现轻微渗水痕迹"},
    ],
    "priority": "中等 — 需在 7 天内复测",
}
print(f"\n  [模态2] 文本记录：")
print(f"    线路：{text_record['section']['line']} {text_record['section']['mileage']}")
for ctx in text_record["context"]:
    print(f"    {ctx['date']}: {ctx['event']}")
print(f"    处理优先级：{text_record['priority']}")

# 模态3：历史病害数据
historical_data = {
    "modality": "structured",
    "source": "病害数据库",
    "mileage": "K128+450",
    "history": [
        {"date": "2024-07-22", "type": "路基沉降", "severity": "轻微", "status": "已修复"},
        {"date": "2025-03-15", "type": "排水不畅", "severity": "中等", "status": "监测中"},
    ],
    "nearby_issues": [
        {"mileage": "K128+350", "type": "边坡冲刷", "distance_m": 100},
        {"mileage": "K128+600", "type": "道床板结", "distance_m": 150},
    ],
}
print(f"\n  [模态3] 历史病害数据：")
for h in historical_data["history"]:
    print(f"    {h['date']}: {h['type']} ({h['severity']}) — {h['status']}")
print(f"    邻近病害：")
for n in historical_data["nearby_issues"]:
    print(f"    {n['mileage']} ({n['distance_m']}m): {n['type']}")

# ==================== 多模态融合分析 ====================

print("\n\n三、多模态融合分析")

# 融合各模态信息，加权评分
risk_factors = []

# 从图像分析提取风险因子
for obs in image_analysis["observations"]:
    if obs["confidence"] > 0.8:
        if "含水" in obs["feature"] or "水" in obs["feature"]:
            risk_factors.append(("含水异常", obs["confidence"] * 0.35, "GPR"))
        if "空洞" in obs["feature"] or "相位反转" in obs["feature"]:
            risk_factors.append(("空洞风险", obs["confidence"] * 0.4, "GPR"))

# 从文本记录提取风险因子
for ctx in text_record["context"]:
    if "降雨" in ctx["event"] or "降水" in ctx["event"]:
        risk_factors.append(("降雨影响", 0.25, "文本"))
    if "渗水" in ctx["event"]:
        risk_factors.append(("渗水痕迹", 0.6, "文本"))
    if "淤积" in ctx["event"]:
        risk_factors.append(("排水不畅", 0.3, "文本"))

# 从历史数据提取风险因子
for h in historical_data["history"]:
    if h["status"] == "监测中":
        risk_factors.append(("历史病害未结案", 0.5, "历史"))
    if h["severity"] == "中等" and h["date"] > "2024-01-01":
        risk_factors.append(("近期中等病害", 0.3, "历史"))

# 邻近病害加权
for n in historical_data["nearby_issues"]:
    if n["distance_m"] < 200:
        risk_factors.append((f"邻近{n['type']}({n['distance_m']}m)", 0.2, "历史"))

print("\n  提取的风险因子：")
total_risk = 0
for factor, weight, source in risk_factors:
    print(f"    [{source}] {factor}: +{weight:.2f}")
    total_risk += weight

# 风险等级判定
if total_risk >= 1.5:
    risk_level = "高"
    action = "建议立即安排现场复测，暂停该区段服务流程降速观察"
elif total_risk >= 0.8:
    risk_level = "中"
    action = "建议 7 天内安排复测，通知工务段加强巡检"
elif total_risk >= 0.3:
    risk_level = "低"
    action = "纳入下次定期巡检计划，持续关注"
else:
    risk_level = "正常"
    action = "无需特殊处理"

print(f"\n  总风险得分：{total_risk:.2f} / 3.0")
print(f"  风险等级：{risk_level}")
print(f"  建议措施：{action}")

# ==================== 综合判定 ====================

print("\n\n四、大模型综合判定（使用本地 Qwen3.5-0.8B）")

system_prompt = (
    "你是行业工务安全分析助手。根据提供的 GPR 图像分析、现场文本记录、历史病害数据三种模态信息，"
    "给出综合安全评估。注意：你只能做辅助判断，不能替代现场检测和专家复核。"
)

user_prompt = f"""请基于以下多模态信息，给出安全评估：

【GPR 图像分析】
源文件：{image_analysis['frame_id']}
观察：{'; '.join(o['feature'] for o in image_analysis['observations'])}
解读：{image_analysis['interpretation']}

【现场文本记录】
线路：{text_record['section']['line']} {text_record['section']['mileage']}
近期事件：{'; '.join(c['event'] for c in text_record['context'])}

【历史病害数据】
历史：{'、'.join(f"{h['type']}({h['severity']})" for h in historical_data['history'])}
邻近：{'、'.join(f"{n['type']}距{n['distance_m']}m" for n in historical_data['nearby_issues'])}

【系统评分】风险得分 {total_risk:.2f}，风险等级 {risk_level}"""

from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

qwen_candidate = COURSE_ROOT / "open_models" / "Qwen3.5-0.8B"
qwen_path = str(qwen_candidate) if qwen_candidate.is_dir() else None

if qwen_path:
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device != "cpu" else torch.float32
    q_tokenizer = AutoTokenizer.from_pretrained(qwen_path)
    q_model = AutoModelForCausalLM.from_pretrained(qwen_path, torch_dtype=dtype).to(device).eval()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = q_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = q_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = q_model.generate(**inputs, max_new_tokens=300, temperature=0.2, do_sample=True,
                                      pad_token_id=q_tokenizer.eos_token_id)
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    llm_response = q_tokenizer.decode(generated, skip_special_tokens=True).strip()
    print(f"\n  本地 Qwen 输出：\n{llm_response}")
else:
    llm_response = f"综合分析：\n1. GPR图像显示 K128+450 处存在含水异常和疑似空洞信号。\n2. 现场记录证实降雨后渗水，与 GPR 特征吻合。\n3. 综合风险评分 {total_risk:.2f} 分，属于{risk_level}风险等级。\n请下载 open_models/Qwen3.5-0.8B 获取真实模型推理结果。"

# ==================== 模拟结果嵌入工单 ====================

print("\n\n五、分析结果嵌入工单系统")
work_order = {
    "wo_id": f"WO-2026-{time.strftime('%m%d%H%M')}",
    "mileage": "K128+450",
    "risk_score": round(total_risk, 2),
    "risk_level": risk_level,
    "modalities_used": ["GPR图像", "文本记录", "历史病害"],
    "llm_conclusion": llm_response[:200],
    "recommended_action": action,
    "human_review_required": total_risk >= 0.8,
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}
print(json.dumps(work_order, ensure_ascii=False, indent=2))

print("\n\n六、多模态在行业安全中的价值")
value_points = [
    "GPR 图像 + 文本记录 + 历史数据 → 三种信息互相印证，减少单一模态的误判",
    "大模型负责整合不同来源的信息，给出结构化的分析报告",
    "系统评分提供可量化的风险基线，降低人工审查的主观偏差",
    "模型只做辅助判断，最终决策权在工程师手中——人机协同是行业安全的基本原则",
]
for v in value_points:
    print(f"  • {v}")

print("\n要点：多模态不是只看图片——而是把不同来源的结构化/非结构化信息一起交给模型做综合判断。")
print("行业安全场景中，模型输出只做辅助参考，不能替代现场检测和专家复核。")