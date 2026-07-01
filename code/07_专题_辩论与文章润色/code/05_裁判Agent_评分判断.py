"""05 裁判 Agent：判断哪个辩手说得好。

课堂目标：
1. 裁判 Agent 不参与辩论，只做评分。
2. 裁判输出 JSON，程序读取 winner 和 score。
3. 评分标准公开，方便学生理解。
"""

import os
import json
import requests
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")


api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    env_file = str(COURSE_ROOT / "code/07_专题_辩论与文章润色/code/.env")
    if os.path.exists(env_file):
        for line in open(env_file, "r", encoding="utf-8"):
            if line.strip().startswith("DEEPSEEK_API_KEY="):
                api_key = line.strip().split("=", 1)[1]
if not api_key:
    raise SystemExit("请设置 DEEPSEEK_API_KEY，或复制仓库根目录 .env.example 为 .env 后填写。")

base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
url = base_url + "/chat/completions"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

topic = "企业内部知识问答系统应该优先使用 RAG，而不是微调。"

debate_record = [
    "正方：RAG 能直接接入最新文档，适合制度变化快的企业场景。",
    "反方：微调能让模型稳定掌握固定任务风格，减少每次检索的不确定性。",
    "正方：RAG 可以给出引用来源，方便审计和追溯。",
    "反方：RAG 依赖文档质量和检索质量，资料差时回答也会差。",
    "正方：微调更新成本高，企业知识经常变化时维护压力更大。",
    "反方：最好的方案可能是 RAG 与微调结合，而不是只选一种。",
]

scoring_rule = {
    "论点清晰": 30,
    "证据与例子": 25,
    "回应对手": 25,
    "表达质量": 20,
}

messages = [
    {
        "role": "system",
        "content": (
            "你是辩论裁判。请根据评分标准判断正方和反方谁说得更好。"
            "必须输出 JSON，字段为 pro_score、con_score、winner、reason、best_sentence、improvement。"
        ),
    },
    {
        "role": "user",
        "content": json.dumps(
            {
                "辩题": topic,
                "辩论记录": debate_record,
                "评分标准": scoring_rule,
                "要求": "分数满分 100，winner 只能是 正方、反方、平局。",
            },
            ensure_ascii=False,
        ),
    },
]

payload = {
    "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    "messages": messages,
    "temperature": 0.2,
    "max_tokens": 700,
    "response_format": {"type": "json_object"},
}

print("辩论记录：")
for line in debate_record:
    print(line)
print("-" * 72)

response = requests.post(url, headers=headers, json=payload, timeout=60)
response.raise_for_status()
judge = json.loads(response.json()["choices"][0]["message"]["content"])

print("裁判 JSON：")
print(json.dumps(judge, ensure_ascii=False, indent=2))

print("\n程序读取结果：")
print("正方分数：", judge.get("pro_score"))
print("反方分数：", judge.get("con_score"))
print("胜方：", judge.get("winner"))
print("理由：", judge.get("reason"))
