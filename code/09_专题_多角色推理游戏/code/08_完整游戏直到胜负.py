"""08 完整游戏直到胜负。

课堂目标：
1. 展示完整 Game State：夜晚、白天、投票、胜负判断。
2. 展示 Multi-Agent：狼人、预言家、村民都由模型扮演。
3. 展示 Memory：公开历史会持续进入下一轮。
4. 展示系统裁判：程序负责更新状态和判断胜负。
5. Orchestrator 模式：每个玩家轮流行动，程序负责串联。

为了课堂演示稳定，使用 5 人极简局：
- A：狼人
- B：预言家
- C/D/E：村民

胜负规则：
- 狼人全部出局：好人阵营获胜。
- 狼人数量大于等于好人数量：狼人阵营获胜。
"""

import os
import json
import requests
from collections import Counter
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")


api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    env_file = str(COURSE_ROOT / "code/09_专题_多角色推理游戏/code/.env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("DEEPSEEK_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
if not api_key:
    raise SystemExit("请先在同目录 .env 中写入 DEEPSEEK_API_KEY，或设置系统环境变量。")

base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
url = base_url + "/chat/completions"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

roles = {
    "A": "狼人",
    "B": "预言家",
    "C": "村民",
    "D": "村民",
    "E": "村民",
}

all_players = list(roles.keys())
alive_players = ["A", "B", "C", "D", "E"]
public_history = ["主持人：游戏开始。"]
seer_memory = []
day = 1
winner = None

print("初始身份，仅课堂演示可见：")
print(json.dumps(roles, ensure_ascii=False, indent=2))
print("\n本脚本会完整打印：夜晚行动、天亮结果、每名玩家状态、存活玩家发言、投票、裁判检查、最终胜负。")
print("=" * 72)

while not winner and day <= 5:
    print(f"\n第 {day} 天 夜晚")
    print("=" * 72)
    print("当前存活玩家：", alive_players)
    print("当前死亡玩家：", [p for p in all_players if p not in alive_players])
    print("最近公开历史：")
    for item in public_history[-6:]:
        print("  -", item)

    # 每轮先统计一次阵营人数。
    wolf_alive = [p for p in alive_players if roles[p] == "狼人"]
    good_alive = [p for p in alive_players if roles[p] != "狼人"]
    print("阵营人数：狼人", len(wolf_alive), "好人", len(good_alive))

    if not wolf_alive:
        winner = "好人阵营"
        break
    if len(wolf_alive) >= len(good_alive):
        winner = "狼人阵营"
        break

    # 夜晚：狼人选择击杀目标。
    wolf = wolf_alive[0]
    kill_candidates = [p for p in alive_players if roles[p] != "狼人"]
    print("\n夜晚行动 1：狼人行动")
    print(f"狼人玩家：{wolf}")
    print("可击杀目标：", kill_candidates)

    messages = [
        {
            "role": "system",
            "content": (
                "你是狼人杀里的狼人。夜晚请从候选人中选择一名击杀目标。"
                "必须输出 JSON，字段为 target、reason。target 必须来自候选人。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "你是": wolf,
                    "你的身份": roles[wolf],
                    "可击杀目标": kill_candidates,
                    "公开历史": public_history[-10:],
                    "任务": "选择今晚击杀目标。",
                },
                ensure_ascii=False,
            ),
        },
    ]

    payload = {
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 400,
        "response_format": {"type": "json_object"},
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    wolf_action = json.loads(response.json()["choices"][0]["message"]["content"])
    killed = wolf_action.get("target")
    if killed not in kill_candidates:
        killed = kill_candidates[0]

    print("狼人 Agent 输出：")
    print(json.dumps(wolf_action, ensure_ascii=False, indent=2))
    print(f"夜晚结果：狼人选择击杀 {killed}。")

    # 夜晚：预言家查验。查验信息只进入 seer_memory，不进入公开历史。
    print("\n夜晚行动 2：预言家行动")
    if "B" in alive_players:
        check_candidates = [p for p in alive_players if p != "B"]
        print("预言家玩家：B")
        print("可查验目标：", check_candidates)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是狼人杀里的预言家。夜晚请从候选人中选择一名查验对象。"
                    "必须输出 JSON，字段为 target、reason。target 必须来自候选人。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "你是": "B",
                        "你的身份": "预言家",
                        "可查验目标": check_candidates,
                        "已有查验记忆": seer_memory,
                        "公开历史": public_history[-10:],
                        "任务": "选择今晚查验对象。",
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        payload = {
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 400,
            "response_format": {"type": "json_object"},
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        seer_action = json.loads(response.json()["choices"][0]["message"]["content"])
        checked = seer_action.get("target")
        if checked not in check_candidates:
            checked = check_candidates[0]
        result = "狼人" if roles[checked] == "狼人" else "好人"
        seer_memory.append(f"第 {day} 夜查验 {checked}：{result}")
        print("预言家 Agent 输出：")
        print(json.dumps(seer_action, ensure_ascii=False, indent=2))
        print(f"查验结果：{checked} -> {result}。这条信息只进入预言家私有记忆。")
        print("预言家私有记忆：", seer_memory)
    else:
        print("预言家 B 已死亡，本夜没有查验。")

    # 天亮：公布死亡，移除玩家。
    if killed in alive_players:
        alive_players.remove(killed)
    public_history.append(f"第 {day} 夜晚：{killed} 死亡。")

    print("\n天亮公布")
    print("-" * 72)
    print(f"昨夜死亡：{killed}")
    print("天亮后存活玩家：", alive_players)
    print("天亮后死亡玩家：", [p for p in all_players if p not in alive_players])

    print(f"\n第 {day} 天 白天")
    print("=" * 72)
    print(f"存活玩家：{alive_players}")
    print(f"夜晚死亡：{killed}")

    wolf_alive = [p for p in alive_players if roles[p] == "狼人"]
    good_alive = [p for p in alive_players if roles[p] != "狼人"]
    if not wolf_alive:
        winner = "好人阵营"
        break
    if len(wolf_alive) >= len(good_alive):
        winner = "狼人阵营"
        break

    # 白天：每个存活玩家发言。
    print("\n白天发言阶段")
    print("-" * 72)
    for player in all_players:
        if player not in alive_players:
            print(f"{player}（{roles[player]}）：已死亡，跳过发言。")
            continue

        private_info = []
        if player == "B":
            private_info = seer_memory

        print(f"\n轮到 {player} 发言")
        print(f"课堂可见身份：{roles[player]}")
        if private_info:
            print("该玩家私有信息：", private_info)
        else:
            print("该玩家私有信息：无")

        messages = [
            {
                "role": "system",
                "content": (
                    "你是狼人杀玩家。请根据公开历史和自己的身份发言。"
                    "必须输出 JSON，字段为 speech、suspect、reason。"
                    "不要直接泄露不该公开的系统信息。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "你是": player,
                        "你的身份": roles[player],
                        "存活玩家": alive_players,
                        "公开历史": public_history[-12:],
                        "你的私有信息": private_info,
                        "任务": "请做一段白天发言，说明你怀疑谁以及原因。",
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        payload = {
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 500,
            "response_format": {"type": "json_object"},
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        speech_data = json.loads(response.json()["choices"][0]["message"]["content"])

        speech = speech_data.get("speech", "")
        line = f"{player}：{speech}"
        public_history.append(line)
        print("发言 Agent 输出 JSON：")
        print(json.dumps(speech_data, ensure_ascii=False, indent=2))
        print("公开发言：", line)
        print("怀疑对象：", speech_data.get("suspect"))
        print("发言理由：", speech_data.get("reason"))

    # 白天：投票。
    print("\n投票阶段")
    print("-" * 72)
    votes = {}

    for player in all_players:
        if player not in alive_players:
            print(f"{player}（{roles[player]}）：已死亡，跳过投票。")
            continue

        candidates = [p for p in alive_players if p != player]
        private_info = seer_memory if player == "B" else []
        print(f"\n轮到 {player} 投票")
        print("可投票对象：", candidates)

        messages = [
            {
                "role": "system",
                "content": (
                    "你是狼人杀玩家，现在必须投票。"
                    "必须输出 JSON，字段为 vote、reason。vote 必须来自候选人。"
                    "reason 只能写公开可说的理由，不能说出自己的真实身份或夜晚行动。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "你是": player,
                        "你的身份": roles[player],
                        "候选人": candidates,
                        "公开历史": public_history[-14:],
                        "你的私有信息": private_info,
                        "任务": "请选择一名投票对象。",
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        payload = {
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 400,
            "response_format": {"type": "json_object"},
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        vote_data = json.loads(response.json()["choices"][0]["message"]["content"])

        vote = vote_data.get("vote")
        if vote not in candidates:
            vote = candidates[0]
        votes[player] = vote
        print("投票 Agent 输出 JSON：")
        print(json.dumps(vote_data, ensure_ascii=False, indent=2))
        print(f"{player} 投票给 {vote}，理由：{vote_data.get('reason')}")

    vote_count = Counter(votes.values())
    out_player = sorted(vote_count.items(), key=lambda x: (-x[1], x[0]))[0][0]
    public_history.append(f"第 {day} 天投票：{json.dumps(votes, ensure_ascii=False)}")
    public_history.append(f"第 {day} 天出局：{out_player}")

    print("\n投票统计：", dict(vote_count))
    print("完整投票明细：")
    print(json.dumps(votes, ensure_ascii=False, indent=2))
    print("白天出局：", out_player)

    if out_player in alive_players:
        alive_players.remove(out_player)

    wolf_alive = [p for p in alive_players if roles[p] == "狼人"]
    good_alive = [p for p in alive_players if roles[p] != "狼人"]

    print("\n裁判检查")
    print("-" * 72)
    print("存活玩家：", alive_players)
    print("狼人存活：", wolf_alive)
    print("好人存活：", good_alive)
    print("死亡玩家：", [p for p in all_players if p not in alive_players])
    print("胜负规则：狼人全出局则好人胜；狼人数量大于等于好人数量则狼人胜；否则进入下一天。")

    if not wolf_alive:
        winner = "好人阵营"
    elif len(wolf_alive) >= len(good_alive):
        winner = "狼人阵营"
    else:
        day += 1

if not winner:
    # 防止课堂演示因为模型投票太分散导致循环过长。
    wolf_alive = [p for p in alive_players if roles[p] == "狼人"]
    good_alive = [p for p in alive_players if roles[p] != "狼人"]
    winner = "狼人阵营" if len(wolf_alive) >= len(good_alive) else "好人阵营"

print("\n" + "=" * 72)
print("游戏结束")
print("最终身份复盘：")
print(json.dumps(roles, ensure_ascii=False, indent=2))
print("最终存活玩家：", alive_players)
print("最终死亡玩家：", [p for p in all_players if p not in alive_players])
print("最终公开历史：")
print(json.dumps(public_history, ensure_ascii=False, indent=2))
print(f"\n系统裁判输出：{winner}获胜。")