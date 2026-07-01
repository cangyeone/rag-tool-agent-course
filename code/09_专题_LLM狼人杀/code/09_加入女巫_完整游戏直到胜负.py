"""09 加入女巫：完整游戏直到胜负。

课堂目标：
1. 在 08 的完整游戏基础上加入女巫。
2. 展示女巫的私有信息：夜晚知道被击杀的人。
3. 展示工具/状态思想：解药和毒药都是一次性资源。
4. 打印完整过程：夜晚行动、女巫选择、天亮结果、白天发言、投票、裁判胜负。

极简 5 人局：
- A：狼人
- B：预言家
- C：女巫
- D/E：村民

女巫规则：
- 解药只能用一次，可以救当晚被狼人击杀的人。
- 毒药只能用一次，可以在夜晚毒死一名存活玩家。
- 为了课堂展示，女巫每晚都会输出 save、poison、reason。

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
    env_file = str(COURSE_ROOT / "code/09_专题_LLM狼人杀/code/.env")
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
    "C": "女巫",
    "D": "村民",
    "E": "村民",
}

all_players = list(roles.keys())
alive_players = ["A", "B", "C", "D", "E"]
public_history = ["主持人：游戏开始。"]
seer_memory = []
witch_memory = []
witch_has_antidote = True
witch_has_poison = True
day = 1
winner = None

print("初始身份，仅课堂演示可见：")
print(json.dumps(roles, ensure_ascii=False, indent=2))
print("\n女巫初始资源：解药=True，毒药=True")
print("本脚本会完整打印：狼人击杀、预言家查验、女巫救/毒、天亮结果、发言、投票、胜负。")
print("=" * 72)

while not winner and day <= 5:
    print(f"\n第 {day} 天 夜晚")
    print("=" * 72)
    print("当前存活玩家：", alive_players)
    print("当前死亡玩家：", [p for p in all_players if p not in alive_players])
    print("最近公开历史：")
    for item in public_history[-6:]:
        print("  -", item)

    wolf_alive = [p for p in alive_players if roles[p] == "狼人"]
    good_alive = [p for p in alive_players if roles[p] != "狼人"]
    print("阵营人数：狼人", len(wolf_alive), "好人", len(good_alive))

    if not wolf_alive:
        winner = "好人阵营"
        break
    if len(wolf_alive) >= len(good_alive):
        winner = "狼人阵营"
        break

    night_deaths = []

    # 1. 狼人行动
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
    killed_by_wolf = wolf_action.get("target")
    if killed_by_wolf not in kill_candidates:
        killed_by_wolf = kill_candidates[0]

    print("狼人 Agent 输出：")
    print(json.dumps(wolf_action, ensure_ascii=False, indent=2))
    print(f"狼人夜晚选择击杀：{killed_by_wolf}")

    # 2. 预言家行动
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

    # 3. 女巫行动
    print("\n夜晚行动 3：女巫行动")
    saved_by_witch = False
    poisoned_by_witch = None
    if "C" in alive_players:
        witch_candidates = [p for p in alive_players if p != "C"]
        print("女巫玩家：C")
        print("今晚被狼人击杀的人：", killed_by_wolf)
        print("女巫资源：解药", witch_has_antidote, "毒药", witch_has_poison)
        print("可毒杀目标：", witch_candidates)

        messages = [
            {
                "role": "system",
                "content": (
                    "你是狼人杀里的女巫。你知道今晚被狼人击杀的人。"
                    "你有一次解药和一次毒药。请决定是否救人、是否毒人。"
                    "必须输出 JSON，字段为 save、poison、reason。"
                    "save 是 true/false。poison 可以是候选人或 null。"
                    "如果没有解药，save 必须为 false。如果没有毒药，poison 必须为 null。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "你是": "C",
                        "你的身份": "女巫",
                        "今晚被击杀": killed_by_wolf,
                        "是否还有解药": witch_has_antidote,
                        "是否还有毒药": witch_has_poison,
                        "可毒杀目标": witch_candidates,
                        "女巫记忆": witch_memory,
                        "公开历史": public_history[-10:],
                        "任务": "决定今晚是否用解药、是否用毒药。",
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        payload = {
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 500,
            "response_format": {"type": "json_object"},
        }
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        witch_action = json.loads(response.json()["choices"][0]["message"]["content"])

        save_choice = bool(witch_action.get("save")) and witch_has_antidote
        poison_choice = witch_action.get("poison")
        if not witch_has_poison or poison_choice not in witch_candidates:
            poison_choice = None

        if save_choice:
            saved_by_witch = True
            witch_has_antidote = False
        if poison_choice:
            poisoned_by_witch = poison_choice
            witch_has_poison = False

        witch_memory.append(
            f"第 {day} 夜：狼人击杀 {killed_by_wolf}；"
            f"女巫救人={saved_by_witch}；女巫毒人={poisoned_by_witch}"
        )

        print("女巫 Agent 输出：")
        print(json.dumps(witch_action, ensure_ascii=False, indent=2))
        print("系统执行后的女巫选择：")
        print("  是否救人：", saved_by_witch)
        print("  毒杀目标：", poisoned_by_witch)
        print("  剩余解药：", witch_has_antidote)
        print("  剩余毒药：", witch_has_poison)
        print("女巫私有记忆：", witch_memory)
    else:
        print("女巫 C 已死亡，本夜不能使用解药或毒药。")

    # 4. 结算夜晚死亡
    if saved_by_witch:
        print(f"\n夜晚结算：{killed_by_wolf} 被女巫救下，没有因狼人击杀死亡。")
    else:
        night_deaths.append(killed_by_wolf)
        print(f"\n夜晚结算：{killed_by_wolf} 被狼人击杀死亡。")

    if poisoned_by_witch and poisoned_by_witch not in night_deaths:
        night_deaths.append(poisoned_by_witch)
        print(f"夜晚结算：{poisoned_by_witch} 被女巫毒死。")

    # 去重，并从 alive_players 中移除死亡玩家。
    night_deaths = [p for p in all_players if p in night_deaths]
    for dead in night_deaths:
        if dead in alive_players:
            alive_players.remove(dead)

    public_history.append(f"第 {day} 夜晚死亡：{night_deaths}")

    print("\n天亮公布")
    print("-" * 72)
    print("昨夜死亡：", night_deaths if night_deaths else "平安夜")
    print("天亮后存活玩家：", alive_players)
    print("天亮后死亡玩家：", [p for p in all_players if p not in alive_players])

    print(f"\n第 {day} 天 白天")
    print("=" * 72)

    wolf_alive = [p for p in alive_players if roles[p] == "狼人"]
    good_alive = [p for p in alive_players if roles[p] != "狼人"]
    if not wolf_alive:
        winner = "好人阵营"
        break
    if len(wolf_alive) >= len(good_alive):
        winner = "狼人阵营"
        break

    # 5. 白天发言
    print("\n白天发言阶段")
    print("-" * 72)
    for player in all_players:
        if player not in alive_players:
            print(f"{player}（{roles[player]}）：已死亡，跳过发言。")
            continue

        private_info = []
        if player == "B":
            private_info = seer_memory
        if player == "C":
            private_info = witch_memory

        print(f"\n轮到 {player} 发言")
        print(f"课堂可见身份：{roles[player]}")
        print("该玩家私有信息：", private_info if private_info else "无")

        messages = [
            {
                "role": "system",
                "content": (
                    "你是狼人杀玩家。请根据公开历史和自己的身份发言。"
                    "必须输出 JSON，字段为 speech、suspect、reason。"
                    "不要说出不该公开的系统信息。"
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

    # 6. 白天投票
    print("\n投票阶段")
    print("-" * 72)
    votes = {}

    for player in all_players:
        if player not in alive_players:
            print(f"{player}（{roles[player]}）：已死亡，跳过投票。")
            continue

        candidates = [p for p in alive_players if p != player]
        private_info = []
        if player == "B":
            private_info = seer_memory
        if player == "C":
            private_info = witch_memory

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

    # 7. 裁判检查胜负
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
    wolf_alive = [p for p in alive_players if roles[p] == "狼人"]
    good_alive = [p for p in alive_players if roles[p] != "狼人"]
    winner = "狼人阵营" if len(wolf_alive) >= len(good_alive) else "好人阵营"

print("\n" + "=" * 72)
print("游戏结束")
print("最终身份复盘：")
print(json.dumps(roles, ensure_ascii=False, indent=2))
print("最终存活玩家：", alive_players)
print("最终死亡玩家：", [p for p in all_players if p not in alive_players])
print("女巫最终资源：解药", witch_has_antidote, "毒药", witch_has_poison)
print("预言家最终记忆：", seer_memory)
print("女巫最终记忆：", witch_memory)
print("最终公开历史：")
print(json.dumps(public_history, ensure_ascii=False, indent=2))
print(f"\n系统裁判输出：{winner}获胜。")
