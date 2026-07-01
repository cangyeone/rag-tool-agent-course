"""06 Ollama 调用兜底。

学习目标：学会用 requests 库直接调用 Ollama 本地 API 进行生成和对话，实现自动回退（fallback）机制——当 Ollama 不可用时给出明确的替代方案提示。

运行方式：python 06_Ollama_调用兜底.py
"""

import json
import requests
import time

print("06 Ollama 调用兜底")
print("=" * 72)

# Ollama 服务地址和备选列表
OLLAMA_BASE_URLS = [
    "http://localhost:11434",
    "http://127.0.0.1:11434",
]
MODEL_NAME = "qwen2.5:0.5b"
QUESTION = "请解释候补申请为什么不能保证成功"

print("一、检查 Ollama 服务是否可用")
available_url = None
for base_url in OLLAMA_BASE_URLS:
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=3)
        if resp.status_code == 200:
            available_url = base_url
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            print(f"  {base_url} ✓ 可用")
            print(f"  已安装模型：{models}")
            break
        else:
            print(f"  {base_url} ✗ HTTP {resp.status_code}")
    except requests.ConnectionError:
        print(f"  {base_url} ✗ 连接被拒绝（Ollama 未启动）")
    except requests.Timeout:
        print(f"  {base_url} ✗ 连接超时")
    except Exception as e:
        print(f"  {base_url} ✗ {type(e).__name__}: {e}")

if not available_url:
    print("\n二、Ollama 不可用 —— 启动回退策略")
    fallback_plan = [
        ("检查安装", "终端运行: ollama --version"),
        ("启动服务", "终端运行: ollama serve"),
        ("拉取模型", f"终端运行: ollama pull {MODEL_NAME}"),
        ("验证模型", f"终端运行: ollama list | grep {MODEL_NAME}"),
        ("如果仍不可用", "跳到 08_模型后端切换.py 使用在线 DeepSeek API"),
    ]
    for step, cmd in fallback_plan:
        print(f"  [{step}] {cmd}")
    print("\n  回退说明：Ollama 需要本地运行服务端。如果无法启动，课程中可以用 DeepSeek API 代替本地模型演示。")
    print("  课程演示不受影响，继续展示 Ollama API 的完整调用流程。")
else:
    print(f"\n二、调用 Ollama /api/generate 接口")
    print(f"  服务地址：{available_url}")
    print(f"  模型名称：{MODEL_NAME}")

    # ===== 方式1：/api/generate（非对话模式）=====
    print("\n  方式1：/api/generate（普通生成模式）")
    payload_generate = {
        "model": MODEL_NAME,
        "prompt": QUESTION,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 150,
        },
    }
    print(f"  请求体：{json.dumps(payload_generate, ensure_ascii=False, indent=2)}")

    try:
        resp = requests.post(
            f"{available_url}/api/generate",
            json=payload_generate,
            timeout=30,
        )
        if resp.status_code == 200:
            result = resp.json()
            print(f"\n  回答内容：{result.get('response', '')[:200]}")
            print(f"  生成耗时：{result.get('total_duration', 0) / 1e9:.2f} 秒")
            print(f"  eval 速度：{result.get('eval_count', 0) / (result.get('eval_duration', 1) / 1e9):.1f} tokens/秒")
        elif resp.status_code == 404:
            print(f"\n  404 错误：模型 '{MODEL_NAME}' 不存在。请运行 ollama pull {MODEL_NAME}")
        else:
            print(f"\n  HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"\n  调用失败：{e}")

    # ===== 方式2：/api/chat（对话模式）=====
    print(f"\n  方式2：/api/chat（对话模式）")
    payload_chat = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是 通用客服助手，回答要简洁专业。"},
            {"role": "user", "content": QUESTION},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    print(f"  请求体：{json.dumps(payload_chat, ensure_ascii=False, indent=2)}")

    try:
        resp = requests.post(
            f"{available_url}/api/chat",
            json=payload_chat,
            timeout=30,
        )
        if resp.status_code == 200:
            result = resp.json()
            content = result["message"]["content"]
            print(f"\n  回答内容：{content[:200]}")
        else:
            print(f"\n  HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"\n  调用失败：{e}")

    # ===== 方式3：流式输出 /api/generate =====
    print(f"\n  方式3：流式输出 /api/generate")
    payload_stream = {
        "model": MODEL_NAME,
        "prompt": QUESTION,
        "stream": True,
        "options": {"temperature": 0.2, "num_predict": 80},
    }
    try:
        print("  ", end="", flush=True)
        with requests.post(
            f"{available_url}/api/generate",
            json=payload_stream,
            stream=True,
            timeout=30,
        ) as resp:
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    token_text = data.get("response", "")
                    print(token_text, end="", flush=True)
                    if data.get("done"):
                        break
        print()
        print("  (流式输出结束)")
    except Exception as e:
        print(f"\n  流式调用失败：{e}")

print("\n\n三、Ollama 作为兜底方案的价值")
benefits = [
    "本机运行 → 不依赖网络、不消耗 API 额度",
    "数据不出本机 → 适合含敏感信息的企业内部文档测试",
    "接口与 OpenAI 兼容（/api/chat）→ 上层代码无需改动",
    "支持多模型切换 → 可以小模型做分类、大模型做生成",
]
for b in benefits:
    print(f"  • {b}")

print("\n要点：Ollama 把本地模型包装成 HTTP 服务，调用方式和云端 API 一致。如果 Ollama 不可用，回退到 DeepSeek API。")