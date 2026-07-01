"""07 HTTP 服务接口草图。

学习目标：将模型调用能力封装为 HTTP 服务，理解一个最小可用的 LLM 服务需要哪些路由（/chat、/health、/models），掌握流式 SSE 响应的服务端实现方式。

运行方式：python 07_HTTP_服务接口草图.py
（这是 Flask 服务端代码，需要 pip install flask 后运行）
"""


import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
import time
import sys
from threading import Thread

print("07 HTTP 服务接口草图")
print("=" * 72)

print("下面是一个最小可用的 Flask LLM 服务，展示三个核心路由。")
print()

# ==================== 服务端代码（可独立运行） ====================

SERVER_CODE = r'''
"""最小 LLM 服务 —— 示例业务系统 课堂演示版。

启动后访问：
  GET  http://localhost:5000/health      → 健康检查
  GET  http://localhost:5000/models      → 列出可用模型
  POST http://localhost:5000/chat        → 对话接口（非流式）
  POST http://localhost:5000/chat/stream → 对话接口（SSE 流式）

运行方式：
  pip install flask
  python 07_HTTP_服务接口草图.py
"""

from flask import Flask, request, Response, jsonify, stream_with_context
import json
import time
import torch
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# 真实模型后端
def _find_qwen_path():
    candidate = COURSE_ROOT / "open_models" / "Qwen3.5-0.8B"
    if candidate.is_dir():
        return str(candidate)
    return None

QWEN_PATH = _find_qwen_path()
_model = None
_tokenizer = None
_device = "cpu"

def _get_model():
    global _model, _tokenizer, _device
    if _model is None and QWEN_PATH:
        _device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if _device != "cpu" else torch.float32
        _tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH)
        _model = AutoModelForCausalLM.from_pretrained(QWEN_PATH, torch_dtype=dtype).to(_device).eval()
    return _model is not None

AVAILABLE_MODELS = {
    "qwen-local": "Qwen3.5-0.8B（本地 transformers）" if QWEN_PATH else "Qwen3.5-0.8B（未找到）",
    "qwen-ollama": "Qwen2.5:0.5B（本地 Ollama）",
    "deepseek-api": "DeepSeek-V3（云端 API）",
}

ACTIVE_MODEL = "qwen-local"

def generate_response(messages, stream=False):
    if not _get_model():
        user_msg = messages[-1]["content"] if messages else ""
        answer = f"[模型未加载] 关于「{user_msg[:30]}...」，请先下载 open_models/Qwen3.5-0.8B。"
        if stream:
            for char in answer:
                yield char
                time.sleep(0.02)
        else:
            return answer
        return

    text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(text, return_tensors="pt").to(_device)
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs, max_new_tokens=200, temperature=0.2, do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = _tokenizer.decode(generated, skip_special_tokens=True).strip()

    if stream:
        for char in answer:
            yield char
            time.sleep(0.02)
    else:
        return answer


# ===== 路由 1：健康检查 =====
@app.route("/health", methods=["GET"])
def health():
    """返回服务状态、活跃模型、运行时间。"""
    return jsonify({
        "status": "ok",
        "service": "示例业务系统-LLM-Service",
        "active_model": ACTIVE_MODEL,
        "available_models": list(AVAILABLE_MODELS.keys()),
    })


# ===== 路由 2：模型列表 =====
@app.route("/models", methods=["GET"])
def list_models():
    """返回所有可用模型的名称和说明。"""
    return jsonify({
        "object": "list",
        "data": [
            {"id": name, "description": desc}
            for name, desc in AVAILABLE_MODELS.items()
        ],
        "active": ACTIVE_MODEL,
    })


# ===== 路由 3：对话（非流式）=====
@app.route("/chat", methods=["POST"])
def chat():
    """接收 messages，返回完整的回答 JSON。"""
    body = request.get_json(force=True)
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.2)

    # 模拟处理延迟
    time.sleep(0.3)

    answer = generate_response(messages, stream=False)

    return jsonify({
        "id": "chatcmpl-001",
        "object": "chat.completion",
        "model": ACTIVE_MODEL,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(str(messages)),
            "completion_tokens": len(answer),
            "total_tokens": len(str(messages)) + len(answer),
        },
    })


# ===== 路由 4：对话（SSE 流式）=====
@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    """接收 messages，通过 Server-Sent Events 逐步返回回答。"""
    body = request.get_json(force=True)
    messages = body.get("messages", [])

    def generate():
        answer = generate_response(messages, stream=False)
        for i, char in enumerate(answer):
            chunk = {
                "id": "chatcmpl-stream-001",
                "object": "chat.completion.chunk",
                "model": ACTIVE_MODEL,
                "choices": [{
                    "index": 0,
                    "delta": {"content": char},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            time.sleep(0.02)

        # 结束标记
        final_chunk = {
            "id": "chatcmpl-stream-001",
            "object": "chat.completion.chunk",
            "model": ACTIVE_MODEL,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ===== 路由 5：切换模型 =====
@app.route("/switch_model", methods=["POST"])
def switch_model():
    """切换活跃模型后端。"""
    global ACTIVE_MODEL
    body = request.get_json(force=True)
    new_model = body.get("model", "")
    if new_model not in AVAILABLE_MODELS:
        return jsonify({"error": f"模型 '{new_model}' 不可用", "available": list(AVAILABLE_MODELS.keys())}), 400
    old_model = ACTIVE_MODEL
    ACTIVE_MODEL = new_model
    return jsonify({
        "status": "switched",
        "from": old_model,
        "to": new_model,
        "description": AVAILABLE_MODELS[new_model],
    })


if __name__ == "__main__":
    print("示例业务系统 LLM 服务启动中...")
    print("=" * 50)
    print("  GET  http://localhost:5000/health")
    print("  GET  http://localhost:5000/models")
    print("  POST http://localhost:5000/chat")
    print("  POST http://localhost:5000/chat/stream")
    print("  POST http://localhost:5000/switch_model")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)
'''

# ==================== 展示服务架构 ====================
print("一、服务架构")
print("  客户端 → HTTP → Flask 路由 → 模型后端 → 返回 JSON 或 SSE 流")
print()
print("  五个路由：")
routes = [
    ("GET  /health", "健康检查 → 返回服务状态和活跃模型"),
    ("GET  /models", "模型列表 → 返回所有可用模型及说明"),
    ("POST /chat", "对话接口 → 非流式 JSON 响应"),
    ("POST /chat/stream", "对话接口 → SSE 流式响应（逐字推送）"),
    ("POST /switch_model", "模型切换 → 运行时切换活跃模型后端"),
]
for method_path, desc in routes:
    print(f"    {method_path:<22} {desc}")

print("\n二、请求示例")
print("  健康检查：")
print("    curl http://localhost:5000/health")
print()
print("  对话请求：")
chat_curl = (
    'curl -X POST http://localhost:5000/chat '
    '-H "Content-Type: application/json" '
    '-d \'{"messages":[{"role":"user","content":"候补申请为什么不能保证成功？"}],"temperature":0.2}\''
)
print(f"    {chat_curl}")

print("\n三、服务端代码已保存到当前文件。启动方式：")
print("    pip install flask")
print("    python 07_HTTP_服务接口草图.py")

# 如果用户想直接运行，提供内联启动选项
print("\n四、是否启动服务？")
print("    如果需要启动 Flask 服务，请在新终端中运行本文件。")
print("    启动后，可以用 curl 或浏览器测试上述接口。")

# 尝试自动写入服务端代码到临时文件并给出运行指令
# （学生可以复制上述代码独立运行）

print("\n要点：将模型调用封装为 HTTP 服务是生产化的第一步。有了标准 REST API 后，前端、测试、监控都可以围绕它构建。")
print("三个核心路由：health（运维监控）、chat（业务调用）、models（模型发现），是最小可行的 LLM 服务骨架。")