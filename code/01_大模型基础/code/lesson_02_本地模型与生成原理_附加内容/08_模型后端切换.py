"""08 模型后端切换。

学习目标：实现一个 BackendRouter 类，根据配置和可用性自动选择模型后端（transformers → Ollama → DeepSeek），理解多级回退（fallback）的工程模式。

运行方式：python 08_模型后端切换.py
"""


import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import os
import json
import time
from pathlib import Path
COURSE_ROOT = Path.cwd().resolve()
if not (COURSE_ROOT / "code").is_dir():
    raise SystemExit("请先进入 rag-tool-agent-course 课程根目录后再运行本脚本。")
from typing import Optional, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("08 模型后端切换")
print("=" * 72)


def _find_qwen_path() -> Optional[Path]:
    candidate = COURSE_ROOT / "open_models" / "Qwen3.5-0.8B"
    if candidate.is_dir():
        return candidate
    return None


# ==================== 后端抽象 ====================
class BackendRouter:
    """模型后端路由器：按优先级尝试多个后端，自动回退。"""

    def __init__(self):
        self.backends: List[Tuple[str, Dict]] = []
        self.active_backend: Optional[str] = None
        self.call_history: List[Dict] = []
        self._qwen_model: Optional[AutoModelForCausalLM] = None
        self._qwen_tokenizer: Optional[AutoTokenizer] = None
        self._qwen_device: str = "cpu"

    def register(self, name: str, config: Dict, priority: int):
        """注册一个后端。priority 越小越优先。"""
        self.backends.append((name, config, priority))
        self.backends.sort(key=lambda x: x[2])  # 按优先级排序

    def check_available(self, name: str) -> Tuple[bool, str]:
        """检查指定后端是否可用。"""
        config = {b[0]: b[1] for b in self.backends}.get(name, {})
        backend_type = config.get("type", "")

        if backend_type == "transformers":
            try:
                import torch
                from pathlib import Path
                model_path = Path(config.get("model_path", ""))
                if model_path.exists():
                    return True, f"模型文件夹存在: {model_path}"
                return False, f"模型文件夹不存在: {model_path}"
            except ImportError:
                return False, "transformers/torch 未安装"

        elif backend_type == "ollama":
            try:
                import requests
                base_url = config.get("base_url", "http://localhost:11434")
                resp = requests.get(f"{base_url}/api/tags", timeout=3)
                if resp.status_code == 200:
                    models = [m["name"] for m in resp.json().get("models", [])]
                    target = config.get("model_name", "")
                    if any(target in m for m in models):
                        return True, f"Ollama 服务运行中，模型 {target} 已安装"
                    return False, f"Ollama 运行中但模型 {target} 未安装"
                return False, f"Ollama 响应异常: HTTP {resp.status_code}"
            except ImportError:
                return False, "requests 未安装"
            except Exception as e:
                return False, f"Ollama 不可用: {e}"

        elif backend_type == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                return False, "环境变量 DEEPSEEK_API_KEY 未设置"
            return True, "API Key 已配置"

        elif backend_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return False, "环境变量 OPENAI_API_KEY 未设置"
            return True, "API Key 已配置"

        return False, f"未知后端类型: {backend_type}"

    def select_backend(self) -> str:
        """按优先级检查所有后端，返回第一个可用的。"""
        results = []
        for name, config, priority in self.backends:
            available, reason = self.check_available(name)
            result = {
                "backend": name,
                "priority": priority,
                "type": config.get("type", ""),
                "available": available,
                "reason": reason,
            }
            results.append(result)
            status = "✓" if available else "✗"
            print(f"  P{priority} [{status}] {name:<15} ({config.get('type', '')}) — {reason}")

        # 选出第一个可用的
        for r in results:
            if r["available"]:
                self.active_backend = r["backend"]
                return r["backend"]

        self.active_backend = None
        return None

    def _load_qwen(self, model_path: str) -> None:
        if self._qwen_model is not None:
            return
        if torch.cuda.is_available():
            self._qwen_device = "cuda"
        elif torch.backends.mps.is_available():
            self._qwen_device = "mps"
        dtype = torch.float16 if self._qwen_device in ("cuda", "mps") else torch.float32
        self._qwen_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._qwen_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype,
        ).to(self._qwen_device).eval()

    def chat(self, messages: List[Dict], temperature: float = 0.2) -> Dict:
        """通过活跃后端发送对话请求。"""
        if not self.active_backend:
            raise RuntimeError("没有可用的后端，请先调用 select_backend()")

        config = {b[0]: b[1] for b in self.backends}[self.active_backend]
        backend_type = config.get("type", "")
        start_time = time.time()

        if backend_type == "transformers":
            model_path = config.get("model_path", "")
            self._load_qwen(model_path)
            text = self._qwen_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self._qwen_tokenizer(text, return_tensors="pt").to(self._qwen_device)
            with torch.no_grad():
                output_ids = self._qwen_model.generate(
                    **inputs, max_new_tokens=256,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    pad_token_id=self._qwen_tokenizer.eos_token_id,
                )
            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            content = self._qwen_tokenizer.decode(generated, skip_special_tokens=True).strip()
            latency_ms = int((time.time() - start_time) * 1000)

        elif backend_type in ("deepseek", "openai"):
            import requests
            api_key = os.getenv(
                "DEEPSEEK_API_KEY" if backend_type == "deepseek" else "OPENAI_API_KEY"
            )
            model_name = config.get("model_name", "deepseek-v4-flash")
            api_base = config.get("api_base", "https://api.deepseek.com")
            url = api_base.rstrip("/") + "/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            latency_ms = int((time.time() - start_time) * 1000)

        else:
            user_content = messages[-1]["content"] if messages else ""
            backend_responses = {
                "ollama": f"[Ollama 本地] 针对「{user_content[:20]}...」的咨询，候补申请按排队顺序兑现，不能保证成功。",
            }
            content = backend_responses.get(backend_type, "[兜底] 当前后端不支持直接推理。")
            latency_ms = int((time.time() - start_time) * 1000)

        resp = {
            "content": content,
            "backend": self.active_backend,
            "latency_ms": latency_ms,
            "messages": messages,
            "temperature": temperature,
            "timestamp": time.time(),
        }
        self.call_history.append(resp)
        return resp


# ==================== 课堂演示 ====================

print("一、配置多后端")
router = BackendRouter()

# 注册四个后端，优先级从小到大
router.register(
    "transformers", {
        "type": "transformers",
        "model_path": str(qwen_path) if (qwen_path := _find_qwen_path()) else str(COURSE_ROOT / "open_models" / "Qwen3.5-0.8B"),
        "description": "本地 transformers 直接加载 Qwen 模型",
        "适合": "离线演示、看清模型原理",
    },
    priority=1,
)
router.register(
    "ollama", {
        "type": "ollama",
        "base_url": "http://localhost:11434",
        "model_name": "qwen2.5:0.5b",
        "description": "本地 Ollama 服务化调用",
        "适合": "本地服务演示、API 兼容测试",
    },
    priority=2,
)
router.register(
    "deepseek", {
        "type": "deepseek",
        "api_base": "https://api.deepseek.com",
        "model_name": "deepseek-v4-flash",
        "description": "DeepSeek 云端 API",
        "适合": "稳定线上服务、生产环境",
    },
    priority=3,
)
router.register(
    "openai", {
        "type": "openai",
        "api_base": "https://api.openai.com/v1",
        "model_name": "gpt-4o-mini",
        "description": "OpenAI 云端 API（备选）",
        "适合": "多供应商容灾",
    },
    priority=4,
)

print(f"  已注册 {len(router.backends)} 个后端")

print("\n二、可用性检查与自动选择")
selected = router.select_backend()

if selected:
    print(f"\n  → 选中后端：{selected}")
    config = {b[0]: b[1] for b in router.backends}[selected]
    print(f"    类型：{config['type']}")
    print(f"    说明：{config['description']}")
    print(f"    适用：{config['适合']}")
else:
    print("\n  → 所有后端均不可用！")
    print("    设置指引：")
    print("    1. transformers: 下载 Qwen3.5-0.8B 到 open_models/")
    print("    2. Ollama: 运行 ollama serve && ollama pull qwen2.5:0.5b")
    print("    3. DeepSeek: export DEEPSEEK_API_KEY=your_api_key_here")
    print("    4. OpenAI: export OPENAI_API_KEY=your_api_key_here")

print("\n三、发送对话请求")
if selected:
    messages = [
        {"role": "system", "content": "你是 通用客服助手，回答要简洁专业。"},
        {"role": "user", "content": "候补申请为什么不能保证成功？"},
    ]
    response = router.chat(messages, temperature=0.2)
    print(f"  后端：{response['backend']}")
    print(f"  延迟：{response['latency_ms']}ms")
    print(f"  回答：{response['content']}")

    # 再发一轮
    messages.append({"role": "assistant", "content": response["content"]})
    messages.append({"role": "user", "content": "那候补申请截止时间是什么？"})
    response2 = router.chat(messages, temperature=0.2)
    print(f"\n  第二轮 ({response2['backend']})：{response2['content']}")

print("\n\n四、调用历史")
for i, call in enumerate(router.call_history, 1):
    user_text = call["messages"][-1]["content"] if call["messages"] else ""
    print(f"  {i}. [{call['backend']}] {call['latency_ms']}ms → {user_text[:40]}...")

print("\n\n五、后端切换策略总结")
strategies = [
    ("P1 本地优先", "transformers → Ollama → DeepSeek → OpenAI", "离线场景、数据不出本机"),
    ("P1 速度优先", "本地 Ollama（如果已有模型）→ DeepSeek", "低延迟要求"),
    ("P1 质量优先", "DeepSeek → OpenAI → drop", "生产客服，需要最佳回答质量"),
    ("多级容灾", "主后端 → 副后端 → 末级兜底", "高可用场景，任一后端故障不影响服务"),
]
for strategy, chain, scene in strategies:
    print(f"  {strategy}: {chain}")
    print(f"    适用：{scene}")

print("\n要点：BackendRouter 把后端选择逻辑集中管理，业务代码不用关心用的哪个模型。新增后端只需 register 一行。")
print("生产环境中，可用性检查应该包括健康探针、延迟测试和错误率统计。")