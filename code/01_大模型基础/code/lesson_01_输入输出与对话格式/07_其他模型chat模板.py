"""08 其他开源模型的 Chat 模板格式。

学习目标：了解不同开源大模型（Qwen、DeepSeek、Llama、Mistral）各自的 Chat 模板格式差异，理解为什么不能混用不同模型的模板，掌握在更换模型时正确适配 chat 格式的方法。

本脚本不需要加载模型，纯文本讲解。

运行方式：python 07_其他模型chat模板.py
"""

print("08 其他开源模型的 Chat 模板格式")
print("=" * 72)

# ========== 一、为什么不同模型有不同的格式 ==========
print("\n" + "=" * 60)
print("一、为什么不同模型有不同的 Chat 模板格式")
print("=" * 60)
print("  每个模型在训练时使用特定的对话格式，模型'认识'自己的格式标记。")
print("  用错格式 = 给模型看它不认识的语言 → 输出质量严重下降。")
print()
print("  类比：你不能用英文的标点规则去写中文文章。")
print("  不同模型的 chat 模板就像不同的'语言'，模型只懂自己训练时用的那种。")

# ========== 二、各模型格式详解 ==========
print("\n" + "=" * 60)
print("二、主流开源模型的 Chat 模板格式对比")
print("=" * 60)

# 相同的 messages 内容
system_content = "你是 通用客服助手，回答要简洁专业。"
user_content = "ORD-1001 没票了，还能怎么办？"

models = [
    {
        "name": "Qwen / Qwen-Chat (ChatML)",
        "format": "ChatML 变体，使用 <|im_start|>/<|im_end|> 标记消息边界",
        "example": (
            f"<|im_start|>system\n{system_content}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        ),
        "special_tokens": [
            ("<|im_start|>", "消息开始"),
            ("<|im_end|>", "消息结束"),
            ("<|endoftext|>", "文档结束/填充"),
        ],
        "notes": "system 在第一轮开头；换行分隔角色和内容；assistant 轮次末尾不加 <|im_end|>",
    },
    {
        "name": "DeepSeek-V3 / DeepSeek-R1",
        "format": "类 ChatML 格式，使用自定义的 BOS/EOS 标记",
        "example": (
            f"<｜begin▁of▁sentence｜>system\n{system_content}<｜end▁of▁sentence｜>"
            f"<｜begin▁of▁sentence｜>user\n{user_content}<｜end▁of▁sentence｜>"
            f"<｜begin▁of▁sentence｜>assistant\n"
        ),
        "special_tokens": [
            ("<｜begin▁of▁sentence｜>", "句子/消息开始"),
            ("<｜end▁of▁sentence｜>", "句子/消息结束"),
        ],
        "notes": "DeepSeek-R1 额外输出 <think>...</think> 包裹思考过程，在最终回答之前",
    },
    {
        "name": "Llama 3 / Llama 3.1",
        "format": "专用标记格式，使用 <|start_header_id|>/<|end_header_id|>/<|eot_id|>",
        "example": (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_content}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_content}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "special_tokens": [
            ("<|begin_of_text|>", "文本开始"),
            ("<|start_header_id|>", "消息头开始（后跟角色名）"),
            ("<|end_header_id|>", "消息头结束"),
            ("<|eot_id|>", "轮次结束"),
        ],
        "notes": "system 支持多个 system 消息（工具调用需要）；双换行分隔头部和内容；支持 tool 角色",
    },
    {
        "name": "Mistral / Mixtral",
        "format": "简化格式，使用 [INST] 标记（无 system 原生支持）",
        "example": (
            f"<s>[INST] {system_content}\n\n{user_content} [/INST]"
        ),
        "special_tokens": [
            ("<s>", "句子开始"),
            ("[INST]", "指令开始"),
            ("[/INST]", "指令结束"),
        ],
        "notes": "Mistral 早期版本不原生支持 system role，通常将 system prompt 拼入 user 消息开头",
    },
]

for model_info in models:
    print(f"\n  {'─' * 50}")
    print(f"  ■ {model_info['name']}")
    print(f"    格式：{model_info['format']}")
    print(f"    特殊标记：")
    for tok, desc in model_info["special_tokens"]:
        print(f"      {tok:<35} {desc}")
    print(f"    渲染示例：")
    for line in model_info["example"].split("\n"):
        print(f"      {line}" if line else "")
    print(f"    注意事项：{model_info['notes']}")

# ========== 三、用错模板的后果 ==========
print("\n\n" + "=" * 60)
print("三、用错模板会怎样？")
print("=" * 60)

print("  假设把 Llama 格式的模板用在 Qwen 模型上：")
wrong_template = (
    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    f"{system_content}<|eot_id|>"
    f"<|start_header_id|>user<|end_header_id|>\n\n"
    f"{user_content}<|eot_id|>"
    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
)
print(f"    输入：{repr(wrong_template[:80])}...")
print(f"    结果：Qwen 不认识 <|begin_of_text|> 等标记 → 当成普通文字处理")
print(f"          → 模型无法区分 system/user/assistant 角色")
print(f"          → 回答质量严重下降，可能出现格式混乱或不知所云")

# ========== 四、OpenAI API 格式 ==========
print("\n" + "=" * 60)
print("四、OpenAI / 兼容 API 的格式（API 层面）")
print("=" * 60)

print(f"  OpenAI Chat Completions API 只接受 messages JSON 数组：")
print(f"    [")
print(f'      {{"role": "system", "content": "{system_content}"}},')
print(f'      {{"role": "user", "content": "{user_content}"}}')
print(f"    ]")
print()
print(f"  API 内部如何转换？用户看不到。")
print(f"  OpenAI 不公开 token 级别的 chat 格式，用户只需按接口规范传 messages。")
print(f"  但使用开源模型（如 Qwen、Llama）部署服务时，仍需在服务端处理 chat template。")

# ========== 五、transformers 如何自动适配 ==========
print("\n" + "=" * 60)
print("五、transformers 的自动适配机制")
print("=" * 60)

print(f"  tokenizer.apply_chat_template(messages, ...)")
print(f"     ↓ 自动检测 tokenizer 的 chat_template 配置")
print(f"     ↓ 每个模型的 tokenizer_config.json 定义了各自的模板")
print(f"     ↓ 对用户透明：切换 tokenizer 即可自动切换格式")
print()
print(f"  示例：")
print(f"    # 使用 Qwen")
print(f"    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')")
print(f"    rendered = tokenizer.apply_chat_template(messages, ...)")
print(f"    # → 自动使用 ChatML 格式")
print()
print(f"    # 使用 Llama")
print(f"    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')")
print(f"    rendered = tokenizer.apply_chat_template(messages, ...)")
print(f"    # → 自动使用 Llama 专用格式")

# ========== 六、DeepSeek-R1 的 think 标签 ==========
print("\n" + "=" * 60)
print("六、DeepSeek-R1 的特殊性 —— 内置 think 标签")
print("=" * 60)

print(f"  DeepSeek-R1 是推理模型，输出中自动包含思考过程：")
print(f"    <think>")
print(f"    好的，用户问ORD-1001没票了还能怎么办。")
print(f"    首先我需要确认候补申请规则，然后给出替代方案...")
print(f"    根据行业规定，乘客可以...")
print(f"    </think>")
print(f"    如果ORD-1001次服务流程订单无库存，您可以：")
print(f"    1. 提交候补申请订单...")
print(f"    2. 考虑同方向其他服务编号...")
print()
print(f"  注意事项：")
print(f"    • <think> 标签是 R1 模型训练时内置的，不需要额外 prompt")
print(f"    • 生产环境需用正则清理 <think>...</think>，只展示最终回答（见 demo 07）")
print(f"    • 思考过程可用于内部审计和调试")

# ========== 七、总结 ==========
print("\n\n" + "=" * 60)
print("七、Chat 模板格式要点")
print("=" * 60)
print("  1. 不同模型有不同的 chat 模板，不能混用")
print("  2. Qwen 用 ChatML (<|im_start|>/<|im_end|>)，DeepSeek 类似但有自定义变体")
print("  3. Llama 3 用 <|start_header_id|>/<|eot_id|>，与 Qwen 完全不同")
print("  4. Mistral 用 [INST] 标记，无原生 system role 支持")
print("  5. 使用 apply_chat_template() 可以自动适配不同模型，无需手动处理")
print("  6. OpenAI API 不暴露 token 级格式，用户只需传 messages JSON 数组")
print("  7. DeepSeek-R1 特有 <think> 标签，需在生产环境清理")
print()
print("  延伸阅读：每个模型的 tokenizer_config.json 中 chat_template 字段定义了渲染规则")
print("  可以修改：查找其他模型（如 Gemma、Phi）的 chat template 文档并补充到对比表中")