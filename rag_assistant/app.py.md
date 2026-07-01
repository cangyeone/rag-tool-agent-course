# app：代码说明

> 对应代码：`RAG_问答机器人/app.py`

## 1. 这段代码对应的行业需求

业务知识库问答：围绕订单服务规则、安全制度、应急预案、巡检记录和培训材料做可追溯检索。

## 2. 课堂目标

说明 `app.py` 的课堂代码。

## 3. 讲解顺序

- 先用业务问题开场：业务现场为什么需要这段能力。
- 再讲输入是什么：用户问题、规则资料、图片/记录、工具参数或环境信息。
- 然后讲处理过程：每一步生成什么中间结果。
- 最后讲输出怎么看：哪些结果可信，哪些地方要人工确认或回到官方系统。

## 4. 代码结构

1. 先写脚本顶部说明：这段代码解决哪个业务系统问题，运行后应该看到什么。
2. 再写 import 和常量：优先使用标准库，路径使用 `Path(__file__).resolve().parent`。
3. 准备课堂模拟数据：例如订单服务规则、服务点、订单编号、巡检记录、工单状态。
4. 按函数逐个生成：`load_rag_index`。
5. 最后写 `if __name__ == "__main__":` 入口，串起课堂演示流程。
6. 运行一次，确认输出里有中间结果、最终结果和可修改点。

## 5. 主要函数或代码块

- `load_rag_index`

## 6. 可修改点

- 把示例问题换成在线客服、安全检查、设备运维或培训答疑中的真实表达。
- 改一条模拟规则或模拟工具返回值，观察最终输出怎么变化。
- 故意制造一个缺字段、空结果或接口不可用的情况，说明兜底逻辑。
- 补充一条测试用例，检查输出是否仍然符合业务边界。

## 7. 运行方式

在脚本所在目录运行：

```bash
python app.py
```

涉及 DeepSeek 时，只使用环境变量 `DEEPSEEK_API_KEY`，不要把 Key 写入文件。

## 8. 完整参考代码

```python
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from rag_core import DEFAULT_MODEL_PATH, LocalRAGIndex, ask_deepseek, build_context


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_DIR = Path(__file__).resolve().parent
STORAGE_DIR = PROJECT_DIR / "storage"
MODEL_PATH = os.getenv("BGE_M3_MODEL_PATH", DEFAULT_MODEL_PATH)


@st.cache_resource(show_spinner="正在加载 BGE-m3 和本地索引，第一次会稍慢...")
def load_rag_index() -> LocalRAGIndex:
    if not (STORAGE_DIR / "chunks.json").exists() or not (STORAGE_DIR / "embeddings.npy").exists():
        raise FileNotFoundError(
            "还没有构建索引。请先运行：python build_index.py"
        )
    return LocalRAGIndex.load(STORAGE_DIR, model_path=MODEL_PATH)


st.set_page_config(
    page_title="RAG 工具公开课 RAG 问答机器人",
    page_icon="🔎",
    layout="wide",
)

st.title("RAG 工具公开课 RAG 问答机器人")
st.caption("基于本地课程资料检索，再调用 DeepSeek 生成回答。")

with st.sidebar:
    st.subheader("检索设置")
    top_k = st.slider("召回片段数", min_value=2, max_value=8, value=5, step=1)
    vector_weight = st.slider(
        "向量检索权重",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="越高越偏语义相似，越低越偏关键词命中。",
    )
    model_name = st.selectbox("DeepSeek 模型", ["deepseek-chat", "deepseek-reasoner"], index=0)
    base_url = st.text_input("DeepSeek API Base", value=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    st.divider()
    st.write("索引目录")
    st.code(str(STORAGE_DIR), language="text")
    st.write("BGE 模型")
    st.code(MODEL_PATH, language="text")


try:
    rag_index = load_rag_index()
except Exception as exc:
    st.error(str(exc))
    st.stop()

question = st.text_area(
    "输入问题",
    value="这两天课程里，RAG 问答机器人一般由哪些步骤组成？",
    height=110,
)

col_ask, col_hint = st.columns([1, 4])
with col_ask:
    ask_clicked = st.button("开始回答", type="primary", use_container_width=True)
with col_hint:
    st.info("建议问课程相关问题，例如：RAG 的切片和 overlap 有什么作用？可视化工具 如何接入 DeepSeek？工具调用和 RAG 有什么区别？")

if ask_clicked:
    if not question.strip():
        st.warning("先输入一个问题。")
        st.stop()

    with st.spinner("正在检索课程资料..."):
        results = rag_index.search(
            question.strip(),
            top_k=top_k,
            vector_weight=vector_weight,
        )
        context = build_context(results)

    with st.expander("检索到的资料片段", expanded=True):
        for item in results:
            st.markdown(
                f"**资料 {item['rank']}｜{item['source']}｜综合分 {item['score']:.3f}**"
            )
            st.write(item["text"][:900] + ("..." if len(item["text"]) > 900 else ""))
            st.divider()

    with st.spinner("正在调用 DeepSeek 生成回答..."):
        try:
            answer = ask_deepseek(
                question=question.strip(),
                context=context,
                model=model_name,
                base_url=base_url,
            )
        except Exception as exc:
            st.error(f"调用 DeepSeek 失败：{exc}")
            st.stop()

    st.subheader("回答")
    st.markdown(answer)
```