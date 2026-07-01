"""
RAG 问答机器人 - 新版界面（多会话支持）

界面布局：
- 左侧：多个对话历史列表
- 右侧上方：知识库选择 + 检索设置
- 右侧下方：提问输入框 + 对话展示

运行方式：
    cd RAG_问答机器人
    python app_new.py
或启动脚本：
    bash start_macos_linux.sh
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import streamlit as st

from rag_core import DEFAULT_MODEL_PATH, LocalRAGIndex, ask_deepseek, build_context


# ── 路径配置 ──

PROJECT_DIR = Path(__file__).resolve().parent
STORAGE_DIR = PROJECT_DIR / "storage"
SESSIONS_DIR = PROJECT_DIR / "sessions"  # 会话历史存储目录
MODEL_PATH = os.getenv("BGE_M3_MODEL_PATH", DEFAULT_MODEL_PATH)

# 确保会话目录存在
SESSIONS_DIR.mkdir(exist_ok=True)


# ── 会话管理 ──

def get_session_file(session_id: str) -> Path:
    """获取会话文件路径"""
    return SESSIONS_DIR / f"{session_id}.json"


def load_session(session_id: str) -> dict:
    """加载会话数据"""
    session_file = get_session_file(session_id)
    if session_file.exists():
        with open(session_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "id": session_id,
        "title": f"新对话 {datetime.now().strftime('%H:%M')}",
        "created_at": datetime.now().isoformat(),
        "messages": [],
    }


def save_session(session_id: str, session_data: dict):
    """保存会话数据"""
    session_file = get_session_file(session_id)
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)


def list_sessions() -> list[dict]:
    """列出所有会话（按创建时间倒序）"""
    sessions = []
    for session_file in SESSIONS_DIR.glob("*.json"):
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)
                sessions.append(session_data)
        except Exception:
            continue
    # 按创建时间倒序排序
    sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return sessions


def create_new_session() -> str:
    """创建新会话，返回会话 ID"""
    session_id = f"session_{int(time.time() * 1000)}"
    session_data = {
        "id": session_id,
        "title": f"新对话 {datetime.now().strftime('%H:%M')}",
        "created_at": datetime.now().isoformat(),
        "messages": [],
    }
    save_session(session_id, session_data)
    return session_id


def delete_session(session_id: str):
    """删除会话"""
    session_file = get_session_file(session_id)
    if session_file.exists():
        session_file.unlink()


def add_message_to_session(session_id: str, role: str, content: str):
    """添加消息到会话"""
    session_data = load_session(session_id)
    session_data["messages"].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(),
    })
    # 如果是对话的第一条用户消息，更新标题
    if len([m for m in session_data["messages"] if m["role"] == "user"]) == 1:
        # 用前 20 个字符作为标题
        title = content[:20] + ("..." if len(content) > 20 else "")
        session_data["title"] = title
    save_session(session_id, session_data)


# ── RAG 索引加载 ──

@st.cache_resource(show_spinner="正在加载 BGE-m3 和本地索引，第一次会稍慢...")
def load_rag_index() -> LocalRAGIndex:
    if not (STORAGE_DIR / "chunks.json").exists() or not (STORAGE_DIR / "embeddings.npy").exists():
        raise FileNotFoundError(
            "还没有构建索引。请先运行：python build_index.py"
        )
    return LocalRAGIndex.load(STORAGE_DIR, model_path=MODEL_PATH)


# ── Streamlit 页面配置 ─

st.set_page_config(
    page_title="个人知识库问答机器人",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义 CSS 样式
st.markdown("""
<style>
    /* 隐藏默认侧边栏顶部空白 */
    [data-testid="stSidebarNav"] { display: none; }
    /* 对话气泡样式 */
    .user-message {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 4px solid #4CAF50;
    }
    /* 会话按钮样式 */
    .session-btn {
        width: 100%;
        text-align: left;
        padding: 10px;
        margin-bottom: 5px;
        border-radius: 5px;
        border: 1px solid #ddd;
        background-color: white;
        cursor: pointer;
    }
    .session-btn:hover {
        background-color: #f0f0f0;
    }
    .session-btn.active {
        background-color: #e3f2fd;
        border-color: #2196F3;
    }
</style>
""", unsafe_allow_html=True)


# ── 初始化会话状态 ──

if "current_session_id" not in st.session_state:
    # 如果有现有会话，加载第一个；否则创建新会话
    existing_sessions = list_sessions()
    if existing_sessions:
        st.session_state.current_session_id = existing_sessions[0]["id"]
    else:
        st.session_state.current_session_id = create_new_session()

if "rag_index" not in st.session_state:
    try:
        st.session_state.rag_index = load_rag_index()
    except Exception as exc:
        st.error(str(exc))
        st.stop()


# ── 主界面布局 ──

# 使用两列布局：左侧会话列表 + 右侧主内容
col_sidebar, col_main = st.columns([1, 3])

# ════════════════════════════════════════════════════════════
# 左侧：会话列表
# ═══════════════════════════════════════════════════════════

with col_sidebar:
    st.markdown("### 💬 多个对话")
    st.divider()
    # 新建对话按钮
    if st.button("➕ 新建对话", use_container_width=True, type="primary"):
        new_session_id = create_new_session()
        st.session_state.current_session_id = new_session_id
        st.rerun()
    st.divider()
    # 会话列表
    sessions = list_sessions()
    for session in sessions:
        is_active = session["id"] == st.session_state.current_session_id
        # 使用容器来创建可点击的会话项
        with st.container():
            cols = st.columns([4, 1])
            with cols[0]:
                if st.button(
                    f"💬 {session['title']}",
                    key=f"session_{session['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.current_session_id = session["id"]
                    st.rerun()
            with cols[1]:
                if st.button("🗑️", key=f"delete_{session['id']}", help="删除此对话"):
                    delete_session(session["id"])
                    # 如果删除的是当前会话，切换到第一个可用会话
                    remaining = list_sessions()
                    if remaining:
                        st.session_state.current_session_id = remaining[0]["id"]
                    else:
                        st.session_state.current_session_id = create_new_session()
                    st.rerun()
    st.divider()
    st.caption(f"共 {len(sessions)} 个对话")


# ════════════════════════════════════════════════════════════
# 右侧：主内容区域
# ═══════════════════════════════════════════════════════════

with col_main:
    # ── 上方：知识库选择 + 检索设置 ──
    st.markdown("### 📚 个人知识库问答")
    # 两列布局：知识库选择 + 检索设置
    col_kb, col_settings = st.columns([1, 1])
    with col_kb:
        st.markdown("**知识库来源**")
        st.info("📖 RAG 工具公开课课程资料")
        st.caption(f"索引位置: {STORAGE_DIR.name}")
    with col_settings:
        st.markdown("**检索设置**")
        top_k = st.slider("召回片段数", min_value=2, max_value=8, value=5, step=1, label_visibility="collapsed")
        vector_weight = st.slider(
            "向量权重",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.05,
            label_visibility="collapsed",
            help="越高越偏语义相似，越低越偏关键词命中。"
        )
    st.divider()
    # ── 中间：对话历史展示 ──
    # 加载当前会话
    current_session = load_session(st.session_state.current_session_id)
    messages = current_session.get("messages", [])
    # 显示对话历史
    chat_container = st.container(height=400, border=True)
    with chat_container:
        if not messages:
            st.markdown(
                """
                <div style='text-align: center; color: #999; padding: 50px;'>
                    👋 欢迎使用个人知识库问答机器人<br>
                    在下方输入您的问题开始对话
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            for msg in messages:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="user-message"><strong>👤 您：</strong><br>{msg["content"]}</div>',
                        unsafe_allow_html=True
                    )
                elif msg["role"] == "assistant":
                    st.markdown(
                        f'<div class="assistant-message"><strong>🤖 AI：</strong><br>{msg["content"]}</div>',
                        unsafe_allow_html=True
                    )
    # ── 下方：提问输入框 ──
    st.divider()
    # 输入框和发送按钮
    input_col, btn_col = st.columns([5, 1])
    with input_col:
        question = st.text_input(
            "提问",
            placeholder="输入您的问题...",
            label_visibility="collapsed",
            key="question_input"
        )
    with btn_col:
        send_clicked = st.button("📤 发送", use_container_width=True, type="primary")
    # 处理发送逻辑
    if send_clicked or (question and question.strip()):
        if not question.strip():
            st.warning("请输入一个问题。")
        else:
            # 添加用户消息到会话
            add_message_to_session(st.session_state.current_session_id, "user", question.strip())
            # 重新加载会话以获取最新消息
            current_session = load_session(st.session_state.current_session_id)
            messages = current_session.get("messages", [])
            # RAG 检索
            with st.spinner("🔍 正在检索知识库..."):
                try:
                    results = st.session_state.rag_index.search(
                        question.strip(),
                        top_k=top_k,
                        vector_weight=vector_weight,
                    )
                    context = build_context(results)
                except Exception as exc:
                    st.error(f"检索失败：{exc}")
                    st.stop()
            # 调用 DeepSeek 生成回答
            with st.spinner(" 正在生成回答..."):
                try:
                    answer = ask_deepseek(
                        question=question.strip(),
                        context=context,
                        model="deepseek-chat",
                        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                    )
                except Exception as exc:
                    st.error(f"生成回答失败：{exc}")
                    st.stop()
            # 添加 AI 回答到会话
            add_message_to_session(st.session_state.current_session_id, "assistant", answer)
            # 刷新页面以显示新消息
            st.rerun()
    # 快捷提示
    with st.expander("💡 示例问题", expanded=False):
        st.markdown(
            """
            - RAG 的切片和 overlap 有什么作用？
            - 可视化工具 如何接入 DeepSeek？
            - 工具调用和 RAG 有什么区别？
            - 这两天课程里，RAG 问答机器人一般由哪些步骤组成？
            """
        )