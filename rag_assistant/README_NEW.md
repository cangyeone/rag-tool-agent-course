# RAG 问答机器人 - 新版界面（多会话支持）

## 📋 功能特性

### ✨ 新版界面亮点

1. **多对话管理**
   - 左侧边栏显示所有历史对话
   - 支持新建、切换、删除对话
   - 自动保存对话历史到本地

2. **清晰的界面布局**
   - **左侧**：多个对话列表
   - **右侧上方**：知识库选择 + 检索设置
   - **右侧下方**：聊天输入框 + 对话展示

3. **友好的用户体验**
   - 对话气泡样式区分用户和 AI 消息
   - 实时保存，刷新页面不丢失
   - 快捷提示问题

##  快速开始

### macOS / Linux

```bash
cd RAG_问答机器人
bash start_new.sh
```

或直接运行：

```bash
python app_new.py
```

### Windows

```powershell
cd RAG_问答机器人
.\start_new.ps1
```

或在 PowerShell 中运行：

```powershell
python app_new.py
```

## 📖 使用说明

### 首次运行

1. 脚本会自动检查并安装依赖
2. 如果未构建索引，会自动运行 `build_index.py`
3. 启动后访问 http://localhost:8501

### 日常使用

1. **新建对话**：点击左侧"➕ 新建对话"按钮
2. **切换对话**：点击左侧对话列表中的任意对话
3. **删除对话**：点击对话右侧的"🗑️"按钮
4. **提问**：在底部输入框输入问题，点击"📤 发送"或按回车

### 检索设置

- **召回片段数**：控制从知识库检索的片段数量（2-8）
- **向量权重**：调节语义相似度和关键词匹配的平衡（0.0-1.0）
  - 越高越偏语义相似
  - 越低越偏关键词命中

## 📁 文件说明

```
RAG_问答机器人/
├── app_new.py              # 新版 Web 应用（多会话支持）
├── app.py                  # 旧版 Web 应用（单会话）
├── rag_core.py             # RAG 核心逻辑
├── build_index.py          # 索引构建脚本
── storage/                # 索引数据存储
│   ├── chunks.json         # 文本片段
│   └── embeddings.npy      # 向量数据
├── sessions/               # 会话历史存储（自动生成）
│   ├── session_xxx.json    # 对话 1
│   └── session_yyy.json    # 对话 2
├── start_new.sh            # macOS/Linux 启动脚本
── start_new.ps1           # Windows 启动脚本
└── README_NEW.md           # 本文档
```

## 💡 示例问题

- RAG 的切片和 overlap 有什么作用？
- 可视化工具 如何接入 DeepSeek？
- 工具调用和 RAG 有什么区别？
- 这两天课程里，RAG 问答机器人一般由哪些步骤组成？

## 🔧 技术栈

- **前端框架**：Streamlit
- **向量数据库**：FAISS
- **嵌入模型**：BGE-m3
- **语言模型**：DeepSeek API
- **分词工具**：jieba

## ️ 环境要求

- Python 3.8+
- 依赖包：
  ```bash
  pip install streamlit numpy faiss-cpu sentence-transformers requests rich jieba
  ```

## 📝 注意事项

1. **API 密钥**：需要在 `.env` 文件中设置 `DEEPSEEK_API_KEY`
2. **会话存储**：所有对话历史保存在 `sessions/` 目录，可手动备份或删除
3. **索引更新**：如果修改了知识库文档，需要重新运行 `build_index.py`

## 🆚 新旧版本对比

| 特性 | 旧版 (app.py) | 新版 (app_new.py) |
|------|--------------|------------------|
| 多对话管理 | ❌ | ✅ |
| 对话历史持久化 |  | ✅ |
| 界面布局 | 侧边栏设置 | 左右分栏 |
| 会话标题自动生成 | ❌ | ✅ |
| 删除对话功能 | ❌ | ✅ |

## 🐛 常见问题

### Q: 启动后无法访问网页？
A: 检查防火墙设置，确保 8501 端口未被占用。

### Q: 对话历史丢失？
A: 检查 `sessions/` 目录是否存在且有写入权限。

### Q: 检索结果不准确？
A: 调整"向量权重"滑块，或增加"召回片段数"。

### Q: DeepSeek API 调用失败？
A: 检查 `.env` 文件中的 `DEEPSEEK_API_KEY` 是否正确配置。

## 📞 技术支持

如有问题，请联系课程助教或查看主 README.md。