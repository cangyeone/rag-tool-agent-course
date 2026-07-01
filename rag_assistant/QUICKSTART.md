# 🚀 快速启动指南 - 新版界面

## ⚡ 30 秒快速启动

### macOS / Linux

```bash
cd RAG_问答机器人
bash start_new.sh
```

### Windows

```powershell
cd RAG_问答机器人
.\start_new.ps1
```

**完成！** 浏览器会自动打开 http://localhost:8501

---

## 📋 首次使用步骤

### 1️ 检查环境（自动完成）

启动脚本会自动检查：
- ✅ Python 是否安装
- ✅ 依赖包是否完整
- ✅ 索引是否已构建

如果缺少依赖或索引，会自动安装/构建。

### 2️⃣ 配置 API Key（仅需一次）

在 `RAG_问答机器人` 目录下创建 `.env` 文件：

```bash
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

**获取 API Key**：访问 https://platform.deepseek.com

### 3️⃣ 开始对话

1. 点击左侧 **"➕ 新建对话"**
2. 在底部输入框输入问题
3. 点击 **"📤 发送"** 或按回车
4. 等待 AI 回答

---

## 💡 常用操作

### 管理对话

| 操作 | 方法 |
|------|------|
| 新建对话 | 点击左侧 "➕ 新建对话" |
| 切换对话 | 点击左侧对话列表 |
| 删除对话 | 点击对话右侧 "🗑️" |
| 查看历史 | 刷新页面，对话自动加载 |

### 调整检索

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| 召回片段数 | 从知识库检索的片段数量 | 5 |
| 向量权重 | 语义相似度 vs 关键词匹配 | 0.75 |

**提示**：
- 向量权重越高 → 越注重语义理解
- 向量权重越低 → 越注重关键词匹配

---

## ❓ 常见问题速查

### Q1: 启动后无法访问网页？

**解决**：
1. 检查终端是否有报错
2. 确认 8501 端口未被占用
3. 尝试手动访问：http://localhost:8501

### Q2: 提示 "还没有构建索引"？

**解决**：
```bash
python build_index.py
```

### Q3: DeepSeek API 调用失败？

**解决**：
1. 检查 `.env` 文件是否存在
2. 确认 `DEEPSEEK_API_KEY` 是否正确
3. 测试网络是否能访问 DeepSeek API

### Q4: 对话历史丢失？

**解决**：
1. 检查 `sessions/` 目录是否存在
2. 确认有写入权限
3. 不要手动删除 JSON 文件

### Q5: 检索结果不准确？

**解决**：
1. 增加"召回片段数"到 6-8
2. 调整"向量权重"（试试 0.5 或 0.9）
3. 优化问题表述，使用更具体的关键词

---

##  示例问题

复制以下问题到输入框快速体验：

```
RAG 的切片和 overlap 有什么作用？
```

```
在线模型 API 如何接入 RAG 助手？
```

```
工具调用和 RAG 有什么区别？
```

```
这两天课程里，RAG 问答机器人一般由哪些步骤组成？
```

---

## 🔧 高级用法

### 手动启动（不使用脚本）

```bash
cd RAG_问答机器人
python app_new.py
```

### 指定端口

```bash
streamlit run app_new.py --server.port 8502
```

### 远程访问

```bash
streamlit run app_new.py --server.address 0.0.0.0 --server.port 8501
```

然后访问：`http://你的IP:8501`

### 备份对话历史

```bash
# macOS/Linux
tar -czf sessions_backup.tar.gz sessions/

# Windows PowerShell
Compress-Archive -Path sessions -DestinationPath sessions_backup.zip
```

### 恢复对话历史

```bash
# macOS/Linux
tar -xzf sessions_backup.tar.gz

# Windows PowerShell
Expand-Archive -Path sessions_backup.zip -DestinationPath .
```

---

## 📚 相关文档

- **详细使用说明**：[README_NEW.md](README_NEW.md)
- **界面布局设计**：[界面布局说明.md](界面布局说明.md)
- **完善总结**：[完善总结.md](完善总结.md)
- **旧版界面**：[app.py](app.py)（单会话版本）

---

## 🆘 获取帮助

1. 查看本文档的"常见问题速查"部分
2. 阅读 [README_NEW.md](README_NEW.md) 完整说明
3. 联系课程助教或技术支持

---

## ✨ 下一步

- [ ] 尝试创建多个对话，体验多会话管理
- [ ] 调整检索参数，观察结果变化
- [ ] 导出重要对话作为学习笔记
- [ ] 探索更多示例问题

**祝您使用愉快！** 🎉