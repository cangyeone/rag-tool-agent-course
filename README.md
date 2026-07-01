# RAG 与工具使用公开课

这是一套以 **Markdown 教程 + 可运行 Python 脚本** 为主的公开课，覆盖 RAG、工具调用、Router、Agent、上下文工程和多轮对话。

仓库已脱敏：不包含行业专有信息、不包含真实业务数据、不包含 API Key，也不依赖任何特定平台。

推荐学员使用 **OpenCode** 学习：先读 Markdown，再让 OpenCode 根据 Markdown 说明解释、实现、修改和运行脚本。

## 目录

| 目录 | 内容 | 使用方式 |
|---|---|---|
| `lessons/` | Markdown 主讲义 | 按章节阅读，建立知识地图 |
| `code/` | 课堂脚本与脚本任务书 | 每个 `.py` 旁边都有同名 `.md`，可让 OpenCode 按说明实现或修改 |
| `rag_assistant/` | 简化 RAG 问答机器人 | 用自己的公开文档构建个人知识库 |
| `personal_kb/` | 个人知识库模块 | 学习解析、索引、检索、问答封装 |
| `.env.example` | 环境变量模板 | 复制为 `.env` 后填入自己的 Key，禁止提交真实 Key |

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

然后从 `lessons/README.md` 开始。

## 用 OpenCode 的推荐方式

1. 用 OpenCode 打开仓库根目录。
2. 阅读 `lessons/README.md` 和某一章 Markdown。
3. 打开对应 `code/` 目录中的脚本和同名 `.md`。
4. 对 OpenCode 说：

```text
请根据这个 Markdown 任务说明，重新实现同名 Python 脚本。
要求：代码简单、顺序清楚、注释详细、不要写入任何 API Key。
```

5. 运行脚本，观察输出。
6. 再让 OpenCode 修改一个参数或功能，比较前后差异。

## 安全说明

- 所有在线模型 Key 都通过环境变量读取。
- 不要提交 `.env` 文件。
- 不要上传真实业务资料、用户数据或未授权文档。
- 本仓库中的示例数据都是通用教学数据。

## 许可证

本课程采用 [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) 许可协议发布。

## 学习目标

完成课程后，学员应该能够：

- 搭建一个最小 RAG 知识库。
- 理解关键词检索、向量检索、混合检索和 RRF。
- 编写简单工具函数，并让模型选择工具。
- 设计工具路由，避免一次性注入过多 schema。
- 构建一个简化 Agent，并处理上下文和记忆。
- 使用 OpenCode 根据 Markdown 任务书修改和扩展代码。
