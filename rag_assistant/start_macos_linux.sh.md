# start macos linux：代码说明

> 对应代码：`RAG_问答机器人/start_macos_linux.sh`

## 1. 这段代码对应的行业需求

业务知识库问答：围绕订单服务规则、安全制度、应急预案、巡检记录和培训材料做可追溯检索。

## 2. 课堂目标

macOS / Linux 启动脚本

## 3. 讲解顺序

- 先用业务问题开场：业务现场为什么需要这段能力。
- 再讲输入是什么：用户问题、规则资料、图片/记录、工具参数或环境信息。
- 然后讲处理过程：每一步生成什么中间结果。
- 最后讲输出怎么看：哪些结果可信，哪些地方要人工确认或回到官方系统。

## 4. 代码结构

1. 先写 `#!/usr/bin/env bash` 和 `set -euo pipefail`，让脚本失败时更清楚。
2. 写路径变量：只用当前脚本所在目录或相对路径。
3. 把安装、启动、停止、状态、日志等动作拆成函数。
4. 在每个关键命令前打印说明，展示当前执行步骤。
5. 最后写参数分发逻辑，例如 `install/start/stop/status/logs`。

## 5. 主要函数或代码块

- 这个脚本没有明显函数拆分，可以按代码段讲解。

## 6. 可修改点

- 把示例问题换成在线客服、安全检查、设备运维或培训答疑中的真实表达。
- 改一条模拟规则或模拟工具返回值，观察最终输出怎么变化。
- 故意制造一个缺字段、空结果或接口不可用的情况，说明兜底逻辑。
- 补充一条测试用例，检查输出是否仍然符合业务边界。

## 7. 运行方式

在脚本所在目录运行：

```bash
chmod +x start_macos_linux.sh
./start_macos_linux.sh status
```

如果是安装脚本，可以把 `status` 换成 `install`、`start`、`stop`、`logs` 等动作。

涉及 DeepSeek 时，只使用环境变量 `DEEPSEEK_API_KEY`，不要把 Key 写入文件。

## 8. 完整参考代码

```bash
#!/usr/bin/env bash
set -euo pipefail

# macOS / Linux 启动脚本。
# 第一次运行会先构建索引，再启动 Streamlit 页面。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export BGE_M3_MODEL_PATH="${BGE_M3_MODEL_PATH:-./models/bge-m3}"

if [[ ! -f "storage/chunks.json" || ! -f "storage/embeddings.npy" ]]; then
  echo "[RAG] 未检测到索引，开始构建。"
  python build_index.py
fi

echo "[RAG] 启动问答机器人：http://localhost:8501"
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```