# build index：代码说明

> 对应代码：`RAG_问答机器人/build_index.py`

## 1. 这段代码对应的行业需求

业务知识库问答：围绕订单服务规则、安全制度、应急预案、巡检记录和培训材料做可追溯检索。

## 2. 课堂目标

解决部分 macOS/conda 环境里 OpenMP 重复加载导致的崩溃

## 3. 讲解顺序

- 先用业务问题开场：业务现场为什么需要这段能力。
- 再讲输入是什么：用户问题、规则资料、图片/记录、工具参数或环境信息。
- 然后讲处理过程：每一步生成什么中间结果。
- 最后讲输出怎么看：哪些结果可信，哪些地方要人工确认或回到官方系统。

## 4. 代码结构

1. 先写脚本顶部说明：这段代码解决哪个业务系统问题，运行后应该看到什么。
2. 再写 import 和常量：优先使用标准库，路径使用 `Path(__file__).resolve().parent`。
3. 准备课堂模拟数据：例如订单服务规则、服务点、订单编号、巡检记录、工单状态。
4. 按函数逐个生成：`main`。
5. 最后写 `if __name__ == "__main__":` 入口，串起课堂演示流程。
6. 运行一次，确认输出里有中间结果、最终结果和可修改点。

## 5. 主要函数或代码块

- `main`

## 6. 可修改点

- 把示例问题换成在线客服、安全检查、设备运维或培训答疑中的真实表达。
- 改一条模拟规则或模拟工具返回值，观察最终输出怎么变化。
- 故意制造一个缺字段、空结果或接口不可用的情况，说明兜底逻辑。
- 补充一条测试用例，检查输出是否仍然符合业务边界。

## 7. 运行方式

在脚本所在目录运行：

```bash
python build_index.py
```

涉及 DeepSeek 时，只使用环境变量 `DEEPSEEK_API_KEY`，不要把 Key 写入文件。

## 8. 完整参考代码

```python
from __future__ import annotations

import argparse
import os
from pathlib import Path

from rag_core import DEFAULT_MODEL_PATH, build_and_save_index


def main() -> None:
    # 解决部分 macOS/conda 环境里 OpenMP 重复加载导致的崩溃。
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    project_dir = Path(__file__).resolve().parent
    default_source = project_dir.parent
    default_storage = project_dir / "storage"

    parser = argparse.ArgumentParser(description="为 RAG 问答机器人构建本地知识索引")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=default_source,
        help="知识库来源目录，默认读取整个 rag-tool-agent-course 课程目录。",
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=default_storage,
        help="索引保存目录。",
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv("BGE_M3_MODEL_PATH", DEFAULT_MODEL_PATH),
        help="BGE-m3 本地模型路径。",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=650,
        help="每个文本片段的大致字符数。",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=120,
        help="相邻片段重叠字符数。",
    )
    args = parser.parse_args()

    print("开始构建 RAG 索引")
    print(f"知识目录：{args.source_dir}")
    print(f"索引目录：{args.storage_dir}")
    print(f"BGE 模型：{args.model_path}")
    print(f"切片参数：chunk_size={args.chunk_size}, overlap={args.overlap}")
    print("-" * 70)

    build_and_save_index(
        source_dir=args.source_dir,
        storage_dir=args.storage_dir,
        model_path=args.model_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
```