"""几百个工具调用课程的公共数据与函数。

这个文件不是课堂主讲 demo，只是放公共工具注册表和通用筛选函数。
所有脚本都可以从这里导入，避免每个文件重复几百行数据。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter


def make_tool(name: str, group: str, description: str, required: list[str], properties: dict) -> dict:
    """创建一个同时包含轻量描述和完整 schema 的工具对象。"""
    short_schema = {key: value.get("type", "string") for key, value in properties.items()}
    return {
        "name": name,
        "group": group,
        "description": description,
        "short_schema": short_schema,
        "full_schema": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        },
    }


core_tools = [
    make_tool(
        name="gmail_search_email",
        group="gmail",
        description="搜索 Gmail 邮件，按发件人、主题、正文关键词、时间范围查询邮件和附件。",
        required=["query"],
        properties={
            "query": {"type": "string", "description": "邮件检索关键词"},
            "from_user": {"type": "string", "description": "发件人姓名或邮箱，可选"},
            "date_range": {"type": "string", "description": "时间范围，例如 last_7_days"},
            "has_attachment": {"type": "boolean", "description": "是否只查带附件邮件"},
        },
    ),
    make_tool(
        name="gmail_send_email",
        group="gmail",
        description="发送 Gmail 邮件，支持收件人、主题、正文和附件。",
        required=["to", "subject", "body"],
        properties={
            "to": {"type": "string", "description": "收件人邮箱"},
            "subject": {"type": "string", "description": "邮件主题"},
            "body": {"type": "string", "description": "邮件正文"},
            "attachments": {"type": "array", "items": {"type": "string"}, "description": "附件路径列表"},
        },
    ),
    make_tool(
        name="presentation_create_doc",
        group="presentation",
        description="创建 演示材料 或 Google Slides 演示材料，生成课程、汇报、培训课程材料。",
        required=["title", "outline"],
        properties={
            "title": {"type": "string", "description": "演示材料 标题"},
            "outline": {"type": "array", "items": {"type": "string"}, "description": "页面大纲"},
            "page_count": {"type": "integer", "description": "页数"},
            "style": {"type": "string", "description": "风格，例如 企业内训、技术汇报"},
        },
    ),
    make_tool(
        name="presentation_update_page",
        group="presentation",
        description="修改 演示材料 页面内容，替换标题、正文、图表和备注。",
        required=["deck_id", "page"],
        properties={
            "deck_id": {"type": "string", "description": "演示材料 文件 ID"},
            "page": {"type": "integer", "description": "页码"},
            "title": {"type": "string", "description": "新标题"},
            "body": {"type": "string", "description": "新正文"},
        },
    ),
    make_tool(
        name="repo_search_code",
        group="code",
        description="搜索代码仓库、文件系统或项目目录中的函数、类、变量和关键词。",
        required=["query"],
        properties={
            "query": {"type": "string", "description": "搜索关键词"},
            "path": {"type": "string", "description": "搜索路径，默认当前项目"},
            "file_glob": {"type": "string", "description": "文件类型过滤，例如 *.py"},
        },
    ),
    make_tool(
        name="repo_edit_file",
        group="code",
        description="修改代码仓库或文件系统中的文件内容，适合修 bug、改配置、补文档。",
        required=["path", "patch"],
        properties={
            "path": {"type": "string", "description": "文件路径"},
            "patch": {"type": "string", "description": "补丁内容"},
            "reason": {"type": "string", "description": "修改原因"},
        },
    ),
    make_tool(
        name="rag_search_knowledge",
        group="rag",
        description="搜索知识库、规章文档、课程资料和业务手册，返回相关片段。",
        required=["query"],
        properties={
            "query": {"type": "string", "description": "检索问题"},
            "top_k": {"type": "integer", "description": "返回片段数量"},
            "source_filter": {"type": "string", "description": "资料来源过滤，可选"},
        },
    ),
    make_tool(
        name="calendar_create_event",
        group="calendar",
        description="创建日程、会议、提醒，设置时间、地点、参与人。",
        required=["title", "time"],
        properties={
            "title": {"type": "string", "description": "日程标题"},
            "time": {"type": "string", "description": "时间"},
            "location": {"type": "string", "description": "地点"},
            "attendees": {"type": "array", "items": {"type": "string"}, "description": "参与人"},
        },
    ),
]


def make_dummy_tools() -> list[dict]:
    """补充干扰工具，模拟几百个工具的真实环境。"""
    groups = {
        "spreadsheet": ["表格", "数据清洗", "公式", "透视表", "CSV", "统计"],
        "browser": ["网页", "搜索", "打开链接", "下载", "浏览器", "页面"],
        "database": ["数据库", "SQL", "表结构", "查询", "更新", "索引"],
        "image": ["图片", "识别", "生成", "OCR", "图像", "尺寸"],
        "workflow": ["工作流", "审批", "节点", "自动化", "状态", "触发器"],
        "monitor": ["日志", "监控", "报警", "指标", "异常", "追踪"],
        "ticket": ["工单", "客服", "投诉", "处理", "流转", "备注"],
        "filesystem": ["文件", "目录", "移动", "复制", "删除", "读取"],
    }
    tools = []
    for group, words in groups.items():
        for i in range(28):
            keyword = words[i % len(words)]
            name = f"{group}_tool_{i + 1:02d}"
            tools.append(make_tool(
                name=name,
                group=group,
                description=f"{group} 工具，用于处理 {keyword} 相关任务，支持查询、创建、更新和摘要。",
                required=["input"],
                properties={
                    "input": {"type": "string", "description": "用户输入"},
                    "mode": {"type": "string", "description": "执行模式"},
                    "limit": {"type": "integer", "description": "返回数量"},
                },
            ))
    return tools


tool_registry = core_tools + make_dummy_tools()


rule_groups = [
    {
        "group": "gmail",
        "keywords": ["邮件", "email", "gmail", "收件箱", "发件人", "附件邮件"],
        "reason": "用户提到邮件，进入 Gmail tool group。",
    },
    {
        "group": "presentation",
        "keywords": ["md", "演示材料", "演示材料", "课程材料", "演示材料", "presentation"],
        "reason": "用户提到 演示材料 或演示材料，进入 presentation tool group。",
    },
    {
        "group": "code",
        "keywords": ["代码仓库", "仓库", "repo", "filesystem", "文件系统", "代码", "bug", "函数"],
        "reason": "用户提到代码仓库或文件系统，进入 code tools。",
    },
    {
        "group": "rag",
        "keywords": ["知识库", "RAG", "检索", "向量", "文档", "召回", "规章"],
        "reason": "用户提到知识库或检索，进入 RAG tools。",
    },
]


def rule_filter(user_query: str) -> tuple[list[dict], list[str], list[str]]:
    """根据关键词过滤到工具组；没命中时回退到全量工具。"""
    matched_groups = []
    reasons = []
    lower_query = user_query.lower()

    for rule in rule_groups:
        if any(keyword.lower() in lower_query for keyword in rule["keywords"]):
            matched_groups.append(rule["group"])
            reasons.append(rule["reason"])

    if matched_groups:
        candidates = [tool for tool in tool_registry if tool["group"] in matched_groups]
    else:
        candidates = tool_registry
        reasons.append("没有命中强规则，退回全量工具池。")

    return candidates, matched_groups, reasons


def tokenize(text: str) -> list[str]:
    """中文字符 + 中文二元组 + 英文词的轻量 tokenizer。"""
    text = text.lower()
    english = re.findall(r"[a-z0-9_]+", text)
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    bigrams = [chinese[i] + chinese[i + 1] for i in range(len(chinese) - 1)]
    return english + chinese + bigrams


def vectorize(text: str) -> Counter:
    return Counter(tokenize(text))


def cosine(a: Counter, b: Counter) -> float:
    common = set(a) & set(b)
    dot = sum(a[token] * b[token] for token in common)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def vector_recall(user_query: str, candidates: list[dict], top_k: int = 12) -> list[dict]:
    """从工具轻量描述中召回 top-k。"""
    query_vec = vectorize(user_query)
    scored = []
    for tool in candidates:
        text = f"{tool['name']} {tool['group']} {tool['description']} {json.dumps(tool['short_schema'], ensure_ascii=False)}"
        scored.append({
            "tool": tool,
            "vector_score": cosine(query_vec, vectorize(text)),
        })
    scored.sort(key=lambda item: item["vector_score"], reverse=True)
    return scored[:top_k]


def simulated_llm_rerank(user_query: str, recalled: list[dict], top_n: int = 5) -> list[dict]:
    """用透明规则模拟 LLM rerank，真实系统可替换为小模型或主模型。"""
    query = user_query.lower()
    reranked = []
    for item in recalled:
        tool = item["tool"]
        text = f"{tool['name']} {tool['group']} {tool['description']}".lower()
        score = item["vector_score"] * 60
        reasons = [f"向量分 {item['vector_score']:.3f}"]

        if "邮件" in query and tool["group"] == "gmail":
            score += 35
            reasons.append("邮件任务强相关")
        if "md" in query and tool["group"] == "presentation":
            score += 35
            reasons.append("演示材料 任务强相关")
        if "代码仓库" in query and tool["group"] == "code":
            score += 35
            reasons.append("代码仓库任务强相关")
        if "知识库" in query and tool["group"] == "rag":
            score += 35
            reasons.append("知识库任务强相关")
        if any(word in query for word in ["找", "搜索", "查", "检索"]):
            if any(word in text for word in ["search", "搜索", "查询", "检索"]):
                score += 15
                reasons.append("动作是查询/搜索")
        if any(word in query for word in ["创建", "生成", "做一个", "写一个"]):
            if any(word in text for word in ["create", "创建", "生成"]):
                score += 15
                reasons.append("动作是创建/生成")
        if any(word in query for word in ["修改", "改", "更新"]):
            if any(word in text for word in ["edit", "update", "修改", "更新"]):
                score += 15
                reasons.append("动作是修改/更新")

        reranked.append({
            "tool": tool,
            "rerank_score": round(score, 2),
            "reason": "；".join(reasons),
        })

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return reranked[:top_n]


def inject_full_schema(final_tools: list[dict]) -> list[dict]:
    """最终只给少量工具的完整 schema。"""
    return [item["tool"]["full_schema"] for item in final_tools]


def estimate_schema_chars(tools: list[dict], full: bool) -> int:
    """粗略估算注入文本长度，用字符数代替 token 数，课堂里更直观。"""
    if full:
        payload = [tool["full_schema"] for tool in tools]
    else:
        payload = [
            {
                "name": tool["name"],
                "group": tool["group"],
                "description": tool["description"],
                "short_schema": tool["short_schema"],
            }
            for tool in tools
        ]
    return len(json.dumps(payload, ensure_ascii=False))
