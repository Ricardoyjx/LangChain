"""提示词模版管理：存放各类业务的 Prompt 模板。

提供两种使用方式:
    1. ``get_prompt(name)`` — 获取 ``ChatPromptTemplate`` 对象，直接与 LCEL 链式调用集成
    2. ``format_prompt(name, **kwargs)`` — 动态拼接模板，返回渲染后的字符串

模板分类:
    - ``extract``     : 金融信息提取（按 Schema 输出 JSON）
    - ``rag``         : RAG 问答（带知识库上下文）
    - ``analysis``    : 金融分析（思维链引导：观点 → 数据 → 风险）
    - ``verify``      : 数值一致性校验
    - ``cite``        : 带引用溯源的 RAG 回答
    - ``classify``    : 用户查询意图分类

用法示例:
    .. code-block:: python

        from generation.prompt_template import get_prompt, format_prompt

        # LCEL 链式调用
        prompt = get_prompt("rag")
        chain = prompt | llm | parser

        # 直接渲染
        text = format_prompt("extract", text=doc_str, schema=schema)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, TypeAlias

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------

TemplateName: TypeAlias = Literal[
    "extract",
    "rag",
    "analysis",
    "verify",
    "cite",
    "classify",
]


# ===================================================================
# 模板定义
# ===================================================================
# 每个模板统一以 _build_<name>(...) -> ChatPromptTemplate 函数定义，
# 并通过 _REGISTRY 字典注册，方便统一管理与测试。

# -------------------------------------------------------------------
# 1. extract — 金融信息提取
# -------------------------------------------------------------------

_EXTRACT_SYSTEM = """\
你是一个专业的金融信息提取助手。你的任务是从给定的文本中提取指定的金融信息。

请严格按照以下 Schema 输出 JSON：
{schema}

约束条件：
1. 如果某个字段的信息在原文中未提及，请填入"原文未提及"。
2. 不要编造任何信息。
3. 只输出 JSON，不要添加任何解释或额外文字。
4. 确保输出的 JSON 是合法有效的。"""

_EXTRACT_HUMAN = "文本内容：\n{text}"


def _build_extract() -> ChatPromptTemplate:
    """构建金融信息提取模板。

    期望参数：
        schema: JSON Schema 字典或格式化后的 schema 字符串
        text:   待提取的原始文本
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", _EXTRACT_SYSTEM),
            ("human", _EXTRACT_HUMAN),
        ]
    )


def _format_extract(
    text: str,
    schema: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> str:
    """渲染金融信息提取提示词（字符串格式）。

    Args:
        text: 待提取的原始文本。
        schema: JSON Schema 字典。为 ``None`` 时使用默认 schema。
    """
    if schema is None:
        schema = {"type": "object", "properties": {}}
    schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
    # 手动插值，避免 ChatPromptTemplate 重复构造
    system = _EXTRACT_SYSTEM.format(schema=schema_str)
    human = _EXTRACT_HUMAN.format(text=text)
    return f"{system}\n\n{human}"


# -------------------------------------------------------------------
# 2. rag — RAG 问答
# -------------------------------------------------------------------

_RAG_SYSTEM = """\
你是一个专业的金融分析助手。请根据以下知识库信息回答用户问题。

知识库信息：
{context}

回答要求：
1. 严格基于知识库内容回答，不要使用你训练数据中的过时信息。
2. 如果知识库中没有相关信息，请明确说明无法回答。
3. 涉及数字时，请引用原文中的数值。
4. 回答应简洁、准确、专业。"""

_RAG_HUMAN = "用户问题：{question}"


def _build_rag() -> ChatPromptTemplate:
    """构建 RAG 问答模板（无对话历史）。

    期望参数：
        context:  知识库上下文文本
        question: 用户问题
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", _RAG_SYSTEM),
            ("human", _RAG_HUMAN),
        ]
    )


def _build_rag_with_history() -> ChatPromptTemplate:
    """构建带对话历史的 RAG 问答模板。

    期望参数：
        context:      知识库上下文文本
        chat_history: 对话历史（List of messages）
        question:     用户最新问题
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", _RAG_SYSTEM),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", _RAG_HUMAN),
        ]
    )


def _format_rag(context: str, question: str, **kwargs: Any) -> str:
    """渲染 RAG 问答提示词（字符串格式）。"""
    system = _RAG_SYSTEM.format(context=context)
    human = _RAG_HUMAN.format(question=question)
    return f"{system}\n\n{human}"


# -------------------------------------------------------------------
# 3. analysis — 金融分析（思维链引导）
# -------------------------------------------------------------------

_ANALYSIS_SYSTEM = """\
你是一位资深金融分析师。请基于以下信息，按照"核心观点 → 数据支撑 → 风险提示"的逻辑结构进行分析。

分析材料：
{context}

输出结构要求：
1. **核心观点**：用 1-2 句话概括最关键的结论。
2. **数据支撑**：列出支撑核心观点的关键数据，必须注明数据来源。
3. **风险提示**：指出当前分析中的潜在风险或不确定性因素。

风格要求：
- 客观中立，不夸大不缩小。
- 数据必须来源于分析材料，不得编造。
- 如有对比分析，请明确说明对比基准。"""

_ANALYSIS_HUMAN = "分析请求：{question}"


def _build_analysis() -> ChatPromptTemplate:
    """构建金融分析模板（思维链引导）。

    期望参数：
        context:  分析所用的材料
        question: 分析请求 / 问题
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", _ANALYSIS_SYSTEM),
            ("human", _ANALYSIS_HUMAN),
        ]
    )


def _format_analysis(context: str, question: str, **kwargs: Any) -> str:
    """渲染金融分析提示词（字符串格式）。"""
    system = _ANALYSIS_SYSTEM.format(context=context)
    human = _ANALYSIS_HUMAN.format(question=question)
    return f"{system}\n\n{human}"


# -------------------------------------------------------------------
# 4. verify — 数值一致性校验
# -------------------------------------------------------------------

_VERIFY_SYSTEM = """\
你是一个金融数据校验员。请检查以下回答中的数值是否与原文一致。

【原文】
{source_text}

【回答】
{answer}

校验要求：
1. 逐项检查回答中的每个数值是否能在原文中找到对应依据。
2. 如果数值与原文一致，标记 ✅。
3. 如果数值与原文不一致（数值偏差、单位错误等），标记 ❌ 并给出原文的正确数值。
4. 如果回答中的数值在原文中找不到对应信息，标记 ⚠️ 原文未提及。

请以 JSON 格式输出校验结果：
{{
    "fields": [
        {{
            "value_in_answer": "回答中的数值/描述",
            "value_in_source": "原文中的数值/描述或'原文未提及'",
            "status": "✅ / ❌ / ⚠️",
            "note": "备注说明"
        }}
    ],
    "overall_verdict": "一致 / 部分不一致 / 无法校验"
}}

只输出 JSON，不要添加任何解释。"""


def _build_verify() -> ChatPromptTemplate:
    """构建数值一致性校验模板。

    期望参数：
        source_text: 原始材料（知识库上下文）
        answer:      模型生成的回答
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", _VERIFY_SYSTEM),
        ]
    )


def _format_verify(source_text: str, answer: str, **kwargs: Any) -> str:
    """渲染数值校验提示词（字符串格式）。"""
    return _VERIFY_SYSTEM.format(source_text=source_text, answer=answer)


# -------------------------------------------------------------------
# 5. cite — 带引用溯源的 RAG 回答
# -------------------------------------------------------------------

_CITE_SYSTEM = """\
你是一个专业的金融分析助手。请基于以下知识库信息回答用户问题，并在每个关键结论后标注引用来源。

知识库信息（每段均带来源标记 [来源 X]）：
{context}

回答要求：
1. 严格基于知识库内容回答。
2. 在每个关键结论或数据后标注引用来源，格式为 [来源 X]。
3. 如果知识库中没有相关信息，请说明无法回答。
4. 回答末尾提供一个"参考资料"列表。"""

_CITE_HUMAN = "用户问题：{question}"


def _build_cite() -> ChatPromptTemplate:
    """构建带引用溯源的 RAG 回答模板。

    期望参数：
        context:  知识库上下文（每段带 [来源 X] 标记）
        question: 用户问题
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", _CITE_SYSTEM),
            ("human", _CITE_HUMAN),
        ]
    )


def _format_cite(context: str, question: str, **kwargs: Any) -> str:
    """渲染带引用溯源的提示词（字符串格式）。"""
    system = _CITE_SYSTEM.format(context=context)
    human = _CITE_HUMAN.format(question=question)
    return f"{system}\n\n{human}"


# -------------------------------------------------------------------
# 6. classify — 用户查询意图分类
# -------------------------------------------------------------------

_CLASSIFY_SYSTEM = """\
你是一个金融查询分类器。请判断用户问题的类别。

类别定义：
- "real_time": 询问实时行情、股价、最新价格等需要实时数据的问题。
- "report":    询问研报分析、财务数据、公司基本面等基于已有文档的问题。
- "comparison": 询问对比分析（如对比两家公司、不同时间段）。
- "summary":   询问总结、概括类问题。
- "definition": 询问金融术语定义或概念解释。
- "other":     其他类别。

输出格式：只输出类别名称（如 "report"），不要输出任何其他内容。

用户问题：{question}"""


def _build_classify() -> ChatPromptTemplate:
    """构建用户查询意图分类模板。

    期望参数：
        question: 用户问题
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", _CLASSIFY_SYSTEM),
        ]
    )


def _format_classify(question: str, **kwargs: Any) -> str:
    """渲染意图分类提示词（字符串格式）。"""
    return _CLASSIFY_SYSTEM.format(question=question)


# ===================================================================
# 模板注册表
# ===================================================================

_REGISTRY: Dict[str, ChatPromptTemplate] = {
    "extract": _build_extract(),
    "rag": _build_rag(),
    "rag_with_history": _build_rag_with_history(),
    "analysis": _build_analysis(),
    "verify": _build_verify(),
    "cite": _build_cite(),
    "classify": _build_classify(),
}

_FORMATTERS: Dict[str, Any] = {
    "extract": _format_extract,
    "rag": _format_rag,
    "analysis": _format_analysis,
    "verify": _format_verify,
    "cite": _format_cite,
    "classify": _format_classify,
}

# 公开的模板名称列表
AVAILABLE_TEMPLATES: List[str] = list(_REGISTRY.keys())


# ===================================================================
# 统一访问接口
# ===================================================================


def get_prompt(name: TemplateName) -> ChatPromptTemplate:
    """获取指定名称的 ``ChatPromptTemplate`` 对象。

    Args:
        name: 模板名称（``AVAILABLE_TEMPLATES`` 中的值）。

    Returns:
        对应的 ``ChatPromptTemplate`` 对象。

    Raises:
        KeyError: 当模板名称不存在时。
    """
    if name not in _REGISTRY:
        raise KeyError(f"未知的模板名称: {name!r}。可用模板: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def format_prompt(name: TemplateName, **kwargs: Any) -> str:
    """直接渲染指定模板为字符串。

    适用于不想依赖 LangChain 模版引擎的场景。

    Args:
        name: 模板名称。
        **kwargs: 模板参数（如 ``text``、``schema``、``context``、``question`` 等）。

    Returns:
        渲染后的提示词字符串。
    """
    if name not in _FORMATTERS:
        raise KeyError(
            f"模板 {name!r} 不支持字符串渲染，请使用 get_prompt() 获取 ChatPromptTemplate 对象。"
            f"支持字符串渲染的模板: {list(_FORMATTERS.keys())}"
        )
    return _FORMATTERS[name](**kwargs)


def list_templates() -> List[Dict[str, str]]:
    """列出所有已注册的模板信息。

    Returns:
        每个模板的 name 和 description。
    """
    descriptions = {
        "extract": "金融信息提取（按 Schema 输出 JSON）",
        "rag": "RAG 问答（带知识库上下文）",
        "rag_with_history": "RAG 问答（带对话历史）",
        "analysis": "金融分析（思维链引导：观点 → 数据 → 风险）",
        "verify": "数值一致性校验",
        "cite": "带引用溯源的 RAG 回答",
        "classify": "用户查询意图分类",
    }
    return [
        {"name": name, "description": descriptions.get(name, "")} for name in _REGISTRY
    ]
