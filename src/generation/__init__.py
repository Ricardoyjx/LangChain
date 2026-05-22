"""结果生成模块（阶段三）：提示词组装与 LLM 调用。

提供两类核心能力：
    - 提示词模板管理：``get_prompt`` / ``format_prompt`` / ``list_templates``
    - LLM 客户端：``create_ollama_client`` / ``verify_ollama_connection``

用法示例:
    .. code-block:: python

        from src.generation import get_prompt, create_ollama_client

        prompt = get_prompt("rag")
        llm = create_ollama_client("qwen3.5:9b", "http://localhost:11434", 0.3)
        chain = prompt | llm
"""

from src.generation.llm_client import create_ollama_client, verify_ollama_connection
from src.generation.prompt_template import (
    AVAILABLE_TEMPLATES,
    TemplateName,
    format_prompt,
    get_prompt,
    list_templates,
)

__all__ = [
    # 提示词模板
    "get_prompt",
    "format_prompt",
    "list_templates",
    "AVAILABLE_TEMPLATES",
    "TemplateName",
    # LLM 客户端
    "create_ollama_client",
    "verify_ollama_connection",
]
