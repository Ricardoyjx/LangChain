"""管线编排：串联数据摄取与查询的全流程。

用法示例:
    # 数据摄取
    pipeline = IngestPipeline()
    result = pipeline.run("data/raw/美的2025年报.pdf")
    print(f"已处理 {result['chunk_count']} 个文本块")

    # 查询
    qpipeline = QueryPipeline()
    answer = qpipeline.run("美的2024年营收是多少？")
    print(answer)
"""

from typing import Any, Dict, List, Optional

from src.data_processing.parsers.factory import process_heterogeneous_data
from src.data_processing.cleaner import clean_text
from src.data_processing.chunker import chunk_text


# ---------------------------------------------------------------------------
# 数据摄取管线
# ---------------------------------------------------------------------------

class IngestPipeline:
    """将原始文档经过 解析 -> 清洗 -> 切分 -> 入库 的全流程。"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def run(self, file_path: str) -> Dict[str, Any]:
        """执行完整的数据摄取流程。

        Args:
            file_path: 原始文档路径（支持 PDF/Word/Excel）。

        Returns:
            dict: {
                "chunks": List[Dict],      # 切分后的文本块，每块含 content + metadata
                "chunk_count": int,        # 文本块数量
                "tables": List[Any],       # 提取的表格数据
                "metadata": Dict,          # 文档元数据
            }
        """
        # 阶段一：解析
        parsed = process_heterogeneous_data(file_path)

        content = parsed.get("content", "")
        tables = parsed.get("tables", [])
        metadata = parsed.get("metadata", {})

        # 阶段二：清洗
        cleaned = clean_text(content)

        # 阶段三：切分（携带源文件元数据到每个分块）
        chunks = chunk_text(
            text=cleaned,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            metadata=metadata,
        )

        return {
            "chunks": chunks,
            "chunk_count": len(chunks),
            "tables": tables,
            "metadata": metadata,
        }


# ---------------------------------------------------------------------------
# 查询管线
# ---------------------------------------------------------------------------

class QueryPipeline:
    """将用户问题经过 检索 -> 后过滤 -> 重排序 -> 生成 的全流程。"""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def run(
        self,
        query: str,
        chunks: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """执行查询流程。

        Args:
            query: 用户问题。
            chunks: 带元数据的候选文本块列表。
            context: 外部传入的上下文（如已有检索结果时）。
            chat_history: 对话历史 [{"role": "user"/"assistant", "content": str}].

        Returns:
            生成的回答文本。
        """
        # TODO: 阶段一 — 向量检索 / 关键词检索 / 混合检索
        # TODO: 阶段二 — 后过滤（去重、时效性、权限）
        # TODO: 阶段三 — 重排序
        # TODO: 阶段四 — LLM 生成
        raise NotImplementedError("QueryPipeline 尚未实现，请先完成 retrieval 和 generation 模块。")
