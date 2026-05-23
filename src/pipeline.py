"""管线编排：串联数据摄取与查询的全流程。

用法示例:
    # 数据摄取
    pipeline = IngestPipeline()
    result = pipeline.run("data/raw/美的2025年报.pdf")

    # 构建索引
    pipeline.build_index()

    # 查询
    qpipeline = QueryPipeline(vector_store=pipeline.vector_store,
                               bm25_retriever=pipeline.bm25_retriever,
                               llm=pipeline.llm)
    answer = qpipeline.run("美的2024年营收是多少？")
    print(answer)
"""

from __future__ import annotations

import copy
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_ollama import OllamaEmbeddings

from src.data_processing.cleaner import clean_text
from src.data_processing.chunker import chunk_text
from src.data_processing.parsers.factory import process_heterogeneous_data
from src.generation import create_ollama_client, get_prompt

# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------

_DEFAULT_EMBEDDING_MODEL = "qwen3.5:9b"
_DEFAULT_LLM_MODEL = "qwen3.5:9b"
_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_TEMPERATURE = 0.3

_RRF_CONSTANT = 60  # RRF 融合常数


# ===================================================================
# 数据摄取管线
# ===================================================================

class IngestPipeline:
    """将原始文档经过 解析 -> 清洗 -> 切分 -> 向量化索引 的全流程。

    支持多文档增量摄入（多次调用 ``run`` 后统一 ``build_index``）。
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embedding_model_name: str = _DEFAULT_EMBEDDING_MODEL,
        llm_model: str = _DEFAULT_LLM_MODEL,
        ollama_base_url: str = _DEFAULT_OLLAMA_URL,
        temperature: float = _DEFAULT_TEMPERATURE,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 累加器：支持增量摄入
        self._all_chunks: List[Dict[str, Any]] = []
        self._all_tables: List[Any] = []
        self._metadata: Dict[str, Any] = {}

        # 嵌入模型（延迟初始化）
        self._embedding_model_name = embedding_model_name

        # 检索器（build_index 后可用）
        self.vector_store: Optional[FAISS] = None
        self.bm25_retriever: Optional[BM25Retriever] = None

        # ---------- 初始化 Ollama 客户端 ----------
        try:
            self._embedding_model = OllamaEmbeddings(model=embedding_model_name)
        except Exception as e:
            raise ConnectionError(
                f"Ollama 嵌入模型初始化失败 (model={embedding_model_name!r}, "
                f"url={ollama_base_url!r}): {e}"
            ) from e

        try:
            self.llm = create_ollama_client(llm_model, ollama_base_url, temperature)
        except Exception as e:
            raise ConnectionError(
                f"Ollama LLM 初始化失败 (model={llm_model!r}, "
                f"url={ollama_base_url!r}): {e}"
            ) from e
        # -----------------------------------------

    # ------------------------------------------------------------------
    def run(self, file_path: str) -> Dict[str, Any]:
        """执行单文档的摄取流程，结果暂存于内部缓冲区。

        Args:
            file_path: 原始文档路径（支持 PDF/Word/Excel）。

        Returns:
            dict: {
                "chunks": List[Dict],
                "chunk_count": int,
                "tables": List[Any],
                "metadata": Dict,
            }
        """
        # 阶段一：解析
        parsed = process_heterogeneous_data(file_path)

        content = parsed.get("content", "")
        tables = parsed.get("tables", [])
        metadata = parsed.get("metadata", {})

        # 合并元数据（后调用的文件覆盖同名键）
        self._metadata.update(metadata)
        self._metadata.setdefault("source_files", []).append(file_path)

        # 阶段二：清洗
        cleaned = clean_text(content)

        # 阶段三：切分
        chunks = chunk_text(
            text=cleaned,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            metadata={"source": file_path, **metadata},
        )

        self._all_chunks.extend(chunks)
        self._all_tables.extend(tables)

        return {
            "chunks": chunks,
            "chunk_count": len(chunks),
            "tables": tables,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    def build_index(self) -> None:
        """将已摄入的所有文本块构建为 FAISS 向量库 + BM25 检索器。

        必须在至少一次 ``run()`` 调用后执行。
        """
        if not self._all_chunks:
            raise ValueError("没有文本块可索引，请先调用 run() 摄入文档。")

        documents = [
            Document(
                page_content=chunk["content"],
                metadata=chunk.get("metadata", {}),
            )
            for chunk in self._all_chunks
        ]

        self.vector_store = FAISS.from_documents(documents, self._embedding_model)
        self.bm25_retriever = BM25Retriever.from_documents(documents)

    # ------------------------------------------------------------------
    def save_index(self, directory: str) -> None:
        """将索引持久化到磁盘。

        Args:
            directory: 输出目录。会在其中创建 vector_db/ 和 bm25/ 子目录。
        """
        if self.vector_store is None:
            raise ValueError("索引未构建，请先调用 build_index()。")

        base = Path(directory)
        base.mkdir(parents=True, exist_ok=True)

        # FAISS 索引
        vector_dir = base / "vector_db"
        self.vector_store.save_local(str(vector_dir))

        # BM25 索引（pickle 序列化）
        bm25_dir = base / "bm25"
        bm25_dir.mkdir(parents=True, exist_ok=True)
        with open(bm25_dir / "retriever.pkl", "wb") as f:
            pickle.dump(self.bm25_retriever, f)

        # chunks 元数据
        meta_path = base / "chunks_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "chunk_count": len(self._all_chunks),
                    "source_files": self._metadata.get("source_files", []),
                    "embedding_model": self._embedding_model_name,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    # ------------------------------------------------------------------
    def load_index(self, directory: str) -> None:
        """从磁盘载入之前持久化的索引。

        Args:
            directory: 之前 ``save_index`` 使用的目录。
        """
        base = Path(directory)

        vector_dir = base / "vector_db"
        if not vector_dir.exists():
            raise FileNotFoundError(f"FAISS 索引目录不存在: {vector_dir}")

        self.vector_store = FAISS.load_local(
            str(vector_dir),
            self._embedding_model,
            allow_dangerous_deserialization=True,
        )

        bm25_dir = base / "bm25"
        bm25_pkl = bm25_dir / "retriever.pkl"
        if bm25_pkl.exists():
            with open(bm25_pkl, "rb") as f:
                self.bm25_retriever = pickle.load(f)

    # ------------------------------------------------------------------
    @property
    def chunk_count(self) -> int:
        """当前已摄入的文本块总数。"""
        return len(self._all_chunks)


# ===================================================================
# 检索工具（RRF 融合）
# ===================================================================

def _rrf_merge(
    vector_results: List[Document],
    bm25_results: List[Document],
    k: int = 5,
    constant: int = _RRF_CONSTANT,
) -> List[Document]:
    """使用倒数排名融合（Reciprocal Rank Fusion）合并两路检索结果。

    Args:
        vector_results: 向量检索结果。
        bm25_results: BM25 关键词检索结果。
        k: 最终返回的文档数。
        constant: RRF 常数（默认 60）。

    Returns:
        融合排序后的文档列表（已去重）。
    """
    scores: Dict[str, float] = {}

    for i, doc in enumerate(vector_results):
        doc_id = doc.metadata.get("chunk_index", id(doc))
        scores[str(doc_id)] = scores.get(str(doc_id), 0.0) + 1.0 / (i + constant)

    for i, doc in enumerate(bm25_results):
        doc_id = doc.metadata.get("chunk_index", id(doc))
        scores[str(doc_id)] = scores.get(str(doc_id), 0.0) + 1.0 / (i + constant)

    # 合并所有文档（按 content 去重）
    seen: set[str] = set()
    merged: List[Document] = []
    for doc in vector_results + bm25_results:
        content = doc.page_content
        if content not in seen:
            seen.add(content)
            merged.append(doc)

    merged.sort(key=lambda d: scores.get(str(d.metadata.get("chunk_index", id(d))), 0.0), reverse=True)
    return merged[:k]


# ===================================================================
# 查询管线
# ===================================================================

class QueryPipeline:
    """将用户问题经过 检索 -> 后过滤 -> 重排序 -> LLM生成 的全流程。

    需传入已构建好索引的 ``vector_store`` 和 ``bm25_retriever``。

    用法示例:
        qpipeline = QueryPipeline(
            vector_store=ingest.vector_store,
            bm25_retriever=ingest.bm25_retriever,
            llm=ingest.llm,
        )
        answer = qpipeline.run("美的2024年营收是多少？")
    """

    def __init__(
        self,
        vector_store: FAISS,
        bm25_retriever: BM25Retriever,
        llm: BaseLanguageModel,
        top_k: int = 5,
        rerank: bool = True,
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.llm = llm
        self.top_k = top_k
        self.rerank = rerank

    # ------------------------------------------------------------------
    def run(
        self,
        query: str,
        prompt_name: str = "rag",
        chat_history: Optional[List[Dict[str, str]]] = None,
        time_filter: Optional[Tuple[str, str]] = None,
    ) -> str:
        """执行完整查询流程。

        Args:
            query: 用户问题。
            prompt_name: 使用的提示词模板名称（默认 ``rag``，
                         可选 ``cite`` / ``analysis`` 等）。
            chat_history: 对话历史。
            time_filter: 可选的时间范围过滤 ``(start_date, end_date)``，
                         格式 ``"YYYY-MM-DD"``。

        Returns:
            LLM 生成的回答文本。
        """
        # 阶段一：混合检索（向量 + BM25）
        retrieved = self._hybrid_retrieve(query)

        # 阶段二：后过滤
        filtered = self._post_filter(retrieved, time_filter=time_filter)

        # 阶段三：重排序（可选）
        if self.rerank and len(filtered) > 1:
            filtered = self._rerank(query, filtered)

        # 阶段四：拼接上下文
        context = self._format_context(filtered)

        # 阶段五：LLM 生成
        answer = self._generate(query, context, prompt_name, chat_history)

        return answer

    # ------------------------------------------------------------------
    def _hybrid_retrieve(self, query: str) -> List[Document]:
        """双路并行召回 + RRF 融合。"""
        vector_results = self.vector_store.similarity_search(query, k=self.top_k)
        bm25_results = self.bm25_retriever.invoke(query)

        return _rrf_merge(vector_results, bm25_results, k=self.top_k)

    # ------------------------------------------------------------------
    def _post_filter(
        self,
        docs: List[Document],
        time_filter: Optional[Tuple[str, str]] = None,
    ) -> List[Document]:
        """后过滤链：去重 + 时效性过滤。"""
        # 1. 基础去重（按 content 去重，保留第一个）
        seen: set[str] = set()
        deduped: List[Document] = []
        for doc in docs:
            key = doc.page_content[:50]  # 用前 50 个字符作为指纹
            if key not in seen:
                seen.add(key)
                deduped.append(doc)

        # 2. 时效性过滤（基于 metadata 中的 date 字段）
        if time_filter:
            start_str, end_str = time_filter
            filtered: List[Document] = []
            for doc in deduped:
                doc_date = doc.metadata.get("date", "")
                if doc_date and start_str <= doc_date <= end_str:
                    filtered.append(doc)
            # 如果过滤后为空则回退到原始结果
            if filtered:
                deduped = filtered

        return deduped

    # ------------------------------------------------------------------
    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """简单重排序：基于 BM25 打分再排（后续可替换为交叉编码器模型）。"""
        # 若文档数少于 3，不做重排
        if len(docs) <= 2:
            return docs

        # 简单策略：按 BM25 相关性降序（利用 BM25Retriever 的能力）
        bm25_scores: Dict[str, float] = {}
        for doc in self.bm25_retriever.invoke(query):
            key = doc.page_content[:50]
            bm25_scores[key] = bm25_scores.get(key, 0) + 1

        def _score(doc: Document) -> float:
            return bm25_scores.get(doc.page_content[:50], 0.0)

        docs_sorted = sorted(docs, key=_score, reverse=True)
        return docs_sorted

    # ------------------------------------------------------------------
    def _format_context(self, docs: List[Document]) -> str:
        """将检索到的文档拼接为 LLM 的上下文文本。

        每个片段前缀标注 [来源 i]，方便模板中的 cite 模式使用。
        """
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "")
            source_tag = f"[来源 {i}] " if source else ""
            parts.append(f"{source_tag}{doc.page_content}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    def _generate(
        self,
        query: str,
        context: str,
        prompt_name: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """调用 LLM 生成回答。"""
        # 带对话历史时使用特定模板
        if chat_history:
            prompt = get_prompt("rag_with_history")
            chain = (
                RunnableParallel(
                    context=lambda _: context,
                    question=lambda _: query,
                    chat_history=lambda _: chat_history,
                )
                | prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            prompt = get_prompt(prompt_name)
            chain = (
                RunnableParallel(
                    context=lambda _: context,
                    question=lambda _: query,
                )
                | prompt
                | self.llm
                | StrOutputParser()
            )

        return chain.invoke({})

    # ------------------------------------------------------------------
    def stream(
        self,
        query: str,
        prompt_name: str = "rag",
        chat_history: Optional[List[Dict[str, str]]] = None,
    ):
        """流式查询，逐 token 产出回答（适用于实时展示）。"""
        retrieved = self._hybrid_retrieve(query)
        filtered = self._post_filter(retrieved)
        if self.rerank and len(filtered) > 1:
            filtered = self._rerank(query, filtered)
        context = self._format_context(filtered)

        if chat_history:
            prompt = get_prompt("rag_with_history")
            chain = (
                RunnableParallel(
                    context=lambda _: context,
                    question=lambda _: query,
                    chat_history=lambda _: chat_history,
                )
                | prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            prompt = get_prompt(prompt_name)
            chain = (
                RunnableParallel(
                    context=lambda _: context,
                    question=lambda _: query,
                )
                | prompt
                | self.llm
                | StrOutputParser()
            )

        yield from chain.stream({})


# ===================================================================
# 便捷入口：一站式 RAG
# ===================================================================

class RAGPipeline:
    """简化版一站式 RAG 管线，封装 IngestPipeline + QueryPipeline。

    适用场景：快速上手、单次实验。

    用法示例:
        rag = RAGPipeline()
        rag.ingest("data/raw/美的2025年报.pdf")
        answer = rag.query("美的2024年营收是多少？")
        print(answer)
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
        rerank: bool = True,
    ):
        self._ingest = IngestPipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._query: Optional[QueryPipeline] = None
        self.top_k = top_k
        self.rerank = rerank

    # ------------------------------------------------------------------
    def ingest(self, file_path: str) -> Dict[str, Any]:
        """摄入并索引单文档。"""
        result = self._ingest.run(file_path)
        self._ingest.build_index()
        self._query = QueryPipeline(
            vector_store=self._ingest.vector_store,
            bm25_retriever=self._ingest.bm25_retriever,
            llm=self._ingest.llm,
            top_k=self.top_k,
            rerank=self.rerank,
        )
        return result

    # ------------------------------------------------------------------
    def ingest_multiple(self, file_paths: List[str]) -> Dict[str, Any]:
        """批量摄入多个文档后统一建索引。"""
        combined = {"chunks": [], "chunk_count": 0, "tables": [], "metadata": {}}
        for fp in file_paths:
            result = self._ingest.run(fp)
            combined["chunks"].extend(result["chunks"])
            combined["chunk_count"] += result["chunk_count"]
            combined["tables"].extend(result["tables"])
            combined["metadata"].update(result["metadata"])

        self._ingest.build_index()
        self._query = QueryPipeline(
            vector_store=self._ingest.vector_store,
            bm25_retriever=self._ingest.bm25_retriever,
            llm=self._ingest.llm,
            top_k=self.top_k,
            rerank=self.rerank,
        )
        return combined

    # ------------------------------------------------------------------
    def query(
        self,
        question: str,
        prompt_name: str = "rag",
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """执行查询。"""
        if self._query is None:
            raise RuntimeError("请先调用 ingest() 摄入文档。")
        return self._query.run(question, prompt_name=prompt_name, chat_history=chat_history)

    # ------------------------------------------------------------------
    def stream(
        self,
        question: str,
        prompt_name: str = "rag",
        chat_history: Optional[List[Dict[str, str]]] = None,
    ):
        """流式查询。"""
        if self._query is None:
            raise RuntimeError("请先调用 ingest() 摄入文档。")
        yield from self._query.stream(
            question, prompt_name=prompt_name, chat_history=chat_history
        )

    # ------------------------------------------------------------------
    @property
    def ingest_pipeline(self) -> IngestPipeline:
        """获取底层 IngestPipeline 实例。"""
        return self._ingest

    @property
    def query_pipeline(self) -> Optional[QueryPipeline]:
        """获取底层 QueryPipeline 实例。"""
        return self._query
