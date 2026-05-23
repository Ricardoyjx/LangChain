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
import logging
import os
import pickle
import tempfile
import time
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

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

logger = logging.getLogger("pipeline")

DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-5s | %(name)s.%(funcName)s | %(message)s"
)
DEFAULT_LOG_DATE_FORMAT = "%H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    fmt: str = DEFAULT_LOG_FORMAT,
    datefmt: str = DEFAULT_LOG_DATE_FORMAT,
) -> None:
    """一键配置 pipeline 日志格式。

    Args:
        level:    日志级别（默认 INFO）。
        fmt:      日志格式字符串。
        datefmt:  时间格式。
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False  # 避免根日志重复输出


# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------

_DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
_DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:9b")
_DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

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

        logger.info(
            "初始化 IngestPipeline | "
            "chunk_size=%d chunk_overlap=%d embedding=%s llm=%s",
            chunk_size, chunk_overlap, embedding_model_name, llm_model,
        )

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
        _t0 = time.time()
        try:
            self._embedding_model = OllamaEmbeddings(model=embedding_model_name)
            logger.info("OllamaEmbeddings 就绪 (%.2fs)", time.time() - _t0)
        except Exception as e:
            raise ConnectionError(
                f"Ollama 嵌入模型初始化失败 (model={embedding_model_name!r}, "
                f"url={ollama_base_url!r}): {e}"
            ) from e

        _t0 = time.time()
        try:
            self.llm = create_ollama_client(llm_model, ollama_base_url, temperature)
            logger.info("ChatOllama 就绪 (%.2fs)", time.time() - _t0)
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
        fname = os.path.basename(file_path)
        logger.info("===== 开始摄入: %s =====", fname)
        _t_start = time.time()

        # 阶段一：解析
        _t0 = time.time()
        parsed = process_heterogeneous_data(file_path)
        logger.info("  解析完成 (%.2fs) | 内容长度: %d chars | 表格: %d 个",
                     time.time() - _t0,
                     len(parsed.get("content", "")),
                     len(parsed.get("tables", [])))

        content = parsed.get("content", "")
        tables = parsed.get("tables", [])
        metadata = parsed.get("metadata", {})

        # 合并元数据（后调用的文件覆盖同名键）
        self._metadata.update(metadata)
        self._metadata.setdefault("source_files", []).append(file_path)

        # 阶段二：清洗
        _t0 = time.time()
        cleaned = clean_text(content)
        logger.info("  清洗完成 (%.2fs) | %d -> %d chars",
                     time.time() - _t0, len(content), len(cleaned))

        # 阶段三：切分
        _t0 = time.time()
        chunks = chunk_text(
            text=cleaned,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            metadata={"source": file_path, **metadata},
        )
        logger.info("  切分完成 (%.2fs) | chunk_size=%d overlap=%d → %d 个文本块",
                     time.time() - _t0, self.chunk_size, self.chunk_overlap,
                     len(chunks))

        self._all_chunks.extend(chunks)
        self._all_tables.extend(tables)

        elapsed = time.time() - _t_start
        logger.info("===== %s 摄入完成 (%.2fs) =====", fname, elapsed)

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

        logger.info("----- 构建索引 -----")
        logger.info("  文本块总数: %d", len(self._all_chunks))

        _t0 = time.time()
        documents = [
            Document(
                page_content=chunk["content"],
                metadata=chunk.get("metadata", {}),
            )
            for chunk in self._all_chunks
        ]

        self.vector_store = FAISS.from_documents(documents, self._embedding_model)
        logger.info("  FAISS 索引完成 (%.2fs) | 向量维度: %d",
                     time.time() - _t0,
                     self.vector_store.index.d if hasattr(self.vector_store, 'index') else '?')

        _t0 = time.time()
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        logger.info("  BM25 索引完成 (%.2fs)", time.time() - _t0)

        logger.info("----- 索引构建完毕 -----")

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
        _t0 = time.time()
        self.vector_store.save_local(str(vector_dir))
        logger.info("FAISS 索引已保存: %s (%.2fs)", vector_dir, time.time() - _t0)

        # BM25 索引（pickle 序列化）
        bm25_dir = base / "bm25"
        bm25_dir.mkdir(parents=True, exist_ok=True)
        with open(bm25_dir / "retriever.pkl", "wb") as f:
            pickle.dump(self.bm25_retriever, f)
        logger.info("BM25 索引已保存: %s", bm25_dir / "retriever.pkl")

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
        logger.info("元数据已保存: %s", meta_path)

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

        _t0 = time.time()
        self.vector_store = FAISS.load_local(
            str(vector_dir),
            self._embedding_model,
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS 索引已载入: %s (%.2fs)", vector_dir, time.time() - _t0)

        bm25_dir = base / "bm25"
        bm25_pkl = bm25_dir / "retriever.pkl"
        if bm25_pkl.exists():
            with open(bm25_pkl, "rb") as f:
                self.bm25_retriever = pickle.load(f)
            logger.info("BM25 索引已载入: %s", bm25_pkl)

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

        logger.info("QueryPipeline 就绪 | top_k=%d rerank=%s", top_k, rerank)

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
        logger.info("===== 查询开始 =====")
        logger.info("  问题: %.120s", query)
        logger.info("  模板: %s", prompt_name)
        if chat_history:
            logger.info("  对话历史: %d 条消息", len(chat_history))
        if time_filter:
            logger.info("  时间过滤: %s ~ %s", time_filter[0], time_filter[1])

        _t_start = time.time()

        # 阶段一：混合检索（向量 + BM25）
        retrieved = self._hybrid_retrieve(query)

        # 阶段二：后过滤
        filtered = self._post_filter(retrieved, time_filter=time_filter)

        # 阶段三：重排序（可选）
        if self.rerank and len(filtered) > 1:
            filtered = self._rerank(query, filtered)

        # 阶段四：拼接上下文
        context = self._format_context(filtered)

        logger.info("  检索结果: %d 条 | 过滤后: %d 条 | 上下文: %d chars",
                     len(retrieved), len(filtered), len(context))

        # 阶段五：LLM 生成
        answer = self._generate(query, context, prompt_name, chat_history)

        elapsed = time.time() - _t_start
        logger.info("===== 查询完成 (%.2fs) | 回答长度: %d chars =====",
                     elapsed, len(answer))

        return answer

    # ------------------------------------------------------------------
    def _hybrid_retrieve(self, query: str) -> List[Document]:
        """双路并行召回 + RRF 融合。"""
        _t0 = time.time()
        vector_results = self.vector_store.similarity_search(query, k=self.top_k)
        _tv = time.time() - _t0

        _t0 = time.time()
        bm25_results = self.bm25_retriever.invoke(query)
        _tb = time.time() - _t0

        merged = _rrf_merge(vector_results, bm25_results, k=self.top_k)

        logger.debug("  检索 | 向量 %d 条 (%.2fs)  BM25 %d 条 (%.2fs) → RRF %d 条",
                      len(vector_results), _tv,
                      len(bm25_results), _tb,
                      len(merged))
        return merged

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

        _before = len(deduped)

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

        logger.debug("  过滤 | 去重后 %d → 时间过滤后 %d", _before, len(deduped))
        return deduped

    # ------------------------------------------------------------------
    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """简单重排序：基于 BM25 打分再排（后续可替换为交叉编码器模型）。"""
        # 若文档数少于 3，不做重排
        if len(docs) <= 2:
            return docs

        _t0 = time.time()

        # 简单策略：按 BM25 相关性降序（利用 BM25Retriever 的能力）
        bm25_scores: Dict[str, float] = {}
        for doc in self.bm25_retriever.invoke(query):
            key = doc.page_content[:50]
            bm25_scores[key] = bm25_scores.get(key, 0) + 1

        def _score(doc: Document) -> float:
            return bm25_scores.get(doc.page_content[:50], 0.0)

        docs_sorted = sorted(docs, key=_score, reverse=True)

        logger.debug("  重排序 (%.2fs)", time.time() - _t0)
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
        _t0 = time.time()

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

        result = chain.invoke({})
        logger.info("  LLM 生成 (%.2fs) | 模板: %s | 输出: %d tokens/chars",
                     time.time() - _t0, prompt_name, len(result))
        return result

    # ------------------------------------------------------------------
    def stream(
        self,
        query: str,
        prompt_name: str = "rag",
        chat_history: Optional[List[Dict[str, str]]] = None,
    ):
        """流式查询，逐 token 产出回答（适用于实时展示）。"""
        logger.info("===== 流式查询开始 =====")
        logger.info("  问题: %.120s", query)

        _t_start = time.time()

        retrieved = self._hybrid_retrieve(query)
        filtered = self._post_filter(retrieved)
        if self.rerank and len(filtered) > 1:
            filtered = self._rerank(query, filtered)
        context = self._format_context(filtered)

        logger.info("  检索结果: %d 条 | 上下文: %d chars",
                     len(filtered), len(context))

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

        token_count = 0
        for chunk in chain.stream({}):
            token_count += len(chunk)
            yield chunk

        elapsed = time.time() - _t_start
        logger.info("===== 流式查询完成 (%.2fs) | 输出: %d tokens =====",
                     elapsed, token_count)


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
        logger.info("初始化 RAGPipeline | chunk_size=%d overlap=%d top_k=%d rerank=%s",
                     chunk_size, chunk_overlap, top_k, rerank)
        self._ingest = IngestPipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._query: Optional[QueryPipeline] = None
        self.top_k = top_k
        self.rerank = rerank

    # ------------------------------------------------------------------
    def ingest(self, file_path: str) -> Dict[str, Any]:
        """摄入并索引单文档。"""
        logger.info("RAGPipeline.ingest: %s", file_path)
        result = self._ingest.run(file_path)
        logger.info("开始构建索引...")
        self._ingest.build_index()
        self._query = QueryPipeline(
            vector_store=self._ingest.vector_store,
            bm25_retriever=self._ingest.bm25_retriever,
            llm=self._ingest.llm,
            top_k=self.top_k,
            rerank=self.rerank,
        )
        logger.info("RAGPipeline 准备就绪 | chunks=%d", self._ingest.chunk_count)
        return result

    # ------------------------------------------------------------------
    def ingest_multiple(self, file_paths: List[str]) -> Dict[str, Any]:
        """批量摄入多个文档后统一建索引。"""
        logger.info("RAGPipeline.ingest_multiple: %d 个文档", len(file_paths))
        combined = {"chunks": [], "chunk_count": 0, "tables": [], "metadata": {}}
        for fp in file_paths:
            result = self._ingest.run(fp)
            combined["chunks"].extend(result["chunks"])
            combined["chunk_count"] += result["chunk_count"]
            combined["tables"].extend(result["tables"])
            combined["metadata"].update(result["metadata"])

        logger.info("所有文档摄入完毕，开始构建索引...")
        self._ingest.build_index()
        self._query = QueryPipeline(
            vector_store=self._ingest.vector_store,
            bm25_retriever=self._ingest.bm25_retriever,
            llm=self._ingest.llm,
            top_k=self.top_k,
            rerank=self.rerank,
        )
        logger.info("RAGPipeline 准备就绪 | 总chunks=%d 总tables=%d",
                     combined["chunk_count"], len(combined["tables"]))
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
        logger.info("RAGPipeline.query")
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
        logger.info("RAGPipeline.stream")
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
