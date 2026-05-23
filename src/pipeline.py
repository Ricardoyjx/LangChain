"""管线编排：串联数据摄取与查询的全流程。

用法示例:
    # 数据摄取
    pipeline = IngestPipeline()
    result = pipeline.run("data/raw/美的2025年报.pdf")

    # 构建索引
    pipeline.build_index()

    # 查询
    qpipeline = QueryPipeline(
        hybrid_search=pipeline.hybrid_search,
        llm=pipeline.llm,
    )
    answer = qpipeline.run("美的2024年营收是多少？")
    print(answer)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_ollama import OllamaEmbeddings

# === 旧 imports (FAISS / BM25Retriever, 已迁移至 src.retrieval) ===
# from langchain_community.retrievers import BM25Retriever
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document
# from langchain_core.language_models import BaseLanguageModel
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableParallel
# from langchain_ollama import OllamaEmbeddings

from src.data_processing.cleaner import clean_text
from src.data_processing.chunker import chunk_text
from src.data_processing.parsers.factory import process_heterogeneous_data
from src.generation import create_ollama_client, get_prompt
from src.retrieval import (
    VectorStore,
    BM25StoreManager,
    HybridSearchManager,
    PostFilterChain,
    DeduplicationFilter,
    TimeRangeFilter,
    BM25Reranker,
)

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
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------

_DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
_DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:9b")
_DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
_DEFAULT_MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
_DEFAULT_MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "default")

# _RRF_CONSTANT = 60  # 旧版 RRF 常数，已由 HybridSearchManager 内部管理


# ===================================================================
# 旧版 RRF 融合函数（已迁移至 HybridSearchManager._rrf_fusion）
# ===================================================================
# def _rrf_merge(
#     vector_results: List[Document],
#     bm25_results: List[Document],
#     k: int = 5,
#     constant: int = _RRF_CONSTANT,
# ) -> List[Document]:
#     scores: Dict[str, float] = {}
#     for i, doc in enumerate(vector_results):
#         doc_id = doc.metadata.get("chunk_index", id(doc))
#         scores[str(doc_id)] = scores.get(str(doc_id), 0.0) + 1.0 / (i + constant)
#     for i, doc in enumerate(bm25_results):
#         doc_id = doc.metadata.get("chunk_index", id(doc))
#         scores[str(doc_id)] = scores.get(str(doc_id), 0.0) + 1.0 / (i + constant)
#     seen: set[str] = set()
#     merged: List[Document] = []
#     for doc in vector_results + bm25_results:
#         content = doc.page_content
#         if content not in seen:
#             seen.add(content)
#             merged.append(doc)
#     merged.sort(key=lambda d: scores.get(str(d.metadata.get("chunk_index", id(d))), 0.0), reverse=True)
#     return merged[:k]


# ===================================================================
# 数据摄取管线
# ===================================================================


class IngestPipeline:
    """将原始文档经过 解析 → 清洗 → 切分 → 向量化索引 的全流程。

    向量存储使用 Milvus（增量写入），关键词检索使用 BM25 (bm25s + jieba)。
    支持多文档增量摄入（多次 ``run`` 后调用 ``build_index`` 构建 BM25 索引）。
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embedding_model_name: str = _DEFAULT_EMBEDDING_MODEL,
        llm_model: str = _DEFAULT_LLM_MODEL,
        ollama_base_url: str = _DEFAULT_OLLAMA_URL,
        temperature: float = _DEFAULT_TEMPERATURE,
        milvus_uri: str = _DEFAULT_MILVUS_URI,
        milvus_collection: str = _DEFAULT_MILVUS_COLLECTION,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.info(
            "初始化 IngestPipeline | "
            "chunk_size=%d chunk_overlap=%d embedding=%s llm=%s milvus=%s/%s",
            chunk_size,
            chunk_overlap,
            embedding_model_name,
            llm_model,
            milvus_uri,
            milvus_collection,
        )

        self._all_chunks: List[Dict[str, Any]] = []
        self._all_tables: List[Any] = []
        self._metadata: Dict[str, Any] = {}
        self._embedding_model_name = embedding_model_name

        # ---------- 初始化嵌入模型 ----------
        _t0 = time.time()
        try:
            self._embedding_model = OllamaEmbeddings(model=embedding_model_name)
            logger.info("OllamaEmbeddings 就绪 (%.2fs)", time.time() - _t0)
        except Exception as e:
            raise ConnectionError(
                f"Ollama 嵌入模型初始化失败 (model={embedding_model_name!r}): {e}"
            ) from e

        # ---------- 初始化 LLM ----------
        _t0 = time.time()
        try:
            self.llm = create_ollama_client(llm_model, ollama_base_url, temperature)
            logger.info("ChatOllama 就绪 (%.2fs)", time.time() - _t0)
        except Exception as e:
            raise ConnectionError(
                f"Ollama LLM 初始化失败 (model={llm_model!r}): {e}"
            ) from e

        # ---------- 初始化 Milvus 向量存储 ----------
        _t0 = time.time()
        try:
            self.vector_store = VectorStore(
                uri=milvus_uri,
                collection_name=milvus_collection,
                embedding_function=self._embedding_model,
            )
            logger.info(
                "Milvus 就绪 (%.2fs) | collection=%s",
                time.time() - _t0,
                milvus_collection,
            )
        except Exception as e:
            raise ConnectionError(f"Milvus 连接失败 (uri={milvus_uri!r}): {e}") from e

        # BM25 索引（run/build_index 后可用）
        self.bm25_store: Optional[BM25StoreManager] = None
        self.hybrid_search: Optional[HybridSearchManager] = None

        # === 旧版 __init__ 记录（FAISS 版本，已迁移） ===
        # self.vector_store: Optional[FAISS] = None
        # self.bm25_retriever: Optional[BM25Retriever] = None

    # ------------------------------------------------------------------
    def run(self, file_path: str) -> Dict[str, Any]:
        """执行单文档摄取：解析 → 清洗 → 切分 → 写入 Milvus（增量）。

        Args:
            file_path: 文档路径（PDF / Word / Excel）。

        Returns:
            { "chunks", "chunk_count", "tables", "metadata" }
        """
        fname = os.path.basename(file_path)
        logger.info("===== 开始摄入: %s =====", fname)
        _t_start = time.time()

        # 阶段一：解析
        _t0 = time.time()
        parsed = process_heterogeneous_data(file_path)
        logger.info(
            "  解析完成 (%.2fs) | 内容: %d chars | 表格: %d 个",
            time.time() - _t0,
            len(parsed.get("content", "")),
            len(parsed.get("tables", [])),
        )

        content = parsed.get("content", "")
        tables = parsed.get("tables", [])
        metadata = parsed.get("metadata", {})

        self._metadata.update(metadata)
        self._metadata.setdefault("source_files", []).append(file_path)

        # 阶段二：清洗
        _t0 = time.time()
        cleaned = clean_text(content)
        logger.info(
            "  清洗完成 (%.2fs) | %d -> %d chars",
            time.time() - _t0,
            len(content),
            len(cleaned),
        )

        # 阶段三：切分
        _t0 = time.time()
        chunks = chunk_text(
            text=cleaned,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            metadata={"source": file_path, **metadata},
        )
        logger.info(
            "  切分完成 (%.2fs) | chunk_size=%d overlap=%d → %d 个文本块",
            time.time() - _t0,
            self.chunk_size,
            self.chunk_overlap,
            len(chunks),
        )

        # 阶段四：写入 Milvus（增量写入，立即持久化）
        _t0 = time.time()
        documents = [
            Document(page_content=chunk["content"], metadata=chunk.get("metadata", {}))
            for chunk in chunks
        ]
        try:
            self.vector_store.add_documents(documents)
        except Exception as _conn_err:
            if "ConnectionNotExistException" in type(
                _conn_err
            ).__name__ or "should create connection first" in str(_conn_err):
                raise ConnectionError(
                    "Milvus 服务未启动。请先启动 Milvus：\n"
                    "  cd docker && docker compose up -d\n"
                    "等待约 30 秒后再重新运行。"
                ) from _conn_err
            raise
        logger.info("  写入 Milvus (%.2fs) | %d 条", time.time() - _t0, len(documents))

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
        """基于已摄入的文本块构建 BM25 关键词索引。

        向量索引已在 run() 时写入 Milvus，此处只需构建 BM25。
        build_index 后可通过 ``self.hybrid_search`` 使用混合检索。
        """
        if not self._all_chunks:
            raise ValueError("没有文本块可索引，请先调用 run() 摄入文档。")

        logger.info("----- 构建 BM25 索引 -----")
        logger.info("  文本块总数: %d", len(self._all_chunks))

        _t0 = time.time()
        corpus = [chunk["content"] for chunk in self._all_chunks]
        self.bm25_store = BM25StoreManager(corpus=corpus)
        logger.info("  BM25 索引完成 (%.2fs)", time.time() - _t0)

        # 组装混合检索管理器
        self.hybrid_search = HybridSearchManager(
            vector_store_manager=self.vector_store,
            keyword_store_manager=self.bm25_store,
        )
        logger.info("  HybridSearchManager 就绪")
        logger.info("----- 索引构建完毕 -----")

        # === 旧版 build_index (FAISS + BM25Retriever) ===
        # documents = [
        #     Document(page_content=chunk["content"], metadata=chunk.get("metadata", {}))
        #     for chunk in self._all_chunks
        # ]
        # self.vector_store = FAISS.from_documents(documents, self._embedding_model)
        # self.bm25_retriever = BM25Retriever.from_documents(documents)

    # ------------------------------------------------------------------
    def save_index(self, directory: str) -> None:
        """持久化 BM25 索引到磁盘。

        Milvus 向量数据由服务端自行持久化，无需额外保存。
        """
        if self.bm25_store is None:
            raise ValueError("BM25 索引未构建，请先调用 build_index()。")

        base = Path(directory)
        base.mkdir(parents=True, exist_ok=True)

        # 保存 BM25 索引
        self.bm25_store.save_index()
        index_path = Path(self.bm25_store.index_path)
        if index_path.exists():
            target = base / "bm25"
            if target.exists():
                import shutil

                shutil.rmtree(target)
            index_path.rename(target)
            logger.info("BM25 索引已保存: %s", target)

        # chunks 元数据
        meta_path = base / "chunks_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "chunk_count": len(self._all_chunks),
                    "source_files": self._metadata.get("source_files", []),
                    "embedding_model": self._embedding_model_name,
                    "milvus_collection": self.vector_store.collection_name,
                    "milvus_uri": self.vector_store.uri,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("元数据已保存: %s", meta_path)

        # === 旧版 save_index (FAISS) ===
        # vector_dir = base / "vector_db"
        # self.vector_store.save_local(str(vector_dir))
        # bm25_dir = base / "bm25"
        # bm25_dir.mkdir(parents=True, exist_ok=True)
        # with open(bm25_dir / "retriever.pkl", "wb") as f:
        #     pickle.dump(self.bm25_retriever, f)

    # ------------------------------------------------------------------
    def load_index(self, directory: str) -> None:
        """从磁盘载入 BM25 索引（Milvus 自动重连已有的 collection）。"""
        base = Path(directory)

        bm25_dir = base / "bm25"
        if not bm25_dir.exists():
            raise FileNotFoundError(f"BM25 索引目录不存在: {bm25_dir}")

        self.bm25_store = BM25StoreManager(index_path=str(bm25_dir))
        self.bm25_store.load_index()
        logger.info("BM25 索引已载入: %s", bm25_dir)

        self.hybrid_search = HybridSearchManager(
            vector_store_manager=self.vector_store,
            keyword_store_manager=self.bm25_store,
        )
        logger.info("HybridSearchManager 已恢复")

        # 加载 chunks 元数据（用于展示文档/文本块数等信息）
        meta_path = base / "chunks_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._metadata = meta.get("source_files", [])
            if isinstance(self._metadata, list):
                self._metadata = {"source_files": self._metadata}
            self._all_chunks = [{}] * meta.get("chunk_count", 0)
            logger.info(
                "元数据已加载: %d 个文本块, %s",
                meta.get("chunk_count", 0),
                meta.get("source_files", []),
            )

        # === 旧版 load_index (FAISS) ===
        # self.vector_store = FAISS.load_local(str(vector_dir), self._embedding_model, allow_dangerous_deserialization=True)
        # with open(bm25_pkl, "rb") as f:
        #     self.bm25_retriever = pickle.load(f)

    # ------------------------------------------------------------------
    @property
    def chunk_count(self) -> int:
        return len(self._all_chunks)


# ===================================================================
# 查询管线
# ===================================================================


class QueryPipeline:
    """用户问题 → 混合检索 → 后过滤 → 重排序 → LLM 生成。

    用法示例:
        qpipeline = QueryPipeline(
            hybrid_search=ingest.hybrid_search,
            llm=ingest.llm,
        )
        answer = qpipeline.run("美的2024年营收是多少？")
    """

    def __init__(
        self,
        hybrid_search: HybridSearchManager,
        llm: BaseLanguageModel,
        top_k: int = 5,
        post_filter_chain: Optional[PostFilterChain] = None,
        reranker: Optional[BM25Reranker] = None,
    ):
        self.hybrid_search = hybrid_search
        self.llm = llm
        self.top_k = top_k
        self.post_filter_chain = post_filter_chain or PostFilterChain()
        self.reranker = reranker

        logger.info(
            "QueryPipeline 就绪 | top_k=%d rerank=%s filters=%d",
            top_k,
            reranker is not None,
            len(self.post_filter_chain.filters),
        )

        # === 旧版 __init__ (FAISS + BM25Retriever) ===
        # self.vector_store = vector_store
        # self.bm25_retriever = bm25_retriever
        # self.rerank = rerank

    # ------------------------------------------------------------------
    def run(
        self,
        query: str,
        prompt_name: str = "rag",
        chat_history: Optional[List[Dict[str, str]]] = None,
        time_filter: Optional[Tuple[str, str]] = None,
    ) -> str:
        logger.info("===== 查询开始 =====")
        logger.info("  问题: %.120s", query)
        logger.info("  模板: %s", prompt_name)

        _t_start = time.time()

        # 阶段一：混合检索
        retrieved = self._hybrid_retrieve(query)

        # 阶段二：后过滤
        filtered = self._post_filter(retrieved, time_filter=time_filter)

        # 阶段三：重排序
        if self.reranker and len(filtered) > 1:
            filtered = self.reranker.rerank(query, filtered)

        # 阶段四：拼上下文
        context = self._format_context(filtered)

        logger.info(
            "  检索: %d 条 → 过滤后: %d 条 | 上下文: %d chars",
            len(retrieved),
            len(filtered),
            len(context),
        )

        # 阶段五：LLM 生成
        answer = self._generate(query, context, prompt_name, chat_history)

        elapsed = time.time() - _t_start
        logger.info(
            "===== 查询完成 (%.2fs) | 回答: %d chars =====", elapsed, len(answer)
        )
        return answer

    # ------------------------------------------------------------------
    def _hybrid_retrieve(self, query: str) -> List[Document]:
        """向量 + BM25 双路召回 + RRF 融合。"""
        _t0 = time.time()
        results = self.hybrid_search.search(query, top_k=self.top_k)
        logger.debug("  混合检索 (%.2fs) → %d 条", time.time() - _t0, len(results))
        return results

        # === 旧版 _hybrid_retrieve (内联 FAISS + BM25 + _rrf_merge) ===
        # vector_results = self.vector_store.similarity_search(query, k=self.top_k)
        # bm25_results = self.bm25_retriever.invoke(query)
        # merged = _rrf_merge(vector_results, bm25_results, k=self.top_k)
        # return merged

    # ------------------------------------------------------------------
    def _post_filter(
        self,
        docs: List[Document],
        time_filter: Optional[Tuple[str, str]] = None,
    ) -> List[Document]:
        """通过 PostFilterChain 执行后过滤。"""
        context: Dict[str, Any] = {}
        if time_filter:
            context["time_filter"] = time_filter
        return self.post_filter_chain.process(docs, context)

        # === 旧版 _post_filter (内联去重+时效) ===
        # seen: set[str] = set()
        # deduped = []
        # for doc in docs:
        #     key = doc.page_content[:50]
        #     if key not in seen:
        #         seen.add(key); deduped.append(doc)
        # if time_filter:
        #     start_str, end_str = time_filter
        #     filtered = [d for d in deduped if start_str <= d.metadata.get("date", "") <= end_str]
        #     if filtered: deduped = filtered
        # return deduped

    # ------------------------------------------------------------------
    def _format_context(self, docs: List[Document]) -> str:
        """拼接 LLM 上下文，前缀标注 [来源 i]。"""
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "")
            tag = f"[来源 {i}] " if source else ""
            parts.append(f"{tag}{doc.page_content}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    def _generate(
        self,
        query: str,
        context: str,
        prompt_name: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        _t0 = time.time()

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
        logger.info(
            "  LLM 生成 (%.2fs) | 模板: %s | 输出: %d chars",
            time.time() - _t0,
            prompt_name,
            len(result),
        )
        return result

    # ------------------------------------------------------------------
    def stream(
        self,
        query: str,
        prompt_name: str = "rag",
        chat_history: Optional[List[Dict[str, str]]] = None,
    ):
        """流式查询，逐 token 产出回答。"""
        logger.info("===== 流式查询开始 =====")
        logger.info("  问题: %.120s", query)

        _t_start = time.time()

        retrieved = self._hybrid_retrieve(query)
        filtered = self._post_filter(retrieved)
        if self.reranker and len(filtered) > 1:
            filtered = self.reranker.rerank(query, filtered)
        context = self._format_context(filtered)

        logger.info(
            "  检索: %d 条 → 过滤后: %d 条 | 上下文: %d chars",
            len(retrieved),
            len(filtered),
            len(context),
        )

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
        logger.info(
            "===== 流式查询完成 (%.2fs) | 输出: %d tokens =====", elapsed, token_count
        )

        # === 旧版 stream 中内联的检索/过滤/重排逻辑同上 ===

    # === 旧版 _rerank (内联 BM25 重排，已迁移至 BM25Reranker) ===
    # def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
    #     if len(docs) <= 2: return docs
    #     bm25_scores = {}
    #     for doc in self.bm25_retriever.invoke(query):
    #         key = doc.page_content[:50]
    #         bm25_scores[key] = bm25_scores.get(key, 0) + 1
    #     def _score(doc): return bm25_scores.get(doc.page_content[:50], 0.0)
    #     return sorted(docs, key=_score, reverse=True)


# ===================================================================
# 一站式 RAG 管线
# ===================================================================


class RAGPipeline:
    """封装 IngestPipeline + QueryPipeline，快速上手。

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
        **kwargs,
    ):
        logger.info(
            "初始化 RAGPipeline | chunk_size=%d overlap=%d top_k=%d rerank=%s",
            chunk_size,
            chunk_overlap,
            top_k,
            rerank,
        )
        self._ingest = IngestPipeline(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs
        )
        self._query: Optional[QueryPipeline] = None
        self.top_k = top_k
        self.rerank = rerank

        # === 旧版 RAGPipeline.__init__ 无 **kwargs 透传，不包含 milvus 参数 ===

    # ------------------------------------------------------------------
    def ingest(self, file_path: str) -> Dict[str, Any]:
        """摄入单文档并构建索引。"""
        logger.info("RAGPipeline.ingest: %s", file_path)
        result = self._ingest.run(file_path)
        logger.info("开始构建索引...")
        self._ingest.build_index()

        # 装配后过滤链
        chain = PostFilterChain()
        chain.add_filter(DeduplicationFilter())
        chain.add_filter(TimeRangeFilter())
        # 可根据需要添加 FreshnessFilter、PermissionFilter 等

        # 装配重排序器
        reranker = BM25Reranker(self._ingest.bm25_store) if self.rerank else None

        self._query = QueryPipeline(
            hybrid_search=self._ingest.hybrid_search,
            llm=self._ingest.llm,
            top_k=self.top_k,
            post_filter_chain=chain,
            reranker=reranker,
        )
        logger.info("RAGPipeline 准备就绪 | chunks=%d", self._ingest.chunk_count)
        return result

        # === 旧版 ingest (FAISS + BM25Retriever 直接传参) ===
        # self._ingest.build_index()
        # self._query = QueryPipeline(
        #     vector_store=self._ingest.vector_store,
        #     bm25_retriever=self._ingest.bm25_retriever,
        #     llm=self._ingest.llm, top_k=self.top_k, rerank=self.rerank,
        # )

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

        chain = PostFilterChain()
        chain.add_filter(DeduplicationFilter())
        chain.add_filter(TimeRangeFilter())

        reranker = BM25Reranker(self._ingest.bm25_store) if self.rerank else None

        self._query = QueryPipeline(
            hybrid_search=self._ingest.hybrid_search,
            llm=self._ingest.llm,
            top_k=self.top_k,
            post_filter_chain=chain,
            reranker=reranker,
        )
        logger.info(
            "RAGPipeline 准备就绪 | 总chunks=%d 总tables=%d",
            combined["chunk_count"],
            len(combined["tables"]),
        )
        return combined

        # === 旧版 ingest_multiple (FAISS + BM25Retriever) ===
        # self._ingest.build_index()
        # self._query = QueryPipeline(
        #     vector_store=self._ingest.vector_store,
        #     bm25_retriever=self._ingest.bm25_retriever,
        #     llm=self._ingest.llm, top_k=self.top_k, rerank=self.rerank,
        # )

    # ------------------------------------------------------------------
    def query(
        self,
        question: str,
        prompt_name: str = "rag",
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        if self._query is None:
            raise RuntimeError("请先调用 ingest() 摄入文档。")
        return self._query.run(
            question, prompt_name=prompt_name, chat_history=chat_history
        )

    # ------------------------------------------------------------------
    def stream(
        self,
        question: str,
        prompt_name: str = "rag",
        chat_history: Optional[List[Dict[str, str]]] = None,
    ):
        if self._query is None:
            raise RuntimeError("请先调用 ingest() 摄入文档。")
        yield from self._query.stream(
            question, prompt_name=prompt_name, chat_history=chat_history
        )

    # ------------------------------------------------------------------
    @property
    def ingest_pipeline(self) -> IngestPipeline:
        return self._ingest

    @property
    def query_pipeline(self) -> Optional[QueryPipeline]:
        return self._query
