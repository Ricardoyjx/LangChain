from typing import List, Optional, Dict, Any
from pymilvus import MilvusClient
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

# ===================================================================
# Milvus 向量存储（基于 pymilvus.MilvusClient，pymilvus 3.x API）
# ===================================================================


class VectorStore:
    """Milvus 向量数据库管理器。

    使用 ``pymilvus.MilvusClient`` 直接操作 Milvus（不依赖 langchain-milvus）。
    嵌入由外部 ``embedding_function`` 完成，存储 + 检索均返回 ``Document``。

    用法:
        store = VectorStore(
            uri="http://localhost:19530",
            collection_name="my_docs",
            embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
        )
        store.add_documents(documents)
        results = store.search("美的2024年营收", k=5)
    """

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        collection_name: str = "default",
        embedding_function: Optional[Embeddings] = None,
    ):
        """
        :param uri:            Milvus 服务地址（远程: http://host:port, 本地文件: ./milvus.db）
        :param collection_name: 集合名称
        :param embedding_function: 嵌入模型，默认使用 DashScope（阿里通义千问）
        """
        self.uri = uri
        self.collection_name = collection_name
        self.embedding_function = embedding_function or DashScopeEmbeddings()

        # 惰性连接：首次操作时初始化 client，首次插入时确定维度并建集合
        self._client: Optional[MilvusClient] = None
        self._dimension: Optional[int] = None
        self._collection_ensured: bool = False

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    @property
    def client(self) -> MilvusClient:
        """惰性获取 MilvusClient 实例。"""
        if self._client is None:
            logger.info("Connecting to Milvus: uri=%s", self.uri)
            self._client = MilvusClient(uri=self.uri)
            # 验证连接——执行一个轻量操作
            try:
                self._client.list_collections()
            except Exception as e:
                raise ConnectionError(
                    f"Milvus 连接失败 (uri={self.uri!r}): {e}\n"
                    f"请确认 Milvus 服务已启动：\n"
                    f"  cd docker && docker compose up -d\n"
                    f"等待约 30 秒后再重新运行。"
                ) from e
        return self._client

    def _get_dimension(self) -> int:
        """嵌入一条样本文本以探测向量维度。"""
        if self._dimension is not None:
            return self._dimension
        sample = self.embedding_function.embed_query("dimension probe")
        self._dimension = len(sample)
        logger.debug("Detected embedding dimension: %d", self._dimension)
        return self._dimension

    def _ensure_collection(self) -> None:
        """确保 Milvus 集合已存在，不存在则创建（自动推断维度）。"""
        if self._collection_ensured:
            return

        if self.client.has_collection(self.collection_name):
            logger.info(
                "Milvus collection '%s' already exists", self.collection_name
            )
            self._collection_ensured = True
            return

        dim = self._get_dimension()
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=dim,
            auto_id=True,               # 自动生成主键
            enable_dynamic_field=True,  # 动态 schema，允许任意 metadata 字段
        )
        logger.info(
            "Created Milvus collection '%s' (dim=%d, auto_id=True)",
            self.collection_name,
            dim,
        )
        self._collection_ensured = True

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def add_documents(
        self, documents: List[Document], ids: Optional[List[str]] = None
    ) -> None:
        """将文档嵌入后写入 Milvus。

        .. note::
            ``ids`` 参数保留用于兼容旧接口；开启 ``auto_id`` 后由 Milvus 自动分配。
        """
        self._ensure_collection()

        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_function.embed_documents(texts)

        data: List[Dict[str, Any]] = []
        for i, doc in enumerate(documents):
            entry: Dict[str, Any] = {
                "vector": embeddings[i],
                "text": doc.page_content,
            }
            # 展平 metadata：简单类型直接写入，复杂类型序列化为 JSON
            for k, v in doc.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    entry[k] = v
                else:
                    entry[k] = _safe_serialize(v)

            data.append(entry)

        self.client.insert(
            collection_name=self.collection_name,
            data=data,
        )
        logger.info(
            "Added %d documents to Milvus collection '%s'",
            len(documents),
            self.collection_name,
        )

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search_documents(
        self, query: str, k: int = 3, filter_dict: Optional[dict] = None
    ) -> List[Document]:
        """按语义相似度检索（HybridSearchManager 优先使用 ``search`` 方法）。

        :param query: 查询文本
        :param k:     返回条数
        :param filter_dict: Milvus 标量过滤表达式（暂未实现，预留接口）
        """
        self._ensure_collection()
        query_vector = self.embedding_function.embed_query(query)

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=k,
            output_fields=["*"],  # 返回所有字段
        )

        return _parse_search_results(results)

    # ------------------------------------------------------------------
    # 删除 / 更新
    # ------------------------------------------------------------------

    def delete_documents(self, ids: List[str]) -> None:
        """按主键删除文档。"""
        self.client.delete(
            collection_name=self.collection_name,
            ids=ids,
        )
        logger.info("Deleted %d documents from Milvus", len(ids))

    def update_documents(
        self, documents: List[Document], ids: List[str]
    ) -> None:
        """先删后增（更新文档向量和元数据）。"""
        self.delete_documents(ids)
        self.add_documents(documents, ids)

    # ------------------------------------------------------------------
    # 别名：HybridSearchManager.search() 兼容
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 3) -> List[Document]:
        """HybridSearchManager 期望的 ``search(query, k)`` 接口。"""
        return self.search_documents(query, k=k)


# ===================================================================
# 帮助函数
# ===================================================================


def _safe_serialize(value: Any) -> str:
    """将非标量类型序列化为 JSON 字符串。"""
    import json

    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _parse_search_results(raw_results) -> List[Document]:
    """将 MilvusClient.search() 的原始返回解析为 ``List[Document]``。"""
    docs: List[Document] = []

    if not raw_results or len(raw_results) == 0:
        return docs

    for hit in raw_results[0]:
        entity: Dict[str, Any] = hit.get("entity", {})
        text: str = entity.pop("text", "") if entity else ""

        # 清理：排除向量和内部 id
        entity.pop("vector", None)
        entity.pop("id", None)

        # 搜索分数写入 metadata
        entity["search_score"] = hit.get("distance", 0.0)

        docs.append(
            Document(page_content=text, metadata=entity)
        )

    return docs
