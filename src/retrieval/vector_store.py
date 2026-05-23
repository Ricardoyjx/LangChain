from typing import List, Optional
from langchain_milvus import Milvus
from pymilvus import PyMilvusDeprecationWarning
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import logging
import warnings

logger = logging.getLogger(__name__)

# 忽略 PyMilvus 的弃用警告
warnings.filterwarnings("ignore", category=PyMilvusDeprecationWarning)


class VectorStore:
    def __init__(
        self,
        uri: str = "http://localhost:19530",
        collection_name: str = "default",
        embedding_function: Optional[Embeddings] = None,
    ):
        """
        初始化 Milvus 向量数据库管理器
        :param uri: Milvus 服务器的 URI
        :param collection_name: 集合名称
        :param embedding_function: 嵌入模型，默认使用 DashScope（阿里通义千问）
        """
        self.embedding_function = embedding_function or DashScopeEmbeddings()
        self.collection_name = collection_name
        self.uri = uri
        # Milvus 构造函数用 connection_args 传 uri，而非直接传 uri 参数
        self.vector_store = Milvus(
            collection_name=collection_name,
            embedding_function=self.embedding_function,
            connection_args={"uri": uri},
        )

    def add_documents(self, documents: List[Document], ids: List[str] = None):
        if ids is None:
            ids = [f"id_{i}" for i in range(len(documents))]
        self.vector_store.add_documents(documents=documents, ids=ids)
        logger.info(
            "Added %d documents to Milvus collection '%s'",
            len(documents),
            self.collection_name,
        )

    def delete_documents(self, ids: List[str]):
        self.vector_store.delete(ids)
        logger.info("Deleted documents IDs: %s", ids)

    def update_documents(self, documents: List[Document], ids: List[str]):
        self.delete_documents(ids)
        self.add_documents(documents, ids)

    def search_documents(
        self, query: str, k: int = 3, filter_dict: dict = None
    ) -> List[Document]:
        """按语义相似度检索"""
        return self.vector_store.similarity_search(query=query, k=k, filter=filter_dict)

    # HybridSearchManager 期望的 search() 接口
    search = search_documents
