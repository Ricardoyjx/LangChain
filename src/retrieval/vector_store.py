# vector_store.py
from langchain_milvus import Milvus
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
import logging

logging.basicConfig(level=logging.INFO)


class VectorStore:
    def __init__(self, uri="http://localhost:19530", collection_name="default"):
        """
        初始化向量数据库管理器（这里以 milvus + 阿里通义千问嵌入模型为例）
        :param uri: Milvus 服务器的 URI
        :param collection_name: 集合名称，类似于数据库中的表名
        """
        self.embedding_function = DashScopeEmbeddings()
        self.vector_store = Milvus(
            collection_name=collection_name,
            embedding_function=self.embedding_function,
            uri=uri,
        )

    def add_documents(self, documents: list[Document], ids: list[str] = None):
        """
        增（Create）：添加文档到向量数据库
        :param documents: 文档列表
        :param ids: 对应的文档ID列表
        """
        if ids is None:
            ids = [f"id_{i}" for i in range(len(documents))]
        self.vector_store.add_documents(documents=documents, ids=ids)
        print(f"Added {len(documents)} documents to the vector store.")
        logging.info(f"Added documents with IDs: {ids}")

    def delete_documents(self, ids: list[str]):
        """
        删（Delete）：根据ID删除文档
        :param ids: 需要删除的文档ID列表
        """
        self.vector_store.delete(ids)
        print(f"成功删除ID为 {ids} 的文档。")
        logging.info(f"Deleted documents with IDs: {ids}")

    def update_documents(self, documents: list[Document], ids: list[str]):
        """
        改（Update）：更新文档（在向量库中通常表现为先删后增）
        :param documents: 更新后的文档列表
        :param ids: 对应的文档ID列表
        """
        self.delete_documents(ids)
        self.add_documents(documents, ids)
        print(f"成功更新ID为 {ids} 的文档。")
        logging.info(f"Updated documents with IDs: {ids}")

    def search_documents(self, query: str, k: int = 3, filter_dict: dict = None):
        """
        查（Read）：根据语义相似度检索文档
        :param query: 用户的查询文本
        :param k: 返回最相似的前 k 个结果
        :param filter_dict: 过滤条件（如 {"source": "黑马程序员"}）
        """
        result = self.vector_store.similarity_search(
            query=query, k=k, filter=filter_dict
        )
        logging.info(f"Search results for query '{query}': {result}")
        return result
