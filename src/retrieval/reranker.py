from typing import List
from langchain_core.documents import Document


class BaseReranker:
    """重排序器的基类"""

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        raise NotImplementedError


class BM25Reranker(BaseReranker):
    """基于 BM25 分数的轻量重排序，适配 BM25StoreManager"""

    def __init__(self, bm25_store):
        """
        :param bm25_store: BM25StoreManager 实例，需实现 .search(query, k) -> List[Dict]
        """
        self.bm25_store = bm25_store

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if len(docs) <= 2:
            return docs

        # 用 BM25 重新检索，命中即加分
        bm25_results = self.bm25_store.search(query, k=len(docs) * 2)
        bm25_scores = {item["content"][:80]: 1.0 for item in bm25_results}

        def _score(doc: Document) -> float:
            return bm25_scores.get(doc.page_content[:80], 0.0)

        return sorted(docs, key=_score, reverse=True)


class CrossEncoderReranker(BaseReranker):
    """交叉编码器重排序（占位，后续可接入 Cohere / BGE-reranker）"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        raise NotImplementedError("需接入具体 reranker 模型")

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if len(docs) <= 2:
            return docs
        if self._model is None:
            self._load_model()
        return docs
