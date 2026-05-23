from typing import List, Dict
import logging

logger = logging.getLogger(__name__)
from langchain_core.documents import Document


class HybridSearchManager:
    def __init__(self, vector_store_manager, keyword_store_manager, k: int = 60):
        """
        :param vector_store_manager: 实现 .search(query, k) -> List[Document] 的对象
        :param keyword_store_manager: 实现 .search(query, k) -> List[Dict] 的对象
        :param k: RRF 平滑参数，论文经典值 60
        """
        self.vector_store = vector_store_manager
        self.keyword_store = keyword_store_manager
        self.k = k

    def _rrf_fusion(
        self, vector_docs: List[Document], keyword_docs: List[Dict]
    ) -> List[Document]:
        """
        RRF 融合，保留原始 Document 的 metadata
        """
        rrf_scores: Dict[str, float] = {}
        # 用 page_content 建索引，保留完整的 Document 对象
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(vector_docs):
            # page_content 必须是可哈希的字符串，否则无法作为 dict key
            raw = doc.page_content
            doc_id = str(raw) if not isinstance(raw, str) else raw
            if not isinstance(raw, str):
                logger.warning(
                    "RRF: doc.page_content is %s (not str), converted; content=%.60s",
                    type(raw).__name__, doc_id,
                )
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        for rank, item in enumerate(keyword_docs):
            doc_id = item["content"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = Document(page_content=doc_id, metadata={})

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        final_docs = []
        for doc_content, score in sorted_docs:
            original = doc_map[doc_content]
            final_docs.append(
                Document(
                    page_content=original.page_content,
                    metadata={**original.metadata, "rrf_score": score},
                )
            )
        return final_docs

    def search(
        self, query: str, top_k: int = 5, vector_k: int = 10, keyword_k: int = 10
    ) -> List[Document]:
        """执行混合检索（向量 + BM25 + RRF 融合）。"""
        vector_results = self.vector_store.search(query, k=vector_k)
        keyword_results = self.keyword_store.search(query, k=keyword_k)

        if not vector_results and not keyword_results:
            return []

        fused_docs = self._rrf_fusion(vector_results, keyword_results)
        return fused_docs[:top_k]
