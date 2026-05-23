# hybrid_search.py
from typing import List, Dict, Any
from langchain_core.documents import Document

# 💡 核心逻辑解读
# RRF 算法的魔力：
# 向量检索输出的是“余弦相似度”（0到1之间），而 BM25 输出的是“词频统计分数”（可能是几十甚至上百）。这两种分数完全不在一个量级，无法直接相加。
# RRF 巧妙地避开了绝对分数，只看排名（Rank）。公式 1 / (k + rank + 1) 意味着：无论底层的分数是多少，只要你在各自的检索结果中排在第 1 名，贡献的 RRF 分数就是最高的。这使得两路结果可以公平地放在一起排序。
# 参数 k 的作用：
# 代码中的 k=60 是 RRF 论文中的经典经验值。它的作用是平滑排名带来的分数差异。k 值越大，排名第 1 和排名第 10 的差距就越小；k 值越小，排名靠前的文档优势就越明显。通常情况下，保持 60 即可满足绝大多数 RAG 场景。
# 召回数量（vector_k / keyword_k）：
# 在混合检索中，我们通常会先从两路各自多召回一些数据（比如各召回 10 条或 20 条），经过 RRF 融合打分后，再截取最靠前的 top_k（比如 3 条或 5 条）喂给大模型。这样可以最大程度地避免漏掉相关的重要信息。


class HybridSearchManager:
    def __init__(self, vector_store_manager, keyword_store_manager, k: int = 60):
        """
        初始化混合检索管理器
        :param vector_store_manager: 之前封装的 VectorStoreManager 实例
        :param keyword_store_manager: 之前封装的 BM25StoreManager 实例
        :param k: RRF 算法的平滑参数，经验值通常为 60
        """
        self.vector_store = vector_store_manager
        self.keyword_store = keyword_store_manager
        self.k = k

    def _rrf_fusion(
        self, vector_docs: List[Document], keyword_docs: List[Dict]
    ) -> List[Document]:
        """
        RRF (Reciprocal Rank Fusion) 倒数排名融合算法核心实现
        核心思想：不比绝对分数，只比排名。排名越靠前，贡献的倒数分数越高。
        """
        rrf_scores: Dict[str, float] = {}

        # 1. 累加向量检索结果的 RRF 分数
        for rank, doc in enumerate(vector_docs):
            # 使用文档内容作为唯一标识（实际业务中建议使用 doc.metadata.get('id')）
            doc_id = doc.page_content
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (self.k + rank + 1)

        # 2. 累加关键词检索结果的 RRF 分数
        for rank, item in enumerate(keyword_docs):
            doc_id = item["content"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (self.k + rank + 1)

        # 3. 按照 RRF 综合得分降序排列，并重建 Document 对象返回
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # 将融合后的结果转换回 LangChain 的 Document 格式
        final_docs = []
        for doc_content, score in sorted_docs:
            # 从两个检索器中找回原始的 Document 对象（这里简单重建）
            final_docs.append(
                Document(page_content=doc_content, metadata={"rrf_score": score})
            )

        return final_docs

    def search(
        self, query: str, top_k: int = 5, vector_k: int = 10, keyword_k: int = 10
    ) -> List[Document]:
        """
        执行混合检索
        :param query: 用户查询文本
        :param top_k: 最终返回的融合后结果数量
        :param vector_k: 向量检索召回的初始数量（通常设大一点，给 RRF 更多候选）
        :param keyword_k: 关键词检索召回的初始数量
        """
        # 1. 并行执行两路检索
        print(f"🔍 正在执行混合检索：Query='{query}'")
        vector_results = self.vector_store.search(query, k=vector_k)
        keyword_results = self.keyword_store.search(query, k=keyword_k)

        print(f"   - 向量检索召回 {len(vector_results)} 条")
        print(f"   - 关键词检索召回 {len(keyword_results)} 条")

        if not vector_results and not keyword_results:
            return []

        # 2. 使用 RRF 算法进行融合重排
        fused_docs = self._rrf_fusion(vector_results, keyword_results)

        # 3. 返回最终的 Top-K 结果
        return fused_docs[:top_k]
