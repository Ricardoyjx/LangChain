"""测试检索模块 —— Milvus、BM25、混合检索、重排序、后过滤。

所有外部依赖（Milvus、bm25s、jieba、嵌入模型）均使用 mock，无需真实服务。
"""

from unittest.mock import MagicMock, patch, PropertyMock, call

import pytest
from langchain_core.documents import Document

from src.retrieval import (
    VectorStore,
    BM25StoreManager,
    HybridSearchManager,
    BaseReranker,
    BM25Reranker,
    CrossEncoderReranker,
    BaseFilter,
    DeduplicationFilter,
    PermissionFilter,
    FreshnessFilter,
    TimeRangeFilter,
    PostFilterChain,
)


# ===================================================================
# VectorStore（Milvus 向量存储）
# ===================================================================

class TestVectorStore:
    @pytest.fixture
    def mock_embedding(self):
        emb = MagicMock()
        emb.embed_query.return_value = [0.1] * 128
        emb.embed_documents.return_value = [[0.1] * 128, [0.2] * 128]
        return emb

    @pytest.fixture
    def mock_milvus_client(self):
        client = MagicMock()
        client.list_collections.return_value = []
        client.has_collection.return_value = False
        client.create_collection.return_value = None
        client.insert.return_value = None
        client.search.return_value = [
            [
                {
                    "id": 0,
                    "distance": 0.95,
                    "entity": {
                        "text": "美的2024年营收3721亿元",
                        "source": "年报.pdf",
                        "date": "2025-03-30",
                    },
                },
                {
                    "id": 1,
                    "distance": 0.88,
                    "entity": {
                        "text": "美的2024年净利润385亿元",
                        "source": "年报.pdf",
                        "date": "2025-03-30",
                    },
                },
            ]
        ]
        client.delete.return_value = None
        return client

    @pytest.fixture
    def store(self, mock_embedding, mock_milvus_client):
        with patch(
            "src.retrieval.vector_store.MilvusClient",
            return_value=mock_milvus_client,
        ):
            vs = VectorStore(
                uri="http://test:19530",
                collection_name="test_collection",
                embedding_function=mock_embedding,
            )
            vs._dimension = 128
            vs._collection_ensured = True
            vs._client = mock_milvus_client
            return vs

    # ── 初始化 ────────────────────────────────────────────

    def test_init_defaults(self, mock_embedding):
        with patch("src.retrieval.vector_store.MilvusClient"):
            vs = VectorStore(embedding_function=mock_embedding)
        assert vs.uri == "http://localhost:19530"
        assert vs.collection_name == "default"
        assert vs._client is None
        assert vs._dimension is None
        assert vs._collection_ensured is False

    def test_init_custom(self, mock_embedding):
        with patch("src.retrieval.vector_store.MilvusClient"):
            vs = VectorStore(
                uri="http://custom:19530",
                collection_name="finance",
                embedding_function=mock_embedding,
            )
        assert vs.uri == "http://custom:19530"
        assert vs.collection_name == "finance"

    # ── 惰性连接 ─────────────────────────────────────────

    def test_client_lazy_connect(self, mock_embedding, mock_milvus_client):
        with patch(
            "src.retrieval.vector_store.MilvusClient",
            return_value=mock_milvus_client,
        ):
            vs = VectorStore(embedding_function=mock_embedding)
            assert vs._client is None
            c = vs.client
            assert c is mock_milvus_client
            mock_milvus_client.list_collections.assert_called_once()

    def test_client_connect_failure(self, mock_embedding):
        client = MagicMock()
        client.list_collections.side_effect = RuntimeError("Connect failed")
        with patch(
            "src.retrieval.vector_store.MilvusClient",
            return_value=client,
        ):
            vs = VectorStore(embedding_function=mock_embedding)
            with pytest.raises(ConnectionError, match="Milvus 连接失败"):
                _ = vs.client

    # ── 维度探测 ─────────────────────────────────────────

    def test_get_dimension(self, mock_embedding):
        with patch("src.retrieval.vector_store.MilvusClient"):
            vs = VectorStore(embedding_function=mock_embedding)
            dim = vs._get_dimension()
            assert dim == 128
            mock_embedding.embed_query.assert_called_once_with("dimension probe")

    # ── 集合管理 ─────────────────────────────────────────

    def test_ensure_collection_creates(self, mock_embedding, mock_milvus_client):
        with patch(
            "src.retrieval.vector_store.MilvusClient",
            return_value=mock_milvus_client,
        ):
            vs = VectorStore(embedding_function=mock_embedding)
            vs._dimension = 128
            vs._ensure_collection()
            mock_milvus_client.create_collection.assert_called_once_with(
                collection_name="default",
                dimension=128,
                auto_id=True,
                enable_dynamic_field=True,
            )

    def test_ensure_collection_already_exists(
        self, mock_embedding, mock_milvus_client
    ):
        mock_milvus_client.has_collection.return_value = True
        mock_milvus_client.describe_collection.return_value = {
            "fields": [{"auto_id": True}]
        }
        with patch(
            "src.retrieval.vector_store.MilvusClient",
            return_value=mock_milvus_client,
        ):
            vs = VectorStore(embedding_function=mock_embedding)
            vs._dimension = 128
            vs._ensure_collection()
            mock_milvus_client.create_collection.assert_not_called()

    def test_ensure_collection_drops_incompatible(
        self, mock_embedding, mock_milvus_client
    ):
        mock_milvus_client.has_collection.return_value = True
        mock_milvus_client.describe_collection.return_value = {
            "fields": [{"auto_id": False, "name": "pk"}]
        }
        with patch(
            "src.retrieval.vector_store.MilvusClient",
            return_value=mock_milvus_client,
        ):
            vs = VectorStore(embedding_function=mock_embedding)
            vs._dimension = 128
            vs._ensure_collection()
            mock_milvus_client.drop_collection.assert_called_once_with("default")
            mock_milvus_client.create_collection.assert_called_once()

    # ── 写入 ─────────────────────────────────────────────

    def test_add_documents(self, store, mock_milvus_client):
        docs = [
            Document(page_content="美的2024年营收3721亿元", metadata={"source": "年报.pdf"}),
            Document(page_content="美的2024年净利润385亿元", metadata={"source": "年报.pdf"}),
        ]
        store.add_documents(docs)
        mock_milvus_client.insert.assert_called_once()
        data = mock_milvus_client.insert.call_args[1]["data"]
        assert len(data) == 2
        assert data[0]["text"] == "美的2024年营收3721亿元"
        assert data[0]["source"] == "年报.pdf"
        assert "vector" in data[0]

    def test_add_documents_skips_reserved_fields(self, store, mock_milvus_client):
        docs = [
            Document(
                page_content="test",
                metadata={"text": "clash", "vector": "clash", "id": "clash", "safe": "ok"},
            )
        ]
        store.add_documents(docs)
        data = mock_milvus_client.insert.call_args[1]["data"][0]
        assert data["_meta_text"] == "clash"
        assert data["_meta_vector"] == "clash"
        assert data["_meta_id"] == "clash"
        assert data["safe"] == "ok"

    def test_add_documents_serializes_non_scalar(self, store, mock_milvus_client):
        docs = [
            Document(page_content="test", metadata={"tags": ["a", "b"], "count": 3})
        ]
        store.add_documents(docs)
        data = mock_milvus_client.insert.call_args[1]["data"][0]
        assert isinstance(data["tags"], str)
        assert "a" in data["tags"]

    # ── 检索 ─────────────────────────────────────────────

    def test_search_documents(self, store):
        results = store.search_documents("美的营收", k=2)
        assert len(results) == 2
        assert all(isinstance(d, Document) for d in results)
        assert results[0].page_content == "美的2024年营收3721亿元"
        assert results[0].metadata["source"] == "年报.pdf"
        assert "search_score" in results[0].metadata

    def test_search_alias(self, store):
        results = store.search("美的营收", k=2)
        assert len(results) == 2

    def test_search_empty_results(self, store, mock_milvus_client):
        mock_milvus_client.search.return_value = []
        results = store.search("不存在的文档", k=5)
        assert results == []

    # ── 删除与更新 ──────────────────────────────────────

    def test_delete_documents(self, store, mock_milvus_client):
        store.delete_documents(["id1", "id2"])
        mock_milvus_client.delete.assert_called_once_with(
            collection_name="test_collection", ids=["id1", "id2"],
        )

    def test_update_documents(self, store, mock_milvus_client):
        docs = [Document(page_content="updated content")]
        store.update_documents(docs, ids=["old_id"])
        mock_milvus_client.delete.assert_called_once()
        mock_milvus_client.insert.assert_called_once()


# ===================================================================
# BM25StoreManager（关键词检索）
# ===================================================================

class TestBM25StoreManager:
    @pytest.fixture
    def mock_bm25(self):
        bm25 = MagicMock()
        bm25.retrieve.return_value = (
            [["美的2024年营收3721亿元", "美的2024年净利润385亿元"]],
            [[0.95, 0.88]],
        )
        return bm25

    @pytest.fixture
    def store_with_corpus(self, mock_bm25):
        with (
            patch("src.retrieval.keyword_store.bm25s.BM25", return_value=mock_bm25),
            patch("src.retrieval.keyword_store.jieba.cut", side_effect=lambda x: [x]),
            patch("src.retrieval.keyword_store.Stemmer.Stemmer"),
        ):
            store = BM25StoreManager(
                corpus=["美的2024年营收3721亿元", "美的2024年净利润385亿元"]
            )
            return store

    # ── 初始化 ───────────────────────────────────────────

    def test_init_empty(self):
        store = BM25StoreManager(corpus=[])
        assert store.corpus == []
        assert store.retriever is None

    def test_init_with_corpus(self, mock_bm25):
        with (
            patch("src.retrieval.keyword_store.bm25s.BM25", return_value=mock_bm25),
            patch("src.retrieval.keyword_store.jieba.cut", side_effect=lambda x: [x]),
            patch("src.retrieval.keyword_store.Stemmer.Stemmer"),
        ):
            store = BM25StoreManager(corpus=["doc1", "doc2"])
        assert store.retriever is mock_bm25

    # ── 分词 ─────────────────────────────────────────────

    def test_tokenize_calls_jieba(self):
        with (
            patch("src.retrieval.keyword_store.bm25s.BM25"),
            patch("src.retrieval.keyword_store.jieba.cut", return_value=["我", "爱", "北京"]),
            patch("src.retrieval.keyword_store.Stemmer.Stemmer") as mock_stemmer_cls,
        ):
            stemmer_instance = MagicMock()
            stemmer_instance.stemWord.side_effect = lambda w: w
            mock_stemmer_cls.return_value = stemmer_instance
            store = BM25StoreManager(corpus=["我爱北京"])
            result = store._tokenize(["我爱北京"])
            assert result == [["我", "爱", "北京"]]

    # ── 索引 ─────────────────────────────────────────────

    def test_build_index_empty_corpus(self):
        store = BM25StoreManager(corpus=[])
        store.build_index()
        assert store.retriever is None

    def test_build_index(self, mock_bm25):
        with (
            patch("src.retrieval.keyword_store.bm25s.BM25", return_value=mock_bm25),
            patch("src.retrieval.keyword_store.jieba.cut", side_effect=lambda x: [x]),
            patch("src.retrieval.keyword_store.Stemmer.Stemmer"),
        ):
            store = BM25StoreManager(corpus=["doc1", "doc2"])
        mock_bm25.index.assert_called_once()

    # ── 持久化 ───────────────────────────────────────────

    def test_save_index(self, store_with_corpus, mock_bm25):
        store_with_corpus.save_index()
        mock_bm25.save.assert_called_once_with("bm25_index")

    def test_load_index(self, mock_bm25):
        with (
            patch("src.retrieval.keyword_store.bm25s.BM25.load", return_value=mock_bm25),
            patch("src.retrieval.keyword_store.jieba.cut"),
            patch("src.retrieval.keyword_store.Stemmer.Stemmer"),
        ):
            store = BM25StoreManager(corpus=[])
            store.load_index()
        assert store.retriever is mock_bm25

    # ── 检索 ─────────────────────────────────────────────

    def test_search(self, store_with_corpus):
        results = store_with_corpus.search("美的营收", k=2)
        assert len(results) == 2
        assert results[0]["rank"] == 1
        assert results[0]["score"] == 0.95
        assert results[0]["content"] == "美的2024年营收3721亿元"

    def test_search_no_index(self):
        store = BM25StoreManager(corpus=[])
        results = store.search("test", k=3)
        assert results == []


# ===================================================================
# HybridSearchManager（混合检索 + RRF 融合）
# ===================================================================

class TestHybridSearchManager:
    @pytest.fixture
    def mock_vector_store(self):
        vs = MagicMock()
        vs.search.return_value = [
            Document(page_content="美的2024年营收3721亿元", metadata={"source": "vec"}),
            Document(page_content="美的集团简介", metadata={"source": "vec"}),
        ]
        return vs

    @pytest.fixture
    def mock_keyword_store(self):
        ks = MagicMock()
        ks.search.return_value = [
            {"rank": 1, "score": 0.95, "content": "美的2024年营收3721亿元"},
            {"rank": 2, "score": 0.88, "content": "美的2024年净利润385亿元"},
        ]
        return ks

    @pytest.fixture
    def hybrid(self, mock_vector_store, mock_keyword_store):
        return HybridSearchManager(
            vector_store_manager=mock_vector_store,
            keyword_store_manager=mock_keyword_store,
        )

    # ── RRF 融合 ─────────────────────────────────────────

    def test_rrf_fusion(self, hybrid, mock_vector_store, mock_keyword_store):
        vec = mock_vector_store.search.return_value
        kw = mock_keyword_store.search.return_value
        results = hybrid._rrf_fusion(vec, kw)
        assert len(results) == 3
        assert results[0].page_content == "美的2024年营收3721亿元"
        assert "rrf_score" in results[0].metadata

    def test_rrf_fusion_empty(self, hybrid):
        results = hybrid._rrf_fusion([], [])
        assert results == []

    # ── 混合检索 ─────────────────────────────────────────

    def test_search(self, hybrid, mock_vector_store, mock_keyword_store):
        results = hybrid.search("美的营收", top_k=3)
        assert len(results) <= 3
        assert all(isinstance(d, Document) for d in results)

    def test_search_both_empty(self, hybrid, mock_vector_store, mock_keyword_store):
        mock_vector_store.search.return_value = []
        mock_keyword_store.search.return_value = []
        results = hybrid.search("不存在", top_k=5)
        assert results == []

    def test_search_only_vector(self, hybrid, mock_vector_store, mock_keyword_store):
        mock_vector_store.search.return_value = [
            Document(page_content="唯一结果", metadata={}),
        ]
        mock_keyword_store.search.return_value = []
        results = hybrid.search("test", top_k=5)
        assert len(results) == 1
        assert results[0].page_content == "唯一结果"

    def test_search_only_keyword(self, hybrid, mock_vector_store, mock_keyword_store):
        mock_vector_store.search.return_value = []
        mock_keyword_store.search.return_value = [
            {"rank": 1, "score": 0.9, "content": "唯一关键词结果"},
        ]
        results = hybrid.search("test", top_k=5)
        assert len(results) == 1
        assert results[0].page_content == "唯一关键词结果"

    def test_search_respects_top_k(self, hybrid, mock_vector_store, mock_keyword_store):
        results = hybrid.search("美的", top_k=1)
        assert len(results) == 1


# ===================================================================
# Rerankers（重排序器）
# ===================================================================

class TestBaseReranker:
    def test_base_rerank_raises(self):
        with pytest.raises(NotImplementedError):
            BaseReranker().rerank("query", [])


class TestBM25Reranker:
    @pytest.fixture
    def mock_bm25_store(self):
        store = MagicMock()
        store.search.return_value = [
            {"content": "doc A content here"},
            {"content": "doc B content here"},
        ]
        return store

    @pytest.fixture
    def reranker(self, mock_bm25_store):
        return BM25Reranker(bm25_store=mock_bm25_store)

    def test_rerank_reorders(self, reranker, mock_bm25_store):
        docs = [
            Document(page_content="doc B content here with more details"),
            Document(page_content="doc A content here"),
        ]
        result = reranker.rerank("query", docs)
        assert result[0].page_content == "doc A content here"
        assert result[1].page_content == "doc B content here with more details"

    def test_rerank_returns_original_when_short(self, reranker, mock_bm25_store):
        docs = [Document(page_content="only one")]
        result = reranker.rerank("query", docs)
        assert result is docs

    def test_rerank_no_match(self, reranker, mock_bm25_store):
        mock_bm25_store.search.return_value = [{"content": "unrelated"}]
        docs = [
            Document(page_content="doc A"),
            Document(page_content="doc B"),
            Document(page_content="doc C"),
        ]
        result = reranker.rerank("query", docs)
        assert [d.page_content for d in result] == ["doc A", "doc B", "doc C"]


class TestCrossEncoderReranker:
    def test_rerank_short_list_returns_original(self):
        reranker = CrossEncoderReranker()
        docs = [Document(page_content="only one")]
        result = reranker.rerank("query", docs)
        assert result is docs

    def test_rerank_many_without_model_raises(self):
        reranker = CrossEncoderReranker()
        docs = [
            Document(page_content="a"),
            Document(page_content="b"),
            Document(page_content="c"),
        ]
        with pytest.raises(NotImplementedError):
            reranker.rerank("query", docs)


# ===================================================================
# Post-Filters（后过滤）
# ===================================================================

class TestBaseFilter:
    def test_base_apply_raises(self):
        with pytest.raises(NotImplementedError):
            BaseFilter(order=1).apply([], {})

    def test_order_property(self):
        f = BaseFilter(order=5)
        assert f.order == 5


class TestDeduplicationFilter:
    @pytest.fixture
    def filter(self):
        return DeduplicationFilter()

    def test_removes_duplicates(self, filter):
        docs = [
            Document(page_content="美的营收3721亿"),
            Document(page_content="美的净利润385亿"),
            Document(page_content="美的营收3721亿"),
        ]
        result = filter.apply(docs, {})
        assert len(result) == 2

    def test_preserves_order(self, filter):
        docs = [
            Document(page_content="B"),
            Document(page_content="A"),
            Document(page_content="C"),
        ]
        result = filter.apply(docs, {})
        assert [d.page_content for d in result] == ["B", "A", "C"]

    def test_all_unique(self, filter):
        docs = [Document(page_content=f"doc {i}") for i in range(3)]
        result = filter.apply(docs, {})
        assert len(result) == 3


class TestPermissionFilter:
    @pytest.fixture
    def filter(self):
        return PermissionFilter()

    def test_keeps_doc_when_no_roles_specified(self, filter):
        docs = [Document(page_content="public doc", metadata={})]
        result = filter.apply(docs, {"user_roles": ["viewer"]})
        assert len(result) == 1

    def test_keeps_doc_when_role_matches(self, filter):
        docs = [
            Document(
                page_content="finance doc",
                metadata={"allowed_roles": ["analyst", "admin"]},
            )
        ]
        result = filter.apply(docs, {"user_roles": ["analyst"]})
        assert len(result) == 1

    def test_filters_doc_when_no_role_match(self, filter):
        docs = [
            Document(page_content="secret doc", metadata={"allowed_roles": ["admin"]})
        ]
        result = filter.apply(docs, {"user_roles": ["viewer"]})
        assert len(result) == 0

    def test_empty_user_roles_keeps_all(self, filter):
        docs = [Document(page_content="doc", metadata={"allowed_roles": ["admin"]})]
        result = filter.apply(docs, {"user_roles": []})
        assert len(result) == 1


class TestFreshnessFilter:
    @pytest.fixture
    def filter(self):
        return FreshnessFilter(max_days=30)

    def test_keeps_recent_docs(self, filter):
        docs = [Document(page_content="recent", metadata={"update_time": "2026-05-20"})]
        result = filter.apply(docs, {})
        assert len(result) == 1

    def test_filters_outdated_docs(self, filter):
        docs = [Document(page_content="old", metadata={"update_time": "2024-01-01"})]
        result = filter.apply(docs, {})
        assert len(result) == 0

    def test_keeps_doc_without_time(self, filter):
        docs = [Document(page_content="no time", metadata={})]
        result = filter.apply(docs, {})
        assert len(result) == 1

    def test_keeps_doc_with_invalid_time(self, filter):
        docs = [
            Document(page_content="invalid date", metadata={"update_time": "not-a-date"})
        ]
        result = filter.apply(docs, {})
        assert len(result) == 1


class TestTimeRangeFilter:
    @pytest.fixture
    def filter(self):
        return TimeRangeFilter()

    def test_filters_by_date_range(self, filter):
        docs = [
            Document(page_content="Q1 report", metadata={"date": "2025-01-15"}),
            Document(page_content="Q2 report", metadata={"date": "2025-04-20"}),
            Document(page_content="Q3 report", metadata={"date": "2025-07-10"}),
        ]
        result = filter.apply(docs, {"time_filter": ("2025-03-01", "2025-06-30")})
        assert len(result) == 1
        assert result[0].page_content == "Q2 report"

    def test_no_time_filter_returns_all(self, filter):
        docs = [Document(page_content="doc", metadata={"date": "2025-01-01"})]
        result = filter.apply(docs, {})
        assert len(result) == 1

    def test_fallback_on_empty_result(self, filter):
        docs = [Document(page_content="old doc", metadata={"date": "2024-01-01"})]
        result = filter.apply(docs, {"time_filter": ("2025-01-01", "2025-12-31")})
        assert len(result) == 1


class TestPostFilterChain:
    @pytest.fixture
    def chain(self):
        c = PostFilterChain()
        c.add_filter(DeduplicationFilter())
        c.add_filter(TimeRangeFilter())
        return c

    def test_process_runs_all_filters_in_order(self, chain):
        docs = [
            Document(page_content="Q1 report", metadata={"date": "2025-01-15"}),
            Document(page_content="Q2 report", metadata={"date": "2025-04-20"}),
            Document(page_content="Q1 report", metadata={"date": "2025-01-15"}),
        ]
        result = chain.process(docs, {"time_filter": ("2025-04-01", "2025-06-30")})
        assert len(result) == 1
        assert result[0].page_content == "Q2 report"

    def test_process_no_context(self, chain):
        docs = [Document(page_content="doc")]
        result = chain.process(docs)
        assert len(result) == 1

    def test_empty_docs(self, chain):
        result = chain.process([])
        assert result == []

    def test_filters_sorted_by_order(self):
        chain = PostFilterChain()
        f1 = MagicMock(spec=BaseFilter)
        f1.order = 3
        f2 = MagicMock(spec=BaseFilter)
        f2.order = 1
        f3 = MagicMock(spec=BaseFilter)
        f3.order = 2
        for f in (f1, f2, f3):
            chain.add_filter(f)
        docs = [Document(page_content="test")]
        chain.process(docs, {})
        f2.apply.assert_called_once()
        f3.apply.assert_called_once()
        f1.apply.assert_called_once()
