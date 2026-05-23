"""调试脚本：捕获生成时的完整异常堆栈"""
import traceback
import sys
sys.path.insert(0, ".")

# 加载环境
from dotenv import load_dotenv
load_dotenv()

from src.pipeline import RAGPipeline, setup_logging
import logging

setup_logging(logging.DEBUG)
pipeline = RAGPipeline(top_k=5)

# 加载已有索引
import os
from pathlib import Path
INDEX_DIR = Path("data/vector_db")

if INDEX_DIR.exists():
    pipeline._ingest.load_index(str(INDEX_DIR))

# 装配后处理链
from src.retrieval import PostFilterChain, DeduplicationFilter, TimeRangeFilter, BM25Reranker
from src.pipeline import QueryPipeline

chain = PostFilterChain()
chain.add_filter(DeduplicationFilter())
chain.add_filter(TimeRangeFilter())
reranker = BM25Reranker(pipeline._ingest.bm25_store) if pipeline._ingest.bm25_store else None

pipeline._query = QueryPipeline(
    hybrid_search=pipeline._ingest.hybrid_search,
    llm=pipeline._ingest.llm,
    top_k=5,
    post_filter_chain=chain,
    reranker=reranker,
)

# 带 chat_history 模拟第二次交互
chat_history = [
    {"role": "user", "content": "去年的营收是多少？"},
    {"role": "assistant", "content": "根据年报数据，美的2024年实现营业总收入4091亿元。"},
]

query = "那净利润呢？"

try:
    for chunk in pipeline.stream(query, chat_history=chat_history):
        print(chunk, end="", flush=True)
    print()
except Exception:
    traceback.print_exc()
