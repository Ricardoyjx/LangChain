"""查看 Milvus 搜索返回的原始数据结构"""
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.retrieval import VectorStore
from langchain_ollama import OllamaEmbeddings

store = VectorStore(
    uri="http://localhost:19530",
    collection_name="default",
    embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
)

# 直接看原始搜索返回
raw = store.client.search(
    collection_name="default",
    data=[store.embedding_function.embed_query("测试")],
    limit=2,
    output_fields=["*"],
)

print("===== 原始返回 =====")
import json
for i, hit in enumerate(raw[0]):
    print(f"\n--- Hit {i+1} ---")
    print(f"  id: {hit.get('id')!r}")
    print(f"  distance: {hit.get('distance')!r}")
    entity = hit.get("entity", {})
    print(f"  entity keys: {list(entity.keys())}")
    print(f"  entity['text']: {entity.get('text', 'MISSING')[:80] if isinstance(entity.get('text'), str) else entity.get('text')!r}")
