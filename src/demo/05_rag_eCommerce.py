import os
from typing import List

# LangChain 核心组件
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# RAG 核心组件
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import login
from dotenv import load_dotenv

# 1. 从环境变量中获取 Token
load_dotenv()
token = os.getenv("HF_TOKEN")

# 2. 打印一下，确认代码里拿到了没有（调试用）
print(f"当前读取到的 Token: {token}")

# 3. 如果拿到了，就执行登录
if token:
    login(token=token)
else:
    print("警告：未找到 HF_TOKEN 环境变量！")


# 1. 准备文档数据（模拟电商客服知识库）
documents = [
    Document(
        page_content="商品支持7天无理由退货，商品需保持完好，包装齐全。特殊商品如内衣、食品不支持无理由退货。",
        metadata={"category": "退货政策"},
    ),
    Document(
        page_content="退款将在收到退货并验货后3-5个工作日内处理，原路退回至支付账户。",
        metadata={"category": "退款流程"},
    ),
    Document(
        page_content="运费问题：7天无理由退货，运费由买家承担；商品质量问题，运费由卖家承担。",
        metadata={"category": "运费规则"},
    ),
    Document(
        page_content="商品质量问题请在签收后48小时内反馈，需要提供照片证据。",
        metadata={"category": "质量投诉"},
    ),
    Document(
        page_content="物流查询请提供订单号，我们将在1个工作日内回复物流信息。",
        metadata={"category": "物流查询"},
    ),
]

# 2. 文档分割（将长文档切分成小块）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # 每块大小
    chunk_overlap=10,  # 重叠部分
    length_function=len,
)

splits = text_splitter.split_documents(documents)
print(f"文档分割后：{len(splits)} 个片段")

# 3. 创建向量存储（使用 Chroma）
# 使用更适合中文的嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db",  # 持久化保存
)

print("向量存储创建完成！")

# 4. 创建检索器（从向量库中检索相关文档）
retriever = vectorstore.as_retriever(
    search_type="similarity",  # 相似度检索
    search_kwargs={"k": 2},  # 返回 top 2 相关文档
)

# 测试检索
query = "退货运费谁承担"
retrieved_docs = retriever.invoke(query)
print(f"\n查询：{query}")
print(f"检索到 {len(retrieved_docs)} 个相关文档：")
for i, doc in enumerate(retrieved_docs):
    print(f"  [{i+1}] {doc.page_content}")

# 5. 创建 RAG 提示词模板
rag_prompt = ChatPromptTemplate.from_template(
    """你是一个专业的电商客服助手。请根据以下知识库信息回答用户问题。

知识库信息：
{context}

用户问题：
{question}

请基于知识库内容回答，如果知识库中没有相关信息，请说明无法回答。"""
)

# 6. 搭建 RAG 链
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3.5:9b", temperature=0.3)

# RAG 链：检索 -> 格式化 -> 模型 -> 输出
rag_chain = (
    {"context": retriever, "question": lambda x: x}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# 7. 测试 RAG 应用
print("\n" + "=" * 50)
print("RAG 应用测试")
print("=" * 50)

test_questions = [
    "退货的运费是谁承担？",
    "商品有质量问题怎么办？",
    "多久能收到退款？",
]

for question in test_questions:
    print(f"\n用户：{question}")
    response = rag_chain.invoke(question)
    print(f"客服：{response}")
