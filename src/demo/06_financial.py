import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import ChatOllama
from langchain_core.callbacks import CallbackManagerForRetrieverRun


# 文档加载和处理
def load_financial_documents(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


def extract_financial_info(text, schema=None, context=None, question=None):
    """提取金融信息或构建RAG提示词"""
    if schema:
        # 信息提取模式
        prompt = f"""
请从以下文本中提取金融信息，按照指定的schema格式输出JSON：
Schema:{json.dumps(schema,ensure_ascii=False,indent=2)}

文本内容：
{text}

请确保输出严格的JSON格式，如果信息在原文中未提及，请标记为"原文未提及"。
"""
        return prompt
    else:
        # RAG问答模式
        prompt = f"""你是一个专业的金融分析助手。请根据以下知识库信息回答用户问题。
知识库信息：
{context}

用户问题：
{question}

请基于知识库内容回答，如果知识库中没有相关信息，请说明无法回答。"""
        return prompt


class FinancialRAG:
    def __init__(self):
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.vectorstore = None
        self.bm25_retriever = None

    def build_knowledge_base(self, documents):
        # 语义切分
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        texts = text_splitter.split_documents(documents)

        # 构建向量库
        self.vectorstore = FAISS.from_documents(texts, self.embedding)
        # 构建BM25检索器
        self.bm25_retriever = BM25Retriever.from_documents(texts)

    def hybrid_search(self, query, k=5):
        """混合检索：结合向量相似度和BM25关键词匹配"""
        # 向量检索 - 返回一半结果
        vector_results = self.vectorstore.similarity_search(query, k=k // 2 + 1)

        rm = CallbackManagerForRetrieverRun()
        # BM25检索
        bm25_results = self.bm25_retriever._get_relevant_documents(
            query, run_manager=None
        )

        # RRF (Reciprocal Rank Fusion) 融合结果
        # 计算每个文档的综合得分
        scores = {}
        for i, doc in enumerate(vector_results):
            scores[id(doc)] = scores.get(id(doc), 0) + 1 / (i + 60)

        for i, doc in enumerate(bm25_results):
            scores[id(doc)] = scores.get(id(doc), 0) + 1 / (i + 60)

        # 合并所有结果
        all_docs = vector_results + bm25_results

        # 按得分排序
        all_docs.sort(key=lambda x: scores.get(id(x), 0), reverse=True)

        # 去重（保留第一个出现的文档）
        seen = set()
        unique_docs = []
        for doc in all_docs:
            doc_id = id(doc)
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)

        return unique_docs[:k]


llm = ChatOllama(model="qwen3.5:9b", temperature=0.3)
print("金融RAG系统初始化完成！")

path = "data/美的2025年报.pdf"

doc = load_financial_documents(path)
print(f"加载文档完成，共 {len(doc)} 个页面")

f_rag = FinancialRAG()
print("构建金融知识库...")

f_rag.build_knowledge_base(doc)
print("金融知识库构建完成！")


# retriever = vectorstore.as_retriever(
#     search_type="similarity",  # 相似度检索
#     search_kwargs={"k": 2},  # 返回 top 2 相关文档
# )

# 使用混合检索
query = "2025年美的的营业收入是多少？"
retrieved_docs = f_rag.hybrid_search(query, k=5)
print(f"\n查询：{query}")
print(f"检索到 {len(retrieved_docs)} 个相关文档：")
for i, doc in enumerate(retrieved_docs):
    print(f"  [{i+1}] {doc.page_content}")


# 使用 extract_financial_info 构建 RAG 提示词
rag_prompt = ChatPromptTemplate.from_template(
    """你是一个专业的金融分析助手。请根据以下知识库信息回答用户问题。
知识库信息：
{context}

用户问题：
{question}

请基于知识库内容回答，如果知识库中没有相关信息，请说明无法回答。"""
)

# 更新 rag_chain 使用混合检索
rag_chain = (
    {"context": lambda x: f_rag.hybrid_search(x, k=5), "question": lambda x: x}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# 7. 测试 RAG 应用
print("\n" + "=" * 50)
print("RAG 应用测试")
print("=" * 50)

test_questions = [
    "2025年美的的营业收入是多少？",
    "2025年美的的净利润是多少？",
    "2025年美的的主要业务是什么？",
]

for question in test_questions:
    print(f"\n用户：{question}")
    response = rag_chain.invoke(question)
    print(f"客服：{response}")
