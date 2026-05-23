import bm25s
import Stemmer
import jieba


class BM25StoreManager:
    def __init__(self, corpus: list[str] = None, index_path: str = "bm25_index"):
        """
        初始化 BM25 关键词检索管理器
        :param corpus: 初始文档列表（字符串列表）
        :param index_path: 索引保存的本地路径
        """
        self.index_path = index_path
        self.corpus = corpus if corpus else []
        self.retriever = None

        # 初始化中文分词器和英文词干提取器
        self.stemmer = Stemmer.Stemmer("english")

        if self.corpus:
            self.build_index()


def _tokenize(self, texts):
    """
    统一的分词处理（支持中英文混合）
    """
    # 使用 jieba 进行中文分词，并用 Stemmer 处理英文词干
    tokenized = []
    for text in texts:
        # 简单的分词逻辑：jieba 切分后，对纯英文单词做词干提取
        words = list(jieba.cut(text))
        processed_words = [
            self.stemmer.stemWord(w) if w.isalpha() else w for w in words
        ]
        tokenized.append(processed_words)
    return tokenized


def build_index(self):
    """构建 BM25 索引"""
    if not self.corpus:
        print("语料库为空，无法构建索引。")
        return

    tokenized_corpus = self._tokenize(self.corpus)
    self.retriever = bm25s.BM25(corpus=self.corpus)
    self.retriever.index(tokenized_corpus)
    print(f"成功为 {len(self.corpus)} 条文档构建 BM25 索引。")


def save_index(self):
    """保存索引到本地"""
    if self.retriever:
        self.retriever.save(self.index_path)
        print(f"索引已保存至 {self.index_path}")


def load_index(self):
    """从本地加载索引"""
    self.retriever = bm25s.BM25.load(self.index_path, load_corpus=True)
    self.corpus = self.retriever.corpus
    print(f"成功从 {self.index_path} 加载 BM25 索引。")


def search(self, query: str, k: int = 3):
    """
    查（Read）：根据关键词检索文档
    :param query: 用户的查询文本
    :param k: 返回最相关的前 k 个结果
    """
    if not self.retriever:
        print("索引未初始化，请先构建或加载索引。")
        return []

    tokenized_query = self._tokenize([query])
    results, scores = self.retriever.retrieve(tokenized_query, k=k)

    # 整理返回结果
    search_results = []
    for i, (doc, score) in enumerate(zip(results[0], scores[0])):
        search_results.append({"rank": i + 1, "score": float(score), "content": doc})
    return search_results
