# post_filter.py
from typing import List, Dict, Any
from langchain_core.documents import Document
from datetime import datetime, timedelta


# 定义基础过滤器抽象类
class BaseFilter:
    """过滤器的基类，定义统一的执行接口"""

    def __init__(self, order: int):
        self.order = order  # 执行顺序，数字越小越先执行

    def apply(self, docs: List[Document], context: Dict[str, Any]) -> List[Document]:
        """
        执行过滤逻辑
        :param docs: 待过滤的文档列表
        :param context: 上下文信息（如当前用户信息、权限列表等）
        """
        raise NotImplementedError


# 1. 去重过滤器 (Deduplication Filter)
class DeduplicationFilter(BaseFilter):
    def __init__(self, order: int = 1):
        super().__init__(order)

    def apply(self, docs: List[Document], context: Dict[str, Any]) -> List[Document]:
        """根据文档内容去重，保留最先出现的文档"""
        seen_contents = set()
        unique_docs = []
        for doc in docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_docs.append(doc)
        return unique_docs


# 2. 权限过滤器 (Permission Filter)
class PermissionFilter(BaseFilter):
    def __init__(self, order: int = 2):
        super().__init__(order)

    def apply(self, docs: List[Document], context: Dict[str, Any]) -> List[Document]:
        """
        校验用户是否有权限访问该文档
        假设文档的 metadata 中包含 allowed_roles (允许访问的角色列表)
        """
        user_roles = context.get(
            "user_roles", []
        )  # 获取当前用户的角色，如 ['employee']
        filtered_docs = []
        for doc in docs:
            allowed_roles = doc.metadata.get("allowed_roles", [])
            # 如果文档没有设置权限限制，或者用户角色在允许列表中，则放行
            if not allowed_roles or any(role in allowed_roles for role in user_roles):
                filtered_docs.append(doc)
        return filtered_docs


# 3. 时效性过滤器 (Freshness Filter)
class FreshnessFilter(BaseFilter):
    def __init__(self, order: int = 3, max_days: int = 365):
        """
        :param max_days: 只保留最近 N 天内的文档
        """
        super().__init__(order)
        self.max_days = max_days

    def apply(self, docs: List[Document], context: Dict[str, Any]) -> List[Document]:
        """过滤掉过期的文档"""
        filtered_docs = []
        cutoff_date = datetime.now() - timedelta(days=self.max_days)

        for doc in docs:
            # 假设文档 metadata 中包含 update_time (格式如 '2025-01-01')
            update_time_str = doc.metadata.get("update_time")
            if update_time_str:
                try:
                    update_time = datetime.strptime(update_time_str, "%Y-%m-%d")
                    if update_time >= cutoff_date:
                        filtered_docs.append(doc)
                except ValueError:
                    # 日期格式错误，保守起见可以选择过滤掉或保留，这里选择保留
                    filtered_docs.append(doc)
            else:
                # 没有更新时间字段的文档，默认保留
                filtered_docs.append(doc)
        return filtered_docs


# 后过滤链管理器
class PostFilterChain:
    def __init__(self):
        self.filters: List[BaseFilter] = []

    def add_filter(self, filter_obj: BaseFilter):
        """添加过滤器，并按 order 自动排序"""
        self.filters.append(filter_obj)
        self.filters.sort(key=lambda f: f.order)

    def process(
        self, docs: List[Document], context: Dict[str, Any] = None
    ) -> List[Document]:
        """
        依次执行链路上的所有过滤器
        :param docs: 混合检索召回的原始文档
        :param context: 传递用户身份等上下文信息
        """
        if context is None:
            context = {}

        print(f"🚀 开始执行后过滤链，原始文档数量: {len(docs)}")
        current_docs = docs
        for filter_obj in self.filters:
            current_docs = filter_obj.apply(current_docs, context)
            print(
                f"   - 经过 {filter_obj.__class__.__name__} 过滤，剩余文档数量: {len(current_docs)}"
            )
        return current_docs
