from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from datetime import datetime, timedelta


class BaseFilter:
    """过滤器的基类"""

    def __init__(self, order: int):
        self.order = order

    def apply(self, docs: List[Document], context: Dict[str, Any]) -> List[Document]:
        raise NotImplementedError


class DeduplicationFilter(BaseFilter):
    def __init__(self, order: int = 1):
        super().__init__(order)

    def apply(self, docs: List[Document], context: Dict[str, Any]) -> List[Document]:
        seen_contents = set()
        unique_docs = []
        for doc in docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_docs.append(doc)
        return unique_docs


class PermissionFilter(BaseFilter):
    def __init__(self, order: int = 2):
        super().__init__(order)

    def apply(self, docs: List[Document], context: Dict[str, Any]) -> List[Document]:
        user_roles = context.get("user_roles", [])
        filtered = []
        for doc in docs:
            allowed_roles = doc.metadata.get("allowed_roles", [])
            if not allowed_roles or any(role in allowed_roles for role in user_roles):
                filtered.append(doc)
        return filtered


class FreshnessFilter(BaseFilter):
    """按 update_time 字段过滤过期文档"""

    def __init__(self, order: int = 3, max_days: int = 365):
        super().__init__(order)
        self.max_days = max_days

    def apply(self, docs: List[Document], context: Dict[str, Any]) -> List[Document]:
        cutoff_date = datetime.now() - timedelta(days=self.max_days)
        filtered = []
        for doc in docs:
            time_str = doc.metadata.get("update_time")
            if time_str:
                try:
                    if datetime.strptime(time_str, "%Y-%m-%d") >= cutoff_date:
                        filtered.append(doc)
                except ValueError:
                    filtered.append(doc)
            else:
                filtered.append(doc)
        return filtered


class TimeRangeFilter(BaseFilter):
    """按 date 字段过滤指定时间范围内的文档（配合 QueryPipeline.time_filter 使用）"""

    def __init__(self, order: int = 4):
        super().__init__(order)

    def apply(self, docs: List[Document], context: Dict[str, Any]) -> List[Document]:
        time_range: Optional[Tuple[str, str]] = context.get("time_filter")
        if not time_range:
            return docs

        start_str, end_str = time_range
        filtered = [doc for doc in docs
                     if start_str <= doc.metadata.get("date", "") <= end_str]
        return filtered or docs  # 过滤后为空则回退


class PostFilterChain:
    def __init__(self):
        self.filters: List[BaseFilter] = []

    def add_filter(self, filter_obj: BaseFilter):
        self.filters.append(filter_obj)
        self.filters.sort(key=lambda f: f.order)

    def process(
        self, docs: List[Document], context: Dict[str, Any] = None
    ) -> List[Document]:
        if context is None:
            context = {}
        current_docs = docs
        for filter_obj in self.filters:
            current_docs = filter_obj.apply(current_docs, context)
        return current_docs
