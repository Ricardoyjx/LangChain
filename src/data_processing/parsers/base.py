from abc import ABC, abstractmethod
from typing import Dict, Any

# 1. 定义统一的输出格式（标准文档格式）
# 无论什么文件，最后都解析成这个样子的字典
STANDARD_SCHEMA = {
    "content": "",  # 核心文本内容
    "metadata": {},  # 元数据（来源、页码、类型等）
    "tables": [],  # 提取到的表格数据（如果有）
}


# 2. 定义解析器的抽象基类（规范所有解析器的行为）
class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> Dict[str, Any]:
        """解析文件并返回统一的标准格式"""
        pass
