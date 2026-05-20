import os
from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path
import pandas as pd

import pdfplumber

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


# 3. 实现具体的解析器（针对异构数据的分别处理）
class PDFParse(BaseParser):
    def parse(self, file_path: str) -> Dict[str, Any]:
        # 这里调用 pdfplumber 等库提取 PDF 内容
        full_text = ""
        all_tables = []

        with pdfplumber.open(file_path) as pdf:
            print(f"该PDF总页数: {len(pdf.pages)}")
            # 提取pdf中文本内容
            for page in pdf.pages:
                text = page.extract_text()
                if text:  # 跳过空白页
                    full_text += text + "\n"

            # 提取pdf中表格内容
            tables = page.extract_tables()
            for table in tables:
                if table:
                    # 将表格转换为 pandas DataFrame 方便后续处理，再转为字典列表存入
                    df = pd.DataFrame(table[1:], columns=table[0])
                    all_tables.append(df.to_dict(orient="records"))

        # 3. 严格按照约定的格式返回字典
        print(f"Parsing PDF file: {file_path}")
        return {
            "content": full_text.strip(),  # 去除首尾空白
            "metadata": {
                "type": "pdf",
                "source": os.path.basename(file_path),
                "total_pages": len(pdf.pages),
            },
            "tables": all_tables,
        }


class WordParser(BaseParser):
    def parse(self, file_path: str) -> Dict[str, Any]:
        # 这里调用 python-docx 提取 Word 内容
        print(f"正在使用 Word 引擎解析: {file_path}")
        return {
            "content": "这是从Word提取的文本...",
            "metadata": {"type": "docx"},
            "tables": [],
        }


class ExcelParser(BaseParser):
    def parse(self, file_path: str) -> Dict[str, Any]:
        # 这里调用 pandas 提取 Excel 内容
        print(f"正在使用 Excel 引擎解析: {file_path}")
        return {
            "content": "这是从Excel转换的文本...",
            "metadata": {"type": "xlsx"},
            "tables": [],
        }


# 4. 解析器工厂（代替 switch-case 的核心分发逻辑）
class DocumentParserFactory:
    # 建立文件后缀与解释器类的映射关系
    _PARSER_MAP = {
        ".pdf": PDFParse,
        ".docx": WordParser,
        ".doc": WordParser,
        ".xlsx": ExcelParser,
        ".xls": ExcelParser,
    }

    @classmethod
    def get_parser(cls, file_path: str) -> BaseParser:
        # 获取文件后缀
        ext = Path(file_path).suffix.lower()
        # 根据后缀返回对应的解析器实例
        parser_class = cls._PARSER_MAP.get(ext)
        if not parser_class:
            raise ValueError(f"Unsupported file type: {ext}")
        return parser_class()


# 5. 统一处理流水线
def process_heterogeneous_data(file_path: str) -> Dict[str, Any]:
    """主函数：处理异构数据文件并返回统一格式的结果"""
    try:
        # 获取对应的解析器
        parser = DocumentParserFactory.get_parser(file_path)
        # 解析文件并返回结果
        result = parser.parse(file_path)
        return result
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return {"content": "", "metadata": {}, "tables": []}


# 模拟运行
if __name__ == "__main__":
    files = [
        "./data/美的2025年报.pdf",
    ]
    for file in files:
        # 无论什么文件，出来的结果都是完全一样的标准字典结构
        result = process_heterogeneous_data(file)
        # print(f"最终标准化结果: {result}\n")
