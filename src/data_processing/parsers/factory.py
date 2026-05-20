import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import pdfplumber

from .base import BaseParser


class PDFParser(BaseParser):
    def parse(self, file_path: str) -> Dict[str, Any]:
        # 这里调用 pdfplumber 等库提取 PDF 内容
        full_text = ""
        all_tables = []

        with pdfplumber.open(file_path) as pdf:
            print(f"该PDF总页数: {len(pdf.pages)}")
            total_pages = len(pdf.pages)
            # 提取pdf中文本内容
            for page in pdf.pages:
                text = page.extract_text()
                if text:  # 跳过空白页
                    full_text += text + "\n"

                # 提取pdf中表格内容（每页独立提取）
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        # 将表格转换为 pandas DataFrame 方便后续处理，再转为字典列表存入
                        df = pd.DataFrame(table[1:], columns=table[0])  # 第一行作为表头                    
                        all_tables.append(df.to_dict(orient="records"))

        # 3. 严格按照约定的格式返回字典
        print(f"Parsing PDF file: {file_path}")
        return {
            "content": full_text.strip(),  # 去除首尾空白
            "metadata": {
                "type": "pdf",
                "source": os.path.basename(file_path),
                "total_pages": total_pages,
            },
            "tables": all_tables,
        }


class WordParser(BaseParser):
    def parse(self, file_path: str) -> Dict[str, Any]:
        """Parse Word document — 尚未实现"""
        raise NotImplementedError(
            "WordParser is not yet implemented. "
            "Please install python-docx and implement document parsing."
        )


class ExcelParser(BaseParser):
    def parse(self, file_path: str) -> Dict[str, Any]:
        """Parse Excel document — 尚未实现"""
        raise NotImplementedError(
            "ExcelParser is not yet implemented. "
            "Please install openpyxl and implement spreadsheet parsing."
        )


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


# 4. 解析器工厂（代替 switch-case 的核心分发逻辑）
class DocumentParserFactory:
    # 建立文件后缀与解释器类的映射关系
    _PARSER_MAP = {
        ".pdf": PDFParser,
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
