from pathlib import Path
from typing import Dict, Any

from .base import BaseParser
from .pdf_parser import PDFParser
from .word_parser import WordParser
from .excel_parser import ExcelParser


class DocumentParserFactory:
    _PARSER_MAP = {
        ".pdf": PDFParser,
        ".docx": WordParser,
        ".doc": WordParser,
        ".xlsx": ExcelParser,
        ".xls": ExcelParser,
    }

    @classmethod
    def get_parser(cls, file_path: str) -> BaseParser:
        ext = Path(file_path).suffix.lower()
        parser_class = cls._PARSER_MAP.get(ext)
        if not parser_class:
            raise ValueError(f"Unsupported file type: {ext}")
        return parser_class()


def process_heterogeneous_data(file_path: str) -> Dict[str, Any]:
    parser_name = "unknown"
    try:
        parser = DocumentParserFactory.get_parser(file_path)
        parser_name = type(parser).__name__
        result = parser.parse(file_path)
        if not result.get("content", "") and not result.get("tables", []):
            print(f"[警告] 解析器 {parser_name} 未从 {file_path} 提取到任何内容")
        return result
    except FileNotFoundError:
        print(f"[错误] 文件不存在: {file_path}")
        return {"content": "", "metadata": {}, "tables": []}
    except PermissionError:
        print(f"[错误] 无权限读取: {file_path}")
        return {"content": "", "metadata": {}, "tables": []}
    except Exception as e:
        print(f"[错误] 解析文件失败 [{type(e).__name__}]: {file_path}")
        print(f"  解析器: {parser_name}")
        print(f"  详情: {e}")
        return {"content": "", "metadata": {}, "tables": []}
