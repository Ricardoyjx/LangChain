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
    try:
        parser = DocumentParserFactory.get_parser(file_path)
        result = parser.parse(file_path)
        return result
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return {"content": "", "metadata": {}, "tables": []}
