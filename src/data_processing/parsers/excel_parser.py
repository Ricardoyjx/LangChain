from typing import Any, Dict

from .base import BaseParser


class ExcelParser(BaseParser):
    def parse(self, file_path: str) -> Dict[str, Any]:
        raise NotImplementedError(
            "ExcelParser is not yet implemented. "
            "Please install openpyxl and implement spreadsheet parsing."
        )
