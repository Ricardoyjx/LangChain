from typing import Any, Dict

from .base import BaseParser


class WordParser(BaseParser):
    def parse(self, file_path: str) -> Dict[str, Any]:
        raise NotImplementedError(
            "WordParser is not yet implemented. "
            "Please install python-docx and implement document parsing."
        )
