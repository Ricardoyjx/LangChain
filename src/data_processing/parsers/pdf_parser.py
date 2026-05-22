import os
from typing import Any, Dict

import pdfplumber

from .base import BaseParser


def _make_unique_columns(columns: list) -> list:
    """对重复列名添加 _N 后缀以确保唯一性。

    例: ["金额", "金额", "日期"] -> ["金额", "金额_1", "日期"]
    """
    seen: dict[str, int] = {}
    result = []
    for col in columns:
        if col in seen:
            cur = seen[col] + 1
            seen[col] = cur
            result.append(f"{col}_{cur}")
        else:
            seen[col] = 0
            result.append(col)
    return result


def _table_to_records(header: list, rows: list) -> list[dict]:
    """将表格头+行数据转为字典列表，自动处理重复列名。"""
    columns = _make_unique_columns(header)
    return [dict(zip(columns, row)) for row in rows]


class PDFParser(BaseParser):
    def parse(self, file_path: str) -> Dict[str, Any]:
        full_text = ""
        all_tables = []

        with pdfplumber.open(file_path) as pdf:
            print(f"该PDF总页数: {len(pdf.pages)}")
            total_pages = len(pdf.pages)
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

                tables = page.extract_tables()
                for table in tables:
                    if table:
                        all_tables.append(_table_to_records(table[0], table[1:]))

        print(f"Parsing PDF file: {file_path}")
        return {
            "content": full_text.strip(),
            "metadata": {
                "type": "pdf",
                "source": os.path.basename(file_path),
                "total_pages": total_pages,
            },
            "tables": all_tables,
        }
