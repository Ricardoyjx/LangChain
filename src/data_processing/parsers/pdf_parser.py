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
        total_pages = 0

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"PDF 文件不存在: {file_path}")

        try:
            pdf = pdfplumber.open(file_path)
        except Exception as e:
            raise ValueError(
                f"无法打开 PDF 文件 (可能已损坏或不是有效 PDF): {file_path}\n"
                f"  详情: {e}"
            ) from e

        with pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"

                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            all_tables.append(_table_to_records(table[0], table[1:]))
                except Exception as e:
                    print(f"  [警告] 第 {page_num}/{total_pages} 页解析异常: {e}")
                    continue

        print(f"  PDF 解析完成: {os.path.basename(file_path)} ({total_pages} 页, "
              f"{len(all_tables)} 个表格)")
        return {
            "content": full_text.strip(),
            "metadata": {
                "type": "pdf",
                "source": os.path.basename(file_path),
                "total_pages": total_pages,
            },
            "tables": all_tables,
        }
