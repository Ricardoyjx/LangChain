import os
from typing import Any, Dict

try:
    import openpyxl
except ImportError:
    openpyxl = None  # type: ignore

from .base import BaseParser


def _make_unique_columns(columns: list) -> list:
    """对重复列名添加 _N 后缀以确保唯一性。"""
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


def _sheet_to_records(ws) -> list[dict]:
    """将 openpyxl Worksheet 转为记录列表（第一行为列名）。"""
    rows_iter = ws.iter_rows(values_only=True)
    try:
        header = [str(c) if c is not None else "" for c in next(rows_iter)]
    except StopIteration:
        return []

    data_rows = []
    for row in rows_iter:
        # 跳过全空行
        if all(cell is None or (isinstance(cell, str) and cell.strip() == "") for cell in row):
            continue
        data_rows.append([str(c) if c is not None else "" for c in row])

    return _table_to_records(header, data_rows)


class ExcelParser(BaseParser):
    def parse(self, file_path: str) -> Dict[str, Any]:
        if openpyxl is None:
            raise ImportError(
                "openpyxl is required. Install it with: pip install openpyxl"
            )

        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)

        sheet_text_parts = []
        all_tables = []

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]

            # 每个 sheet 生成文本摘要
            records = _sheet_to_records(ws)
            if records:
                all_tables.append({"sheet": sheet_name, "data": records})
                # 用前几行作为文本描述
                lines = [f"[Sheet: {sheet_name}]"]
                for row in records[:5]:
                    line = " | ".join(str(v) for v in row.values())
                    lines.append(line)
                if len(records) > 5:
                    lines.append(f"... ({len(records)} 行)")
                sheet_text_parts.append("\n".join(lines))

        wb.close()

        print(f"Parsing Excel file: {file_path}")
        return {
            "content": "\n\n".join(sheet_text_parts).strip(),
            "metadata": {
                "type": "excel",
                "source": os.path.basename(file_path),
                "sheet_count": len(wb.sheetnames),
                "sheets": wb.sheetnames,
            },
            "tables": all_tables,
        }
