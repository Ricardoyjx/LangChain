"""测试数据解析器 —— 所有外部依赖均使用 mock，无需真实文件。"""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.data_processing.parsers import (
    BaseParser,
    PDFParser,
    WordParser,
    ExcelParser,
    DocumentParserFactory,
    process_heterogeneous_data,
)


# ── 基类测试 ─────────────────────────────────────────────

class TestBaseParser:
    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseParser()

    def test_subclass_must_implement_parse(self):
        class IncompleteParser(BaseParser):
            pass
        with pytest.raises(TypeError):
            IncompleteParser()


# ── PDF 解析器 ──────────────────────────────────────────

class TestPDFParser:
    @pytest.fixture
    def mock_page(self):
        page = MagicMock()
        page.extract_text.return_value = "Hello PDF\n这是第一页内容。"
        page.extract_tables.return_value = []
        return page

    @pytest.fixture
    def mock_pdf(self, mock_page):
        pdf = MagicMock()
        pdf.pages = [mock_page, mock_page]
        pdf.__enter__.return_value = pdf
        return pdf

    def test_parse_returns_expected_structure(self, mock_pdf):
        with patch("pdfplumber.open", return_value=mock_pdf):
            result = PDFParser().parse("/fake/sample.pdf")
        assert isinstance(result, dict)
        assert "content" in result
        assert "metadata" in result
        assert "tables" in result

    def test_parse_content(self, mock_pdf):
        with patch("pdfplumber.open", return_value=mock_pdf):
            result = PDFParser().parse("/fake/sample.pdf")
        assert "Hello PDF" in result["content"]
        assert "第一页内容" in result["content"]

    def test_parse_metadata(self, mock_pdf):
        with patch("pdfplumber.open", return_value=mock_pdf):
            result = PDFParser().parse("/fake/sample.pdf")
        meta = result["metadata"]
        assert meta["type"] == "pdf"
        assert meta["source"] == "sample.pdf"
        assert meta["total_pages"] == 2

    def test_parse_tables_with_data(self):
        page = MagicMock()
        page.extract_text.return_value = ""
        page.extract_tables.return_value = [
            [["姓名", "年龄"], ["张三", "28"], ["李四", "35"]]
        ]
        pdf = MagicMock()
        pdf.pages = [page]
        pdf.__enter__.return_value = pdf

        with patch("pdfplumber.open", return_value=pdf):
            result = PDFParser().parse("/fake/sample.pdf")

        assert len(result["tables"]) == 1
        assert result["tables"][0] == [
            {"姓名": "张三", "年龄": "28"},
            {"姓名": "李四", "年龄": "35"},
        ]

    def test_parse_file_not_found(self):
        with patch("pdfplumber.open", side_effect=FileNotFoundError("No such file")):
            with pytest.raises(FileNotFoundError):
                PDFParser().parse("/fake/missing.pdf")


# ── Word 解析器 ─────────────────────────────────────────

class TestWordParser:
    @pytest.fixture
    def mock_document(self):
        doc = MagicMock()

        p1 = MagicMock()
        p1.text = "测试文档标题"
        p2 = MagicMock()
        p2.text = "第二段内容，包含中文文本。"
        type(doc).paragraphs = PropertyMock(return_value=[p1, p2])

        def make_row(*values):
            row = MagicMock()
            row.cells = [MagicMock(text=v) for v in values]
            return row

        table = MagicMock()
        table.rows = [
            make_row("姓名", "年龄", "城市"),
            make_row("张三", "28", "北京"),
            make_row("李四", "35", "上海"),
        ]
        doc.tables = [table]
        return doc

    def test_parse_returns_expected_structure(self, mock_document):
        with patch("src.data_processing.parsers.word_parser.Document", return_value=mock_document):
            result = WordParser().parse("/fake/sample.docx")
        assert isinstance(result, dict)
        assert "content" in result
        assert "metadata" in result
        assert "tables" in result

    def test_parse_content(self, mock_document):
        with patch("src.data_processing.parsers.word_parser.Document", return_value=mock_document):
            result = WordParser().parse("/fake/sample.docx")
        assert "测试文档标题" in result["content"]
        assert "第二段内容" in result["content"]

    def test_parse_metadata(self, mock_document):
        with patch("src.data_processing.parsers.word_parser.Document", return_value=mock_document):
            result = WordParser().parse("/fake/sample.docx")
        meta = result["metadata"]
        assert meta["type"] == "docx"
        assert meta["source"] == "sample.docx"
        assert meta["paragraph_count"] >= 2
        assert meta["table_count"] >= 1

    def test_parse_tables(self, mock_document):
        with patch("src.data_processing.parsers.word_parser.Document", return_value=mock_document):
            result = WordParser().parse("/fake/sample.docx")
        tables = result["tables"]
        assert len(tables) == 1
        assert tables[0][0] == {"姓名": "张三", "年龄": "28", "城市": "北京"}
        assert tables[0][1] == {"姓名": "李四", "年龄": "35", "城市": "上海"}

    def test_parse_file_not_found(self):
        with patch(
            "src.data_processing.parsers.word_parser.Document",
            side_effect=FileNotFoundError("No such file"),
        ):
            with pytest.raises(FileNotFoundError):
                WordParser().parse("/fake/missing.docx")


# ── Excel 解析器 ────────────────────────────────────────

class TestExcelParser:
    @pytest.fixture
    def mock_workbook(self):
        wb = MagicMock()
        wb.sheetnames = ["销售数据", "汇总"]

        ws1 = MagicMock()
        ws1.iter_rows.return_value = iter([
            ("产品", "数量", "金额"),
            ("A产品", 100, 1500),
            ("B产品", 200, 3200),
            ("C产品", 150, 2250),
        ])

        ws2 = MagicMock()
        ws2.iter_rows.return_value = iter([
            ("月份", "收入"),
            ("一月", 10000),
            ("二月", 12000),
        ])

        def getitem(name):
            mapping = {"销售数据": ws1, "汇总": ws2}
            return mapping[name]

        wb.__getitem__.side_effect = getitem
        return wb

    def test_parse_returns_expected_structure(self, mock_workbook):
        with patch("openpyxl.load_workbook", return_value=mock_workbook):
            result = ExcelParser().parse("/fake/sample.xlsx")
        assert isinstance(result, dict)
        assert "content" in result
        assert "metadata" in result
        assert "tables" in result

    def test_parse_metadata(self, mock_workbook):
        with patch("openpyxl.load_workbook", return_value=mock_workbook):
            result = ExcelParser().parse("/fake/sample.xlsx")
        meta = result["metadata"]
        assert meta["type"] == "excel"
        assert meta["source"] == "sample.xlsx"
        assert meta["sheet_count"] == 2
        assert "销售数据" in meta["sheets"]
        assert "汇总" in meta["sheets"]

    def test_parse_tables(self, mock_workbook):
        with patch("openpyxl.load_workbook", return_value=mock_workbook):
            result = ExcelParser().parse("/fake/sample.xlsx")
        tables = result["tables"]
        assert len(tables) == 2

        sheet1 = tables[0]
        assert sheet1["sheet"] == "销售数据"
        assert len(sheet1["data"]) == 3
        assert sheet1["data"][0] == {"产品": "A产品", "数量": "100", "金额": "1500"}
        assert sheet1["data"][1] == {"产品": "B产品", "数量": "200", "金额": "3200"}

        sheet2 = tables[1]
        assert sheet2["sheet"] == "汇总"
        assert sheet2["data"][0] == {"月份": "一月", "收入": "10000"}

    def test_parse_content_contains_sheet_names(self, mock_workbook):
        with patch("openpyxl.load_workbook", return_value=mock_workbook):
            result = ExcelParser().parse("/fake/sample.xlsx")
        assert "销售数据" in result["content"]
        assert "汇总" in result["content"]

    def test_parse_file_not_found(self):
        with patch("openpyxl.load_workbook", side_effect=FileNotFoundError("No such file")):
            with pytest.raises(FileNotFoundError):
                ExcelParser().parse("/fake/missing.xlsx")


# ── 工厂测试 ────────────────────────────────────────────

class TestDocumentParserFactory:
    @pytest.mark.parametrize("ext,expected_cls", [
        (".pdf", PDFParser),
        (".docx", WordParser),
        (".doc", WordParser),
        (".xlsx", ExcelParser),
        (".xls", ExcelParser),
    ])
    def test_get_parser_by_extension(self, ext, expected_cls):
        parser = DocumentParserFactory.get_parser(f"/tmp/sample{ext}")
        assert isinstance(parser, expected_cls)

    def test_get_parser_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            DocumentParserFactory.get_parser("/tmp/sample.unknown")


# ── 端到端测试（整合 mock）──────────────────────────────

class TestProcessHeterogeneousData:
    def test_process_pdf(self):
        page = MagicMock()
        page.extract_text.return_value = "Mocked PDF content"
        page.extract_tables.return_value = []

        pdf = MagicMock()
        pdf.pages = [page]
        pdf.__enter__.return_value = pdf

        with patch("pdfplumber.open", return_value=pdf):
            result = process_heterogeneous_data("/fake/doc.pdf")
        assert result["content"] == "Mocked PDF content"

    def test_process_unsupported_returns_empty(self):
        result = process_heterogeneous_data("/tmp/file.unknown")
        assert result == {"content": "", "metadata": {}, "tables": []}

    def test_process_error_returns_empty(self):
        with patch("pdfplumber.open", side_effect=RuntimeError("Unexpected error")):
            result = process_heterogeneous_data("/fake/doc.pdf")
        assert result == {"content": "", "metadata": {}, "tables": []}
