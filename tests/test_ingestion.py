from pathlib import Path
import sys
import os
from types import ModuleType

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "app"))

# Stub external dependencies
sys.modules.setdefault("pymupdf", ModuleType("pymupdf"))

langchain_stub = ModuleType("langchain")
text_splitter_stub = ModuleType("langchain.text_splitter")

class DummySplitter:
    def __init__(self, *a, **kw):
        pass

    def split_text(self, text):
        return [text]

text_splitter_stub.RecursiveCharacterTextSplitter = DummySplitter
langchain_stub.text_splitter = text_splitter_stub
sys.modules.setdefault("langchain", langchain_stub)
sys.modules.setdefault("langchain.text_splitter", text_splitter_stub)

from app import ingestion


class DummyPage:
    def __init__(self, text):
        self._text = text

    def get_text(self, kind):
        return self._text


class DummyPdf:
    def __init__(self, texts):
        self.pages = [DummyPage(t) for t in texts]

    def __len__(self):
        return len(self.pages)

    def load_page(self, index):
        return self.pages[index]


def test_process_document_with_mocked_pymupdf(tmp_path, monkeypatch):
    dummy_pdf = DummyPdf(["page one", "page two"])

    def fake_open(*args, **kwargs):
        return dummy_pdf

    monkeypatch.setattr(ingestion, "pymupdf", type("m", (), {"open": staticmethod(fake_open)}))

    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.write_bytes(b"pdfcontent")

    result = ingestion.process_document(pdf_path)
    assert result["num_chunks"] > 0
    assert result["doc_id"]
    assert result["file_hash"]
    assert len(result["chunks"]) == result["num_chunks"]
    assert all(chunk["doc_id"] == result["doc_id"] for chunk in result["chunks"])
    assert all(chunk["filename"] == pdf_path.name for chunk in result["chunks"])
