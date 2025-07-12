import json
import sys
import os
from types import ModuleType
from pathlib import Path

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "app"))

sys.modules.setdefault("streamlit", ModuleType("streamlit"))

vectorstore_stub = ModuleType("vectorstore")
vectorstore_stub.QdrantVectorStore = object
sys.modules.setdefault("vectorstore", vectorstore_stub)

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

from app.pages import page_add_documents as pages


def test_cache_roundtrip(tmp_path):
    cache_file = tmp_path / "cache.json"
    # monkeypatch the module-level CACHE_FILE
    pages.CACHE_FILE = str(cache_file)

    data = {"abc": "123"}
    pages.save_cache(data)
    assert cache_file.exists()
    loaded = pages.load_cache()
    assert loaded == data

    # corrupt file -> load_cache returns empty dict
    cache_file.write_text("not json")
    assert pages.load_cache() == {}
