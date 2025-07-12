import sys
import os
from types import ModuleType

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "app"))

vectorstore_stub = ModuleType("vectorstore")
vectorstore_stub.QdrantVectorStore = object
sys.modules.setdefault("vectorstore", vectorstore_stub)

from app.retriever import get_relevant_chunks


class DummyVectorStore:
    def __init__(self, result):
        self.result = result
        self.called_args = None

    def embed_and_search(self, query, top_k=5, filter_doc_ids=None):
        self.called_args = (query, top_k, filter_doc_ids)
        return self.result


def test_get_relevant_chunks_passes_arguments():
    store = DummyVectorStore([{"text": "chunk"}])
    chunks = get_relevant_chunks("question", store, filter_doc_ids=["1"])
    assert chunks == [{"text": "chunk"}]
    assert store.called_args == ("question", 5, ["1"])
