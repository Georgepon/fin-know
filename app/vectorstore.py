import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("intfloat/multilingual-e5-base")
        self.index_path = "index/faiss.index"
        self.metadata_path = "index/metadata.npy"
        self.dimension = 768
        self.metadata_store = []
        self.model = SentenceTransformer("intfloat/multilingual-e5-base")
        self.index = None

        os.makedirs("index", exist_ok=True)
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            self.metadata_store = np.load(
                self.metadata_path, allow_pickle=True
            ).tolist()
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

    def load_index(self, faiss_path: str, metadata_path: str = "my_index_metadata.pkl"):
        import pickle

        import faiss

        self.index = faiss.read_index(faiss_path)
        with open(metadata_path, "rb") as f:
            self.metadata_store = pickle.load(f)

    def embed_and_store_chunks(self, chunks):
        texts = ["passage: " + chunk["text"] for chunk in chunks]
        vectors = self.model.encode(texts, convert_to_numpy=True)

        self.index.add(vectors)
        self.metadata_store.extend(chunks)

        faiss.write_index(self.index, self.index_path)
        np.save(self.metadata_path, self.metadata_store)
