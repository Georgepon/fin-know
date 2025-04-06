import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the multilingual embedding model
model = SentenceTransformer("intfloat/multilingual-e5-base")

# Define where to store the index and metadata
index_path = "index/faiss.index"
metadata_path = "index/metadata.npy"

# Ensure the index directory exists
os.makedirs("index", exist_ok=True)

# In-memory store for metadata
metadata_store = []

# Load existing FAISS index and metadata, or create a new one
d = 768  # embedding dimension for multilingual-e5-base
if os.path.exists(index_path):
    index = faiss.read_index(index_path)
    metadata_store = np.load(metadata_path, allow_pickle=True).tolist()
else:
    index = faiss.IndexFlatL2(d)


# Embed chunks and store them in the index
def embed_and_store_chunks(chunks):
    texts = ["passage: " + chunk["text"] for chunk in chunks]
    vectors = model.encode(texts, convert_to_numpy=True)

    index.add(vectors)
    metadata_store.extend(chunks)

    faiss.write_index(index, index_path)
    np.save(metadata_path, metadata_store)
