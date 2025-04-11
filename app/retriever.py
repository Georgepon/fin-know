from app.vectorstore import QdrantVectorStore


def get_relevant_chunks(question: str, vectorstore: QdrantVectorStore):
    return vectorstore.embed_and_search(question, top_k=5)
