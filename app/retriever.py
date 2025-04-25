from typing import Any, Dict, List

from app.vectorstore import QdrantVectorStore


def get_relevant_chunks(
    question: str, vectorstore: QdrantVectorStore
) -> List[Dict[str, Any]]:
    return vectorstore.embed_and_search(question, top_k=5)
