from typing import Any, Dict, List, Optional

from vectorstore import QdrantVectorStore


def get_relevant_chunks(
    question: str,
    vectorstore: QdrantVectorStore,
    filter_doc_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    return vectorstore.embed_and_search(
        question, top_k=5, filter_doc_ids=filter_doc_ids
    )
