def get_relevant_chunks(question: str, vectorstore, top_k: int = 5) -> list:
    """
    Retrieve the top_k most relevant document chunks from the vectorstore given a user question.

    Args:
        question (str): The user's input question.
        vectorstore: An instance of VectorStore containing the FAISS index and metadata.
        top_k (int, optional): Number of most similar chunks to return. Defaults to 5.

    Returns:
        list: A list of metadata dictionaries corresponding to the top_k relevant chunks.
    """
    query = "query: " + question
    query_vector = vectorstore.model.encode([query], convert_to_numpy=True)

    distances, indices = vectorstore.index.search(query_vector, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(vectorstore.metadata_store):
            results.append(vectorstore.metadata_store[idx])

    return results
