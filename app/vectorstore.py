import os
from typing import Any, Dict, List
from uuid import uuid4

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

load_dotenv()


class QdrantVectorStore:
    """
    Handles vector storage and retrieval using Qdrant Cloud.

    This class initializes a connection to a Qdrant collection, allows you to store embeddings with metadata,
    and search for the most relevant vectors given a query.
    """

    def __init__(self) -> None:
        """
        Initializes the Qdrant client using environment variables and creates the collection if it doesn't exist.
        Environment Variables Required:
            - QDRANT_URL: URL of the Qdrant Cloud instance
            - QDRANT_API_KEY: API key for authentication
            - QDRANT_COLLECTION: Name of the collection to use
        """
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = os.getenv("QDRANT_COLLECTION")

        self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)

        self._init_collection()

    def _init_collection(self) -> None:
        """
        Checks if the specified collection exists in Qdrant. If not, it creates one with cosine distance
        and 384-dimensional vectors.
        """
        if self.collection_name not in self.client.get_collections().collections:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

    def embed_texts_groq(self, texts: List[str]) -> List[List[float]]:
        """
        Sends a list of texts to Groq's hosted embedding API and returns their vector representations.

        Args:
            texts (List[str]): List of strings to embed.

        Returns:
            List[List[float]]: List of embeddings corresponding to each input string.
        """
        response = requests.post(
            "https://api.groq.com/openai/v1/embeddings",
            headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
            json={"model": "nomic-ai/nomic-embed-text-v1", "input": texts},
        )
        response.raise_for_status()
        return [d["embedding"] for d in response.json()["data"]]

    def upsert(
        self, embeddings: List[List[float]], metadata_list: List[Dict[str, Any]]
    ) -> None:
        """
        Stores or updates vectors and their associated metadata in the Qdrant collection.

        Args:
            embeddings (List[List[float]]): List of vector embeddings to store.
            metadata_list (List[Dict[str, Any]]): List of metadata dictionaries corresponding to each embedding.
        """
        points = [
            PointStruct(id=str(uuid4()), vector=embedding, payload=metadata)
            for embedding, metadata in zip(embeddings, metadata_list)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the Qdrant collection for the most similar vectors to a given query vector.

        Args:
            query_vector (List[float]): The query embedding vector.
            top_k (int, optional): The number of top matches to return. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: List of metadata payloads from the top-k most similar vectors.
        """
        results = self.client.search(
            collection_name=self.collection_name, query_vector=query_vector, limit=top_k
        )
        return [hit.payload for hit in results]

    def embed_and_store_chunks(self, chunks: List[Dict[str, str]]) -> None:
        """
        Embeds and stores a list of chunk dictionaries into Qdrant.

        Args:
            chunks (List[Dict[str, str]]): List of chunks where each chunk is a dict with keys like 'text' and 'chunk_id'.
        """
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embed_texts_groq(texts)
        self.upsert(embeddings, chunks)

    def embed_and_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Embeds a text query and retrieves the top-k most relevant chunks from the vector database.

        Args:
            query (str): The question or search input.
            top_k (int): Number of top chunks to return.

        Returns:
            List[Dict[str, Any]]: Metadata of the top matched chunks.
        """
        query_vector = self.embed_texts_groq([query])[0]
        return self.search(query_vector, top_k=top_k)
