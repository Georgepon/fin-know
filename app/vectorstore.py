import math
import os
import time
from typing import Any, Dict, List
from uuid import uuid4

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models

load_dotenv()


class QdrantVectorStore:
    """
    Handles vector storage and retrieval using Qdrant Cloud (Synchronous).
    """

    def __init__(self) -> None:
        """
        Initializes the synchronous Qdrant and OpenAI clients.
        Initializes the collection directly.
        """
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = os.getenv("QDRANT_COLLECTION")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        if not self.qdrant_url:
            raise ValueError("QDRANT_URL environment variable not set.")

        # Initialize Sync Qdrant Client
        self.client = QdrantClient(
            url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=30.0
        )

        # Initialize Sync OpenAI Client
        self.openai_client = OpenAI(api_key=self.openai_api_key)

        # Initialize collection synchronously
        self._init_collection()

    def _init_collection(self) -> None:
        """
        (Sync) Checks if the specified collection exists in Qdrant. If not, creates one.
        """
        print("Initializing Qdrant collection (sync)...")
        try:
            collections_response = self.client.get_collections()
            collection_names = [col.name for col in collections_response.collections]

            if self.collection_name not in collection_names:
                print(f"Creating Qdrant collection: {self.collection_name}")
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1536, distance=models.Distance.COSINE
                    ),
                )
                print(f"Collection {self.collection_name} created.")
            else:
                print(f"Collection {self.collection_name} already exists.")
                pass

        except Exception as e:
            print(f"Error initializing Qdrant collection (sync): {e}")
            raise

    def embed_texts_openai(self, texts: List[str]) -> List[List[float]]:
        """
        (Sync) Uses OpenAI's API to generate embeddings.
        """
        if not texts:
            return []
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small", input=texts
            )
            return [d.embedding for d in response.data]
        except Exception as e:
            print(f"Error getting OpenAI embeddings (sync) for {len(texts)} texts: {e}")
            raise

    def upsert(
        self, embeddings: List[List[float]], metadata_list: List[Dict[str, Any]]
    ) -> None:
        """
        (Sync) Stores vectors and metadata. Generates UUID for each point ID.
        """
        if not embeddings:
            print("No embeddings provided to upsert.")
            return

        points = [
            models.PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload=metadata,
            )
            for embedding, metadata in zip(embeddings, metadata_list)
        ]
        try:
            self.client.upsert(
                collection_name=self.collection_name, points=points, wait=True
            )
        except Exception as e:
            print(f"Error upserting {len(points)} points to Qdrant (sync): {e}")
            raise

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        (Sync) Searches the Qdrant collection.
        """
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
            )
            return [hit.payload for hit in results]
        except Exception as e:
            print(f"Error searching Qdrant (sync): {e}")
            raise

    def embed_and_store_chunks(
        self, chunks: List[Dict[str, str]], batch_size: int = 128, progress_bar=None
    ) -> None:
        """
        (Sync) Embeds and stores chunks sequentially in batches.
        Updates an optional Streamlit progress bar if provided.
        """
        if not chunks:
            print("No chunks provided to embed and store.")
            return

        num_chunks = len(chunks)
        num_batches = math.ceil(num_chunks / batch_size)
        print(
            f"Starting SYNC embedding and storage for {num_chunks} chunks in {num_batches} batches (size: {batch_size})..."
        )

        total_processed_chunks = 0
        start_time = time.time()
        for i in range(num_batches):
            current_batch_num = i + 1
            start_index = i * batch_size
            end_index = min(current_batch_num * batch_size, num_chunks)
            batch_chunks_metadata = chunks[start_index:end_index]
            batch_texts = [chunk["text"] for chunk in batch_chunks_metadata]

            try:
                embeddings = self.embed_texts_openai(batch_texts)
                if embeddings:
                    self.upsert(embeddings, batch_chunks_metadata)
                    total_processed_chunks += len(batch_texts)
                else:
                    print(
                        f"Warning: Embedding returned empty for batch {current_batch_num}. Skipping upsert."
                    )

                # Update progress bar if provided
                if progress_bar:
                    progress_percentage = min(1.0, current_batch_num / num_batches)
                    progress_bar.progress(
                        progress_percentage,
                        text=f"Embedding batch {current_batch_num}/{num_batches}",
                    )

            except Exception as e:
                print(f"Error processing batch {current_batch_num}/{num_batches}: {e}")
                # Update progress bar to show error maybe?
                if progress_bar:
                    progress_bar.progress(
                        1.0, text=f"Error on batch {current_batch_num}! Check logs."
                    )
                # Decide if processing should stop on error
                break  # Stop processing further batches on error

        end_time = time.time()
        print("-" * 30)
        print(f"SYNC processing complete in {end_time - start_time:.2f} seconds.")
        print(
            f"Successfully processed {total_processed_chunks}/{num_chunks} chunks across {num_batches} batches."
        )
        print("-" * 30)

        # Ensure progress bar reaches 100% if it exists and no error occurred mid-way
        if progress_bar and total_processed_chunks == num_chunks:
            progress_bar.progress(1.0, text="Embedding complete!")

    def embed_and_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        (Sync) Embeds a query and searches.
        """
        query_vector = self.embed_texts_openai([query])
        if not query_vector:
            print("Warning: Failed to embed query.")
            return []
        return self.search(query_vector[0], top_k=top_k)
