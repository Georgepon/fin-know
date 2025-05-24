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

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_doc_ids: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        (Sync) Searches the Qdrant collection.
        Optionally filters by a list of document IDs.
        """
        search_filter = None
        if filter_doc_ids:
            print(f"Applying search filter for {len(filter_doc_ids)} doc IDs.")
            # Create a filter that matches if doc_id is in the provided list
            search_filter = models.Filter(
                should=[
                    models.FieldCondition(
                        key="doc_id", match=models.MatchValue(value=doc_id)
                    )
                    for doc_id in filter_doc_ids
                ]
            )

        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,  # Pass the filter here
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

        if progress_bar and total_processed_chunks == num_chunks:
            progress_bar.progress(1.0, text="Embedding complete!")

    def embed_and_search(
        self, query: str, top_k: int = 10, filter_doc_ids: List[str] | None = None
    ) -> List[Dict[str, Any]]:
        """
        (Sync) Embeds a query and searches, optionally filtering by document IDs.
        """
        query_vector = self.embed_texts_openai([query])
        if not query_vector:
            print("Warning: Failed to embed query.")
            return []
        # Pass the filter_doc_ids down to the search method
        return self.search(query_vector[0], top_k=top_k, filter_doc_ids=filter_doc_ids)

    def get_indexed_document_ids(self) -> List[str]:
        """
        (Sync) Retrieves a list of unique document IDs present in the collection.
        Uses scrolling to efficiently fetch payloads.
        """
        unique_doc_ids = set()
        next_offset = None  # Initialize offset for scrolling

        print(
            f"Fetching unique document IDs from collection: {self.collection_name}..."
        )
        try:
            while True:
                # Scroll through points, fetching only the payload containing 'doc_id'
                results, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=250,  # Adjust batch size as needed
                    offset=next_offset,
                    with_payload=["doc_id"],  # Fetch only the doc_id field
                    with_vectors=False,  # Don't need vectors
                )

                # Extract doc_ids from the current batch
                for hit in results:
                    if hit.payload and "doc_id" in hit.payload:
                        unique_doc_ids.add(hit.payload["doc_id"])

                # If no more results, break the loop
                if not next_offset:
                    break

            return list(unique_doc_ids)

        except Exception as e:
            print(f"Error fetching document IDs: {e}")
            raise

    def delete_documents_by_ids(self, doc_ids_to_delete: List[str]) -> None:
        """
        (Sync) Deletes all points (chunks) associated with the given document IDs.

        Args:
            doc_ids_to_delete (List[str]): A list of document IDs whose points should be deleted.
        """
        if not doc_ids_to_delete:
            print("No document IDs provided for deletion.")
            return

        print(
            f"Attempting to delete points for {len(doc_ids_to_delete)} document IDs..."
        )

        # Construct a filter to match any of the provided doc_ids
        # We use a 'should' filter which acts like an OR condition.
        # If we wanted to match ALL conditions, we'd use 'must'.
        qdrant_filter = models.Filter(
            should=[
                models.FieldCondition(
                    key="doc_id",  # The field in the payload to filter on
                    match=models.MatchValue(value=doc_id),  # Match this specific value
                )
                for doc_id in doc_ids_to_delete
            ]
        )

        try:
            # Perform the delete operation
            response = self.client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_filter,
                wait=True,  # Wait for the operation to complete
            )
            print(f"Qdrant delete operation status: {response.status}")
            if response.status == models.UpdateStatus.COMPLETED:
                print(
                    f"Successfully deleted points for IDs: {', '.join(doc_ids_to_delete)}"
                )
            else:
                print(
                    f"Deletion might not be fully completed. Status: {response.status}"
                )

        except Exception as e:
            print(f"Error deleting points from Qdrant (sync): {e}")
            # Depending on requirements, you might want to raise the exception
