import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


def reset_qdrant_collection():
    """Recreate the configured Qdrant collection.

    This deletes all existing data and starts the collection fresh.
    """
    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION")

    if not all([qdrant_url, collection_name]):
        print(
            "Error: QDRANT_URL and QDRANT_COLLECTION environment variables must be set."
        )
        print("QDRANT_API_KEY is optional if your Qdrant instance does not require it.")
        return

    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=30.0)
        print(f"Connected to Qdrant at {qdrant_url}.")

        # Define the vector parameters based on your project's configuration
        vector_size = 1536  # From your app/vectorstore.py
        distance_metric = models.Distance.COSINE  # From your app/vectorstore.py

        print(f"Attempting to delete and recreate collection: '{collection_name}'...")

        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size, distance=distance_metric
            ),
        )

        print(f"Collection '{collection_name}' has been successfully reset.")
        print("All previous data in this collection has been deleted.")

    except Exception as e:
        print(f"An error occurred while resetting the collection: {e}")


if __name__ == "__main__":
    print("This script will delete all data in your Qdrant collection and recreate it.")
    confirm = input("Are you sure you want to proceed? (yes/no): ")
    if confirm.lower() == "yes":
        reset_qdrant_collection()
    else:
        print("Operation cancelled by user.")
