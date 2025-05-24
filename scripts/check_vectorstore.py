import json
import os

from app.vectorstore import QdrantVectorStore

CACHE_FILE = "processed_cache.json"


def load_cache() -> dict:
    """Loads the processed file hash cache from a JSON file."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                # Invert the cache for easy lookup: doc_id -> hash
                original_cache = json.load(f)
                inverted_cache = {v: k for k, v in original_cache.items()}
                return inverted_cache
        except json.JSONDecodeError:
            print(f"Warning: Cache file {CACHE_FILE} is corrupted.")
            return {}
        except Exception as e:
            print(f"Error loading or inverting cache: {e}")
            return {}
    return {}


# --- Main logic ---
try:
    # Initialize Vector Store
    vectorstore = QdrantVectorStore()

    # Get indexed document IDs from Qdrant
    indexed_doc_ids = vectorstore.get_indexed_document_ids()
    print(f"\nDocument IDs found in Qdrant: {len(indexed_doc_ids)}")

    # Load and invert the cache (doc_id -> hash)
    id_to_hash_cache = load_cache()
    print(f"Cache entries loaded (doc_id -> hash): {len(id_to_hash_cache)}")

    # Find corresponding hashes for indexed documents
    found_hashes = []
    missing_in_cache = []
    if indexed_doc_ids:
        print("\nComparing Qdrant IDs with Cache...")
        for doc_id in indexed_doc_ids:
            if doc_id in id_to_hash_cache:
                found_hashes.append(id_to_hash_cache[doc_id])
            else:
                missing_in_cache.append(doc_id)

        print(f"\n--- Files Found in Vector Store (by Hash) ---")
        if found_hashes:
            for file_hash in found_hashes:
                print(f"- Hash: {file_hash}")
        else:
            print("No matching hashes found in the cache for documents in Qdrant.")

        if missing_in_cache:
            print("\n--- Document IDs in Qdrant but NOT in Cache ---")
            for doc_id in missing_in_cache:
                print(f"- ID: {doc_id}")

    else:
        print("\nQdrant collection appears to be empty or no IDs were retrieved.")

except ValueError as e:
    print(f"Configuration Error: {e}")
    print(
        "Please ensure QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, and OPENAI_API_KEY are set in your environment or .env file."
    )
except Exception as e:
    print(f"An unexpected error occurred: {e}")
