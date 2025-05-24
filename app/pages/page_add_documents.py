import hashlib
import json
import os

import streamlit as st
from ingestion import process_document
from vectorstore import QdrantVectorStore

# Define cache file relative to the main app script's location or workspace root.
CACHE_FILE = "processed_cache.json"


def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Cache file {CACHE_FILE} is corrupted. Starting fresh.")
            return {}
    return {}


def save_cache(cache_data: dict) -> None:
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache_data, f)
    except IOError as e:
        print(f"Error saving cache to {CACHE_FILE}: {e}")


def show_add_documents_page():
    st.title("⬆️ Add documents to knowledge base")
    st.markdown(
        """
        Upload your PDF documents here. The system will process them,
        extract text, create embeddings, and store them for querying.
        """
    )
    st.divider()

    processed_files_cache = load_cache()

    if "vectorstore" not in st.session_state:
        try:
            st.session_state.vectorstore = QdrantVectorStore()
            print("Initialized vectorstore on Add Documents page.")
        except Exception as e:
            st.error(f"Failed to initialize vector store connection: {e}")
            st.stop()  # Stop if vectorstore connection fails

    uploaded_file = st.file_uploader(
        "Choose a PDF file to upload and process",
        type=["pdf"],
        key="file_uploader_add_docs",  # Unique key
    )
    status_placeholder_add_docs = st.empty()

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        current_file_hash = hashlib.sha256(file_bytes).hexdigest()
        uploaded_file.seek(0)

        st.write(
            f"**File selected:** {uploaded_file.name} (Hash: `{current_file_hash[:8]}...`)"
        )

        if current_file_hash in processed_files_cache:
            status_placeholder_add_docs.info(
                f"💾 This file ({uploaded_file.name}) has a matching hash in the local cache."
            )
            st.warning(
                "This document content (based on its hash) appears to be in the local cache. "
                "If you recently cleared the vector store and want to re-process, ensure `processed_cache.json` is also cleared and restart the app."
            )
        else:
            st.info(
                f"🚀 New file detected ({uploaded_file.name}). Starting processing..."
            )
            progress_bar_placeholder = st.empty()
            with st.status(
                f"Processing {uploaded_file.name}...", expanded=True
            ) as status_bar:
                try:
                    if (
                        "vectorstore" not in st.session_state
                        or st.session_state.vectorstore is None
                    ):
                        st.session_state.vectorstore = QdrantVectorStore()
                        print(
                            "Re-initialized vectorstore before processing on Add Docs page."
                        )

                    status_bar.write("📄 Extracting text and chunking document...")
                    doc_data = process_document(uploaded_file)
                    chunks = doc_data["chunks"]
                    new_doc_id = doc_data["doc_id"]
                    num_chunks = doc_data["num_chunks"]
                    status_bar.write(f"Found {num_chunks} text chunks.")

                    status_bar.write(
                        f"🧠 Embedding {num_chunks} chunks and storing in Qdrant..."
                    )
                    progress_bar = progress_bar_placeholder.progress(
                        0, text="Embedding 0%..."
                    )
                    st.session_state.vectorstore.embed_and_store_chunks(
                        chunks, progress_bar=progress_bar
                    )
                    progress_bar_placeholder.empty()

                    processed_files_cache[current_file_hash] = new_doc_id
                    save_cache(processed_files_cache)

                    status_bar.update(
                        label=f"✅ Processing successful!",
                        state="complete",
                        expanded=False,
                    )
                    status_placeholder_add_docs.success(
                        f"✅ Successfully processed {uploaded_file.name}"
                    )

                except Exception as e:
                    progress_bar_placeholder.empty()
                    status_bar.update(
                        label=f"❌ Error processing", state="error", expanded=True
                    )
                    st.error(f"Error during processing: {e}")
                    status_placeholder_add_docs.error("Processing failed.")
    else:
        status_placeholder_add_docs.info("Upload a new PDF file above.")

    st.divider()
    st.caption(f"Note: Processed file status is tracked locally in `{CACHE_FILE}`.")
