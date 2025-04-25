import hashlib
import json
import os
from io import BytesIO

import streamlit as st
from ingestion import process_document
from llm import generate_answer
from retriever import get_relevant_chunks
from vectorstore import QdrantVectorStore

# --- Cache Setup ---
CACHE_FILE = "processed_cache.json"


def load_cache() -> dict:
    """Loads the processed file hash cache from a JSON file."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Cache file {CACHE_FILE} is corrupted. Starting fresh.")
            return {}
    return {}


def save_cache(cache_data: dict) -> None:
    """Saves the processed file hash cache to a JSON file."""
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache_data, f)
    except IOError as e:
        print(f"Error saving cache to {CACHE_FILE}: {e}")


# Load cache at the start
processed_files_cache = load_cache()


# --- Streamlit App ---

# Initialize session state keys if they don't exist
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None  # Will be re-initialized if needed
if "current_file_hash" not in st.session_state:
    st.session_state.current_file_hash = None
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

st.set_page_config(page_title=f"Fin-Know: Financial Q&A", layout="wide")
st.title(f"📄 Fin-Know — Ask questions about your financial PDFs")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    # Read file content and calculate hash
    file_bytes = uploaded_file.getvalue()  # Read bytes
    current_file_hash = hashlib.sha256(file_bytes).hexdigest()
    uploaded_file.seek(0)  # Reset pointer for potential reprocessing

    # --- Cache Check & Processing Logic ---
    if (
        current_file_hash != st.session_state.current_file_hash
        or not st.session_state.file_processed
    ):
        st.session_state.file_processed = False
        st.session_state.vectorstore = None  # Reset vectorstore instance
        st.session_state.doc_id = None

        if current_file_hash in processed_files_cache:
            # --- Cache Hit ---
            st.info(f"💾 Found cached version for {uploaded_file.name}.")
            st.session_state.doc_id = processed_files_cache[current_file_hash]
            try:
                # Initialize vectorstore for querying
                st.session_state.vectorstore = QdrantVectorStore()
            except Exception as e:
                st.error(f"Error initializing vector store for cached file: {e}")
                st.stop()

            st.session_state.current_file_hash = current_file_hash
            st.session_state.file_processed = True

        else:
            # --- Cache Miss ---
            # Create a placeholder for the progress bar
            progress_bar_placeholder = st.empty()

            with st.spinner(f"Processing {uploaded_file.name} (synchronously)..."):
                # Create the actual progress bar inside the placeholder
                progress_bar = progress_bar_placeholder.progress(
                    0, text="Starting embedding..."
                )
                try:
                    doc_data = process_document(uploaded_file)
                    chunks = doc_data["chunks"]
                    st.session_state.doc_id = doc_data["doc_id"]

                    # Initialize and use vectorstore synchronously
                    st.session_state.vectorstore = QdrantVectorStore()
                    # Call synchronous method, passing the progress bar
                    st.session_state.vectorstore.embed_and_store_chunks(
                        chunks, progress_bar=progress_bar
                    )

                    # Update and save cache
                    processed_files_cache[current_file_hash] = st.session_state.doc_id
                    save_cache(processed_files_cache)
                    # Remove spinner and progress bar on success (optional, or keep bar at 100%)
                    # progress_bar_placeholder.empty()
                    st.success(
                        f"✅ Processed and stored {uploaded_file.name} (synchronously)."
                    )
                    st.session_state.current_file_hash = current_file_hash
                    st.session_state.file_processed = True

                except Exception as e:
                    # Remove progress bar on error
                    progress_bar_placeholder.empty()
                    st.error(f"Error processing document synchronously: {e}")
                    st.session_state.file_processed = False
                    st.stop()
                # finally:
                # Ensure progress bar is removed or set to final state regardless of success/error
                # progress_bar_placeholder.empty()

    # --- Question Answering Section (only if a doc is loaded/cached) ---
    if (
        st.session_state.doc_id
        and st.session_state.vectorstore
        and st.session_state.file_processed
    ):
        st.markdown(
            f"**Ready to answer questions about: {uploaded_file.name}** (Doc ID: `{st.session_state.doc_id}`)"
        )

        question = st.text_input(
            "What do you want to know?",
            placeholder="e.g. What was the net income in 2024?",
        )

        if question:
            with st.spinner("Thinking..."):
                try:
                    vectorstore_instance = st.session_state.vectorstore
                    if vectorstore_instance:
                        # Call synchronous retrieval function
                        results = get_relevant_chunks(question, vectorstore_instance)

                        # LLM call remains sync
                        context_str = "\n\n---\n\n".join(
                            [chunk["text"] for chunk in results]
                        )
                        answer = generate_answer(question, context_str)

                        st.markdown("### 🧠 Answer")
                        st.success(answer)

                        with st.expander("📄 Show retrieved context chunks"):
                            for i, chunk in enumerate(results):
                                st.markdown(
                                    f"**Chunk {i+1}** (Source Doc: `{chunk.get('doc_id', 'N/A')}`)"
                                )
                                st.code(chunk["text"][:1500], language="markdown")
                    else:
                        st.error("Vectorstore not available for question answering.")
                except Exception as e:
                    st.error(f"Error during question answering: {e}")
    elif uploaded_file:
        # Handle cases where processing might have failed before reaching Q&A
        if not st.session_state.file_processed:
            st.warning(
                "File processing did not complete successfully. Cannot answer questions."
            )
