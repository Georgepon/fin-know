import json
import os

import streamlit as st

# Use relative imports if running as a module (python -m streamlit run app/frontend.py)
# Or keep absolute if running streamlit run app/frontend.py from parent dir
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


# Callback function to clear the question input
def clear_question_input():
    if "question_input" in st.session_state:
        st.session_state.question_input = ""


# Helper function to get available documents for selection
def get_available_docs(vectorstore: QdrantVectorStore, current_cache: dict) -> dict:
    """Gets indexed doc IDs and maps them to known hashes from cache."""
    doc_options = {}
    try:
        indexed_ids = vectorstore.get_indexed_document_ids()
        id_to_hash = {v: k for k, v in current_cache.items()}
        for doc_id in indexed_ids:
            file_hash = id_to_hash.get(doc_id)
            if file_hash:
                doc_options[doc_id] = f"Doc (hash: {file_hash[:8]}...)"
            else:
                doc_options[doc_id] = f"Doc (ID: {doc_id[:8]}... - No cache match)"
    except Exception as e:
        st.warning(f"Could not retrieve indexed documents: {e}")
    return doc_options


# Load cache at the start
processed_files_cache = load_cache()


# --- Streamlit App (Query Page) ---

st.set_page_config(page_title="Query Documents", layout="wide")
st.title("💬 Query Processed Documents")
st.markdown("Select documents from the sidebar and ask questions below.")

# Initialize session state keys if they don't exist
# Ensure vectorstore is initialized here as well, as this might be the first page visited
if "vectorstore" not in st.session_state:
    try:
        st.session_state.vectorstore = QdrantVectorStore()
        print("Initialized vectorstore on Query page.")
    except Exception as e:
        st.error(f"Failed to initialize vector store connection: {e}")
        st.session_state.vectorstore = None  # Ensure it's None on failure

if "selected_doc_ids" not in st.session_state:
    st.session_state.selected_doc_ids = []
if "available_docs" not in st.session_state:
    st.session_state.available_docs = {}
    # Attempt to populate available_docs on first load if vectorstore initialized
    if st.session_state.vectorstore:
        st.session_state.available_docs = get_available_docs(
            st.session_state.vectorstore, processed_files_cache
        )


# --- Sidebar for Document Selection ---
with st.sidebar:
    st.header("Document Selection")
    # Refresh available documents list
    # This ensures it's up-to-date if a file was just uploaded on the other page
    if st.session_state.vectorstore:
        st.session_state.available_docs = get_available_docs(
            st.session_state.vectorstore, processed_files_cache
        )

    if st.session_state.vectorstore and st.session_state.available_docs:
        options = list(st.session_state.available_docs.keys())
        labels = [st.session_state.available_docs[id] for id in options]

        # Preserve selection across page loads if possible
        current_selection = [
            id for id in st.session_state.selected_doc_ids if id in options
        ]

        selected_ids = st.multiselect(
            label="Choose documents to search within:",
            options=options,
            format_func=lambda id: st.session_state.available_docs.get(id, id),
            default=current_selection,
            key="doc_selector",
            on_change=clear_question_input,
        )
        st.session_state.selected_doc_ids = selected_ids
        st.caption(f"{len(selected_ids)} document(s) selected.")
    elif st.session_state.vectorstore:
        st.info(
            "No processed documents found. Please upload documents via the 'Upload Documents' page."
        )
    else:
        st.warning("Vector store connection not available.")

# --- Main Area: Question Answering ---

st.divider()

if not st.session_state.vectorstore:
    st.error(
        "Cannot connect to the vector database. Please check configuration and ensure Qdrant is running."
    )
elif not st.session_state.available_docs:
    st.info(
        "No documents available to query. Please use the 'Upload Documents' page to process files first."
    )
else:
    st.header(f"Ask questions about selected documents")
    if st.session_state.selected_doc_ids:
        selected_labels = [
            st.session_state.available_docs.get(id, id[:8])
            for id in st.session_state.selected_doc_ids
        ]
        st.caption(f"Searching within: {', '.join(selected_labels)}")
    else:
        st.caption("No documents selected in the sidebar to search within.")

    question = st.text_input(
        "What do you want to know?",
        placeholder="e.g. What was the total revenue?",
        key="question_input",
        disabled=not st.session_state.selected_doc_ids,
    )

    if question and st.session_state.selected_doc_ids:
        with st.spinner("🤔 Thinking..."):
            try:
                vectorstore_instance = st.session_state.vectorstore
                results = get_relevant_chunks(
                    question,
                    vectorstore_instance,
                    filter_doc_ids=st.session_state.selected_doc_ids,
                )
                if results:
                    context_str = "\n\n---\n\n".join(
                        [chunk["text"] for chunk in results]
                    )
                    answer = generate_answer(question, context_str)

                    st.markdown("### 🧠 Answer")
                    with st.chat_message("ai"):
                        st.write(answer)

                    with st.expander("📄 Show retrieved context chunks"):
                        for i, chunk in enumerate(results):
                            doc_label = st.session_state.available_docs.get(
                                chunk.get("doc_id"), chunk.get("doc_id", "N/A")[:8]
                            )
                            st.info(f"**Chunk {i+1}** (Source: `{doc_label}`)")
                            st.text_area(
                                f"chunk_{i}",
                                chunk["text"],
                                height=150,
                                disabled=True,
                                label_visibility="collapsed",
                            )
                else:
                    st.warning(
                        "Could not retrieve relevant context from the selected documents for this question."
                    )

            except Exception as e:
                st.error(f"Error during question answering: {e}")
    elif question and not st.session_state.selected_doc_ids:
        st.warning(
            "Please select at least one document in the sidebar to search within."
        )
