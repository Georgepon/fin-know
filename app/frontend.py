import streamlit as st
from ingestion import process_document
from llm import generate_answer
from retriever import get_relevant_chunks
from vectorstore import QdrantVectorStore

if "doc_data" not in st.session_state:
    st.session_state.doc_data = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None

st.set_page_config(page_title=f"Fin-Know: Financial Q&A", layout="wide")
st.title(f"📄 Fin-Know — Ask questions about your financial PDFs")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.last_uploaded_filename:
        with st.spinner("Processing document..."):
            doc_data = process_document(uploaded_file)
            chunks = doc_data["chunks"]

            vectorstore = QdrantVectorStore()
            vectorstore.embed_and_store_chunks(chunks)

            # Store in session
            st.session_state.doc_data = doc_data
            st.session_state.vectorstore = vectorstore
            st.session_state.last_uploaded_filename = uploaded_file.name
    else:
        doc_data = st.session_state.doc_data
        vectorstore = st.session_state.vectorstore

    # Now ready for questions
    question = st.text_input(
        "What do you want to know?", placeholder="e.g. What was the net income in 2024?"
    )

    if question:
        with st.spinner("Thinking..."):
            results = get_relevant_chunks(question, vectorstore)
            answer = generate_answer(question, results)

        st.markdown("### 🧠 Answer")
        st.success(answer)

        with st.expander("📄 Show retrieved context chunks"):
            for i, chunk in enumerate(results):
                st.markdown(f"**Chunk {i+1}**")
                st.code(chunk["text"][:1500], language="markdown")
