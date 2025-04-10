import streamlit as st

from app.ingestion import process_document
from app.llm import generate_answer
from app.retriever import get_relevant_chunks
from app.vectorstore import VectorStore

st.set_page_config(page_title=f"Fin-Know: Financial Q&A", layout="wide")
st.title(f"📄 Fin-Know — Ask questions about your financial PDFs")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing document..."):
        doc_data = process_document(uploaded_file)
        chunks = doc_data["chunks"]

        vectorstore = VectorStore()
        vectorstore.embed_and_store_chunks(chunks)

    # Question input
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
                st.code(chunk["text"][:1500], language="markdown")  # Limit for sanity
