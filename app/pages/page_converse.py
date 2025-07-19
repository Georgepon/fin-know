import streamlit as st
from llm import generate_answer
from retriever import get_relevant_chunks
from vectorstore import QdrantVectorStore


def show_converse_page():
    st.title("💬 Chat with the RAG-Powered AI")
    st.markdown(
        "Ask questions about your documents below. The search will cover all processed documents."
    )

    # Initialize vectorstore in session state if it doesn't exist
    if "vectorstore" not in st.session_state:
        try:
            st.session_state.vectorstore = QdrantVectorStore()
            print("Initialized vectorstore on Converse page.")
        except Exception as e:
            st.error(f"Failed to initialize vector store connection: {e}")
            st.session_state.vectorstore = None

    st.divider()

    if not st.session_state.get("vectorstore"):  # Use .get for safer access
        st.error(
            "Cannot connect to the vector database. Please check configuration and ensure Qdrant is running."
        )
    else:
        # Show which documents are indexed
        docs = []
        try:
            docs = st.session_state.vectorstore.get_indexed_documents()
        except Exception as e:
            st.warning(f"Unable to fetch document list: {e}")

        with st.expander("📚 Documents currently indexed"):
            if docs:
                for doc_id, filename in docs:
                    st.markdown(f"- `{filename}` (`{doc_id[:8]}...`)")
            else:
                st.caption("No documents found in the vector store.")

        st.header("Ask questions about your documents")

        question = st.text_input(
            "What do you want to know?",
            placeholder="e.g. What was the total revenue?",
            key="question_input_converse",  # Unique key
        )

        if question:
            with st.spinner("🤔 Thinking..."):
                try:
                    vectorstore_instance = st.session_state.vectorstore
                    results = get_relevant_chunks(
                        question,
                        vectorstore_instance,
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
                                file_title = chunk.get("filename", "Unknown Source")
                                doc_id_val = chunk.get("doc_id", "N/A")

                                display_label = file_title
                                if (
                                    file_title == "Unknown Source"
                                    and doc_id_val != "N/A"
                                ):
                                    display_label = (
                                        f"{file_title} (ID: {doc_id_val[:8]}...)"
                                    )
                                elif file_title != "Unknown Source":
                                    display_label = file_title
                                else:
                                    display_label = "Source N/A"

                                st.info(f"**Chunk {i+1}** (Source: `{display_label}`)")
                                st.text_area(
                                    f"chunk_converse_{i}",  # Unique key
                                    chunk["text"],
                                    height=150,
                                    disabled=True,
                                    label_visibility="collapsed",
                                )
                    else:
                        st.warning(
                            "Could not retrieve relevant context for this question."
                        )
                except Exception as e:
                    st.error(f"Error during question answering: {e}")
