import json
import os

import streamlit as st
from llm import generate_answer
from pages import page_add_documents, page_chat, page_converse
from retriever import get_relevant_chunks
from vectorstore import QdrantVectorStore

# Set the overall app configuration
st.set_page_config(
    page_title="Fin-Know Assistant", layout="wide", initial_sidebar_state="expanded"
)

# Define the pages for st.navigation
pg = st.navigation(
    {
        "Chat": [
            st.Page(
                page_converse.show_converse_page,
                title="RAG-Powered Q&A",
                icon="💬",
            ),
            st.Page(
                page_chat.show_chat_page,
                title="General Q&A",
                icon="🤖",
            ),
        ],
        "Manage Documents": [
            st.Page(
                page_add_documents.show_add_documents_page,
                title="Add documents to knowledge base",
                icon="➕",
            )
        ],
    }
)

# Run the selected page
pg.run()
