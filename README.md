# FinKnow

## Summary and Purpose
FinKnow is a small reference application that demonstrates how to build a Retrieval-Augmented Generation (RAG) workflow for financial documents. Users can upload PDF statements, embed their contents and query them using a language model. The project also exposes basic API endpoints for programmatic access.

## Pages and Features
- **RAG-Powered Q&A** – ask questions about uploaded PDFs with answers generated from retrieved document chunks.
- **General Q&A** – chat directly with the language model without document context.
- **Add Documents** – upload and process PDF files. Extracted chunks are embedded and stored in Qdrant. A local cache tracks processed files.

Additional features include chunking of documents with LangChain, OpenAI embeddings, and a simple FastAPI backend.

## Tech Stack
- **Python & Streamlit** for the user interface.
- **FastAPI** providing `/upload` and `/ask` endpoints.
- **OpenAI Embeddings** for vector generation.
- **Qdrant** as the vector database.
- **Groq** LLM API (llama3-8b-8192) for answering questions.
- **LangChain** utilities for document chunking.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Provide the following environment variables (e.g. in a `.env` file):
   - `OPENAI_API_KEY` – API key for OpenAI embeddings
   - `GROQ_API_KEY` – API key for Groq LLM
   - `QDRANT_URL` – URL of your Qdrant instance
   - `QDRANT_API_KEY` – API key for Qdrant (if needed)
   - `QDRANT_COLLECTION` – collection name to store embeddings
3. Start the Streamlit interface:
   ```bash
   streamlit run app/frontend.py
   ```
   The optional FastAPI service can be launched with:
   ```bash
   uvicorn app.main:app --reload
   ```
