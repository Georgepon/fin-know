# fin-know
Fin-Know is a generative AI application which reads through companies' financial statements, and can converse on them with the user.

## Features
- Upload PDFs
- Ask questions
- Uses Retrieval-Augmented Generation (RAG)

## Tech Stack
- FastAPI
- LangChain
- FAISS
- OpenAI API

## Run it
```bash
uvicorn app.main:app --reload
```