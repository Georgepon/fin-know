from fastapi import FastAPI, Form, UploadFile

from app.ingestion import process_document
from app.llm import generate_answer
from app.retriever import get_relevant_chunks
from app.vectorstore import QdrantVectorStore

app = FastAPI()

vectorstore = QdrantVectorStore()


@app.post("/upload")
async def upload_document(file: UploadFile):
    doc_data = process_document(file)
    vectorstore.embed_and_store_chunks(doc_data["chunks"])
    return {"num_chunks": doc_data["num_chunks"], "doc_id": doc_data["doc_id"]}


@app.post("/ask")
async def ask_question(question: str = Form(...)):
    context_chunks = get_relevant_chunks(question, vectorstore)
    context_str = "\n\n---\n\n".join([chunk["text"] for chunk in context_chunks])
    answer = generate_answer(question, context_str)
    return {"answer": answer}
