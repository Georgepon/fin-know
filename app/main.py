from fastapi import FastAPI, Form, UploadFile

from app.ingestion import process_document
from app.llm import generate_answer
from app.retriever import get_relevant_chunks

app = FastAPI()


@app.post("/upload")
async def upload_document(file: UploadFile):
    return process_document(file)


@app.post("/ask")
async def ask_question(question: str = Form(...)):
    context = get_relevant_chunks(question)
    answer = generate_answer(context, question)
    return {"answer": answer}
