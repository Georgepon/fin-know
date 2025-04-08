import os

from app.ingestion import process_document
from app.llm import generate_answer
from app.retriever import get_relevant_chunks
from app.vectorstore import VectorStore

# ---- CONFIG ----
pdf_path = "example_docs/2024-lbg-annual-report.pdf"
question = "What was the net income reported by Lloyds in 2024?"

# ---- STEP 1: Parse & Chunk ----
print("🔍 Parsing and chunking PDF...")
doc_data = process_document(pdf_path)
chunks = doc_data["chunks"]

# ---- STEP 2: Embed and store chunks ----
print("📦 Storing chunks in vectorstore...")
vectorstore = VectorStore()
vectorstore.embed_and_store_chunks(chunks)

# ---- STEP 3: Retrieve relevant chunks ----
print("🧠 Retrieving top chunks for question...")
results = get_relevant_chunks(question, vectorstore, top_k=3)

# Optional: print retrieved chunks
print("\n--- Retrieved Chunks ---")
for i, chunk in enumerate(results):
    print(f"\nChunk {i+1}:\n{chunk['text'][:500]}...")  # Truncate for readability

# ---- STEP 4: Generate answer using local LLM ----
print("\n💬 Generating answer...")
answer = generate_answer(question, results)

print("\n✅ Final Answer:\n", answer)
