# app/llm.py

from typing import List

from llama_cpp import Llama

# Load model once when this module is imported
llm = Llama(
    model_path="models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=4096,
    n_threads=8,
    verbose=True,
)


def generate_answer(question: str, chunks: List[dict]) -> str:
    """
    Generate an answer using the local LLM based on retrieved document chunks.

    Args:
        question (str): The user's input question.
        chunks (List[dict]): List of retrieved document chunks (each has "text").

    Returns:
        str: Generated answer.
    """
    # Combine context into a single string
    context = "\n\n".join(chunk["text"] for chunk in chunks)

    # Build the full prompt
    prompt = f"""
        You are a helpful assistant with access to financial documents. Use the context below to answer the user's question. 
        If the answer is not in the context, say "The answer is not found in the document."

        Context:
        {context}

        Question: {question}
        Answer:
    """.strip()

    # Run the model
    print("⚙️ Sending prompt to LLM...")

    response = llm(prompt, max_tokens=512, stop=["Question:", "Context:"])

    print("✅ Response received from LLM")

    return response["choices"][0]["text"].strip()
