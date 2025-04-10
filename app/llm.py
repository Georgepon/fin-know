import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1"
)


def generate_answer(
    question: str,
    context: str,
    model: str = "llama3-8b-8192",
    max_tokens: int = 512,
) -> str:
    """
    Generate an answer using Groq-hosted LLM based on a question and retrieved document context.

    Args:
        question (str): The user question.
        context (str): Concatenated relevant chunks.
        model (str): Groq model to use.
        max_tokens (int): Max tokens to generate.

    Returns:
        str: The generated answer.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions using the provided document context. Only use the context below to answer the question.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}",
            },
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
