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
    """Generate an answer using the Groq-hosted LLM.

    Args:
        question: The user question.
        context: Concatenated relevant chunks.
        model: Groq model to use.
        max_tokens: Maximum tokens to generate.

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


def generate_chat_response(
    message: str,
    model: str = "llama3-8b-8192",
    max_tokens: int = 512,
) -> str:
    """Generate a chat response without document context.

    Args:
        message: The user's message.
        model: Groq model to use.
        max_tokens: Maximum tokens to generate.

    Returns:
        str: The generated response.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful and friendly AI assistant. Provide direct and conversational responses to the user's messages.",
            },
            {
                "role": "user",
                "content": message,
            },
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
