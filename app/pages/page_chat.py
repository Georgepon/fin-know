import streamlit as st
from llm import generate_chat_response


def show_chat_page():
    st.title("💬 Chat with the AI")
    st.markdown(
        "Chat directly with the AI without any document context. This is a direct conversation without RAG."
    )

    st.divider()

    st.header("Chat with the AI")

    question = st.text_input(
        "What would you like to discuss?",
        placeholder="e.g. Tell me about yourself",
        key="question_input_chat",  # Unique key
    )

    if question:
        with st.spinner("🤔 Thinking..."):
            try:
                # Direct chat without context
                answer = generate_chat_response(question)

                st.markdown("### 🧠 Response")
                with st.chat_message("ai"):
                    st.write(answer)

            except Exception as e:
                st.error(f"Error during chat: {e}")
