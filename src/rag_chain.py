import os
import sys

# Ensure project root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.vector_store import search_similar_chunks

# Load local environment if running locally
load_dotenv()


def get_active_api_key() -> str:
    """Dynamically retrieve clean API key from environment."""
    raw_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    return raw_key.strip().strip('"').strip("'")


def format_context(retrieved_chunks: list[dict]) -> str:
    """Formats retrieved vector store chunks into string context."""
    context_str = ""
    for idx, chunk in enumerate(retrieved_chunks, 1):
        context_str += f"--- CONTEXT CHUNK {idx} (Title: {chunk['title']}) ---\n"
        context_str += f"{chunk['content']}\n\n"
    return context_str


def generate_rag_answer(user_question: str, top_k: int = 3) -> dict:
    """
    End-to-end RAG pipeline using LangChain Expression Language (LCEL).
    """
    api_key = get_active_api_key()

    if not api_key:
        return {
            "question": user_question,
            "answer": "Error: GEMINI_API_KEY is missing or empty. Please set GEMINI_API_KEY in Streamlit Secrets.",
            "source_chunks": []
        }

    # 1. Retrieve relevant chunks from ChromaDB
    chunks = search_similar_chunks(user_question, top_k=top_k)
    formatted_context = format_context(chunks)

    # 2. Define LangChain Prompt Template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a precise and helpful AI assistant answering questions about India's Health Transformation.

CRITICAL INSTRUCTIONS:
1. Answer the question using ONLY the provided PIB context snippets below.
2. Do NOT use external general knowledge or invent facts not present in the text.
3. If the provided context does not contain enough information to answer the question, state clearly: "Based on the provided document, I cannot answer this question."
4. Keep the answer clear, short, structured, and directly aligned with the PIB backgrounder document.

--- PROVIDED CONTEXT ---
{context}

--- USER QUESTION ---
{question}

--- GROUNDED ANSWER ---"""
    )

    # 3. List of models to try (fallback strategy for high demand / rate limits)
    candidate_models = [
        "gemini-2.5-flash",
        "gemini-2.0-flash"
    ]

    answer_text = None
    last_error = None
    used_model = None

    # 4. Invoke using pure LangChain Pipe Chain (prompt | llm)
    for model_name in candidate_models:
        try:
            # Initialize LangChain LLM instance
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.2,
                max_retries=3
            )

            # Build LangChain Runnable Chain
            chain = prompt_template | llm

            # Invoke LangChain Runnable
            response = chain.invoke({
                "context": formatted_context,
                "question": user_question
            })

            answer_text = response.content
            if answer_text:
                used_model = model_name
                break
        except Exception as e:
            last_error = str(e)
            continue

    if not answer_text:
        answer_text = f"The service is experiencing high demand. Please try again. (Error details: {last_error})"

    return {
        "question": user_question,
        "answer": str(answer_text).strip(),
        "source_chunks": chunks,
        "model_used": used_model
    }


if __name__ == "__main__":
    test_q = "What is Ayushman Bharat PM-JAY and how many beneficiaries are covered?"
    print(f"Testing Question: '{test_q}'\n")

    result = generate_rag_answer(test_q, top_k=3)

    print("================ LANGCHAIN LLM ANSWER ================")
    print(result["answer"])
    print(f"\n[Generated using model: {result.get('model_used')}]")
    print("\n================ SOURCES USED ================")
    for chunk in result["source_chunks"]:
        print(f"• [{chunk['chunk_id']}] {chunk['title']}")