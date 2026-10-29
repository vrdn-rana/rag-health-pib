import os
import sys

# Ensure project root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from src.vector_store import search_similar_chunks

# Load local environment if running locally
load_dotenv()


def get_active_api_key() -> str:
    """Dynamically retrieve clean API key from environment or secrets."""
    raw_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    return raw_key.strip().strip('"').strip("'")


def format_context_prompt(user_question: str, retrieved_chunks: list[dict]) -> str:
    """Formats the retrieved chunks into a clean prompt context."""
    context_str = ""
    for idx, chunk in enumerate(retrieved_chunks, 1):
        context_str += f"--- CONTEXT CHUNK {idx} (Title: {chunk['title']}) ---\n"
        context_str += f"{chunk['content']}\n\n"

    prompt = f"""You are a precise and helpful AI assistant answering questions about India's Health Transformation.

CRITICAL INSTRUCTIONS:
1. Answer the question using ONLY the provided PIB context snippets below.
2. Do NOT use external general knowledge or invent facts not present in the text.
3. If the provided context does not contain enough information to answer the question, state clearly: "Based on the provided document, I cannot answer this question."
4. Keep the answer clear, short, structured, and directly aligned with the PIB backgrounder document.

--- PROVIDED CONTEXT ---
{context_str}

--- USER QUESTION ---
{user_question}

--- GROUNDED ANSWER ---"""
    
    return prompt


def generate_rag_answer(user_question: str, top_k: int = 3) -> dict:
    """
    End-to-end RAG pipeline:
    1. Search top_k chunks from ChromaDB.
    2. Generate grounded response using Gemini.
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

    # 2. Construct grounded prompt
    prompt = format_context_prompt(user_question, chunks)

    # 3. Initialize Gemini Client with explicit API key
    client = genai.Client(api_key=api_key)
    
    # Target models
    candidate_models = [
        "gemini-2.0-flash",
        "gemini-1.5-flash"
    ]

    answer = None
    last_error = None
    used_model = None

    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            answer = response.text
            if answer:
                used_model = model_name
                break
        except Exception as e:
            last_error = str(e)
            continue

    if not answer:
        answer = f"Error generating answer with Gemini API: {last_error}"

    return {
        "question": user_question,
        "answer": answer.strip(),
        "source_chunks": chunks,
        "model_used": used_model
    }