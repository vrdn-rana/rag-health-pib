import os
import sys

# Ensure project root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from google import genai
from src.vector_store import search_similar_chunks

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


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
    2. Dynamically discover active Gemini models for API key.
    3. Generate grounded response using Gemini.
    """
    if not GEMINI_API_KEY:
        return {
            "question": user_question,
            "answer": "Error: GEMINI_API_KEY is not set. Please create a .env file with GEMINI_API_KEY=your_key",
            "source_chunks": []
        }

    # 1. Retrieve relevant chunks from ChromaDB
    chunks = search_similar_chunks(user_question, top_k=top_k)

    # 2. Construct grounded prompt
    prompt = format_context_prompt(user_question, chunks)

    # 3. Initialize Gemini Client
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Dynamically fetch available models from your API key
    candidate_models = []
    try:
        for m in client.models.list():
            if "gemini" in m.name.lower():
                candidate_models.append(m.name)
    except Exception:
        pass

    # Add standard fallback aliases if model listing returns empty
    candidate_models += [
        "models/gemini-2.0-flash",
        "models/gemini-1.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash"
    ]

    answer = None
    last_error = None
    used_model = None

    # Try each available model until one succeeds
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


if __name__ == "__main__":
    test_q = "What is Ayushman Bharat PM-JAY and how many beneficiaries are covered?"
    print(f"Testing Question: '{test_q}'\n")

    result = generate_rag_answer(test_q, top_k=3)

    print("================ LLM ANSWER ================")
    print(result["answer"])
    print(f"\n[Generated using model: {result.get('model_used')}]")
    print("\n================ SOURCES USED ================")
    for chunk in result["source_chunks"]:
        print(f"• [{chunk['chunk_id']}] {chunk['title']}")